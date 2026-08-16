from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import glob
import urllib.request
import json
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import ee
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="PM2.5 Forecast API (SVR vs ARIMA)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 0. เริ่มต้น Google Earth Engine & ดึงจุดความร้อน (Hotspots)
# ==============================================================================
gee_available = False
try:
    ee.Initialize()
    print("✓ Google Earth Engine initialized successfully")
    gee_available = True
except Exception as e:
    print("⚠️ GEE Initialization warning:", e)

def fetch_gee_hotspots(start_date_str, end_date_str):
    if not gee_available:
        return {}
    try:
        phayao_roi = ee.Geometry.Point([99.9018, 19.1662]).buffer(50000)
        firms = ee.ImageCollection('FIRMS') \
                  .filterDate(start_date_str, end_date_str) \
                  .filterBounds(phayao_roi)
        
        def count_fire(image):
            fire_mask = image.select('T21').gt(310)
            stats = fire_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=phayao_roi,
                scale=1000,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'hotspots': stats.get('T21')
            })
            
        fire_features = firms.map(count_fire).getInfo()
        hotspot_map = {}
        for feat in fire_features.get('features', []):
            props = feat.get('properties', {})
            d = props.get('date')
            c = props.get('hotspots', 0)
            hotspot_map[d] = c if c is not None else 0
        return hotspot_map
    except Exception as e:
        print("⚠️ ไม่สามารถดึงข้อมูล Hotspot จาก GEE ได้:", e)
        return {}

# ==============================================================================
# 1. โหลดข้อมูลฝุ่น PM2.5, สภาพอากาศ และจุดความร้อน
# ==============================================================================
print("1. Loading PM2.5 Data...")

files = sorted(glob.glob('สนามกีฬาจังหวัดพะเยา*.xlsx'))
if not files:
    print("❌ ไม่พบไฟล์ Excel 'สนามกีฬาจังหวัดพะเยา*.xlsx'")

dfs = [pd.read_excel(f, skiprows=2).dropna() for f in files]
for df in dfs:
    df.columns = ['Date', 'PM25']

df_ground = pd.concat(dfs, ignore_index=True)
df_ground['Date'] = pd.to_datetime(df_ground['Date'], format='%d/%m/%Y', errors='coerce')
df_ground['PM25'] = pd.to_numeric(df_ground['PM25'], errors='coerce')
df_ground = df_ground.dropna().sort_values('Date').set_index('Date')

df_daily = df_ground[~df_ground.index.duplicated(keep='first')].asfreq('D').interpolate(method='time')
full_idx = pd.date_range(start=df_daily.index.min(), end='2026-12-31', freq='D')
df_full = df_daily.reindex(full_idx)

# Open-Meteo Weather Data
start_date = df_daily.index.min().strftime('%Y-%m-%d')
url = f"https://archive-api.open-meteo.com/v1/archive?latitude=19.167&longitude=99.900&start_date={start_date}&end_date=2026-12-31&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean&timezone=Asia%2FBangkok"

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    df_weather = pd.DataFrame(data['daily'])
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.set_index('time')
    df_full = df_full.join(df_weather, how='left')
    use_weather = True
except Exception:
    use_weather = False

if use_weather:
    df_full = df_full.groupby(df_full.index.month).transform(lambda x: x.fillna(x.mean()))

# GEE Hotspots Data
hotspot_data = fetch_gee_hotspots(start_date, '2026-12-31')
if hotspot_data:
    df_full['hotspots'] = df_full.index.strftime('%Y-%m-%d').map(hotspot_data).fillna(0)
    use_hotspots = True
else:
    use_hotspots = False

# ==============================================================================
# 2. Feature Engineering (ปรับปรุงเพิ่ม Cyclical Features - [ข้อ 3])
# ==============================================================================
df_model = df_full.copy()
df_model['Month'] = df_model.index.month
df_model['DayOfWeek'] = df_model.index.dayofweek
df_model['DayOfYear'] = df_model.index.dayofyear
df_model['Is_High_Season'] = df_model['Month'].isin([1, 2, 3, 4]).astype(int)

# เพิ่ม Cyclical Features (Sin/Cos) ช่วยเพิ่มความแม่นยำทางฤดูกาล
df_model['Sin_DayOfYear'] = np.sin(2 * np.pi * df_model['DayOfYear'] / 365.25)
df_model['Cos_DayOfYear'] = np.cos(2 * np.pi * df_model['DayOfYear'] / 365.25)

df_model['Lag1'] = df_model['PM25'].shift(1)
df_model['Lag2'] = df_model['PM25'].shift(2)
df_model['Lag3'] = df_model['PM25'].shift(3)
df_model['Lag7'] = df_model['PM25'].shift(7)
df_model['PM25_Diff1'] = df_model['Lag1'] - df_model['Lag2']
df_model['Rolling7_Mean'] = df_model['PM25'].shift(1).rolling(window=7).mean()
df_model['Rolling14_Mean'] = df_model['PM25'].shift(1).rolling(window=14).mean()
df_model['EWMA7'] = df_model['PM25'].shift(1).ewm(span=7).mean()

if use_weather:
    df_model['Temp_Range'] = df_model['temperature_2m_max'] - df_model['temperature_2m_min']
    df_model['Wind_x_Humidity'] = df_model['wind_speed_10m_max'] * df_model['relative_humidity_2m_mean']

if use_hotspots:
    df_model['Hotspot_Lag1'] = df_model['hotspots'].shift(1)
    df_model['Hotspot_Roll3'] = df_model['hotspots'].shift(1).rolling(window=3).mean()

for i in range(1, 8):
    df_model[f'Target_t{i}'] = df_model['PM25'].shift(-i)

features = ['Month', 'DayOfWeek', 'DayOfYear', 'Is_High_Season', 'Sin_DayOfYear', 'Cos_DayOfYear', 'Lag1', 'Lag2', 'Lag3', 'Lag7', 'PM25_Diff1', 'Rolling7_Mean', 'Rolling14_Mean', 'EWMA7']
if use_weather:
    features += ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'Temp_Range', 'precipitation_sum', 'wind_speed_10m_max', 'relative_humidity_2m_mean', 'Wind_x_Humidity']
if use_hotspots:
    features += ['hotspots', 'Hotspot_Lag1', 'Hotspot_Roll3']

df_train = df_model.dropna(subset=features + [f'Target_t{i}' for i in range(1, 8)])

# ==============================================================================
# 3. เทรนและเปรียบเทียบ 2 โมเดล (SVR vs ARIMA) - [ปรับแต่งเพิ่มความแม่นยำ]
# ==============================================================================
mean_pm = float(df_train['PM25'].mean())
model_results = {}
trained_models = {}

print("\n" + "="*60)
print("📊 กำลังเปรียบเทียบประสิทธิภาพโมเดล (SVR vs ARIMA)...")
print("="*60)

# 3.1 SVR Model (ใช้ RobustScaler + จูน C=150.0 และ epsilon=0.01)
svr_models = {}
svr_mae_list = []
for i in range(1, 8):
    svr_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler(),
        SVR(kernel='rbf', C=150.0, gamma='scale', epsilon=0.01)
    )
    svr_pipe.fit(df_train[features], df_train[f'Target_t{i}'])
    svr_models[f't{i}'] = svr_pipe
    
    preds = svr_pipe.predict(df_train[features])
    mae = mean_absolute_error(df_train[f'Target_t{i}'], preds)
    svr_mae_list.append(mae)

svr_avg_mae = round(float(np.mean(svr_mae_list)), 2)
svr_acc = round(max(0.0, (1.0 - (svr_avg_mae / mean_pm)) * 100.0), 1)
model_results["SVR"] = {"mae": svr_avg_mae, "accuracy": svr_acc}
trained_models["SVR"] = svr_models

print(f"🔹 SVR             | MAE: ±{svr_avg_mae:.2f} µg/m³ | Accuracy: {svr_acc:.1f}%")

# 3.2 ARIMA Model (ปรับ order=(7, 1, 1) - [ข้อ 2])
try:
    ts_data = df_daily['PM25'].dropna()
    arima_fit = ARIMA(ts_data, order=(7, 1, 1)).fit()
    arima_preds = arima_fit.fittedvalues
    
    arima_mae = round(float(mean_absolute_error(ts_data[1:], arima_preds[1:])), 2)
    arima_acc = round(max(0.0, (1.0 - (arima_mae / mean_pm)) * 100.0), 1)
    
    model_results["ARIMA"] = {"mae": arima_mae, "accuracy": arima_acc}
    trained_models["ARIMA"] = arima_fit
    
    print(f"🔹 ARIMA           | MAE: ±{arima_mae:.2f} µg/m³ | Accuracy: {arima_acc:.1f}%")
except Exception as e:
    print("⚠️ เกิดข้อผิดพลาดในการเทรน ARIMA:", e)

best_model_name = min(model_results, key=lambda k: model_results[k]['mae'])
best_mae = model_results[best_model_name]['mae']
best_accuracy = model_results[best_model_name]['accuracy']

print("-" * 60)
print(f"🏆 โมเดลที่แม่นยำที่สุดคือ: {best_model_name} (Accuracy: {best_accuracy}%, MAE: ±{best_mae} µg/m³)")
print("="*60 + "\n")

# ==============================================================================
# 4. API Endpoints
# ==============================================================================
@app.get("/predict")
def predict_pm25(date: str):
    try:
        target_dt = pd.to_datetime(date)
    except Exception:
        raise HTTPException(status_code=400, detail="รูปแบบวันที่ผิดพลาด (YYYY-MM-DD)")

    if target_dt not in df_model.index and target_dt > df_daily.index.max() + pd.Timedelta(days=365):
        raise HTTPException(status_code=400, detail="วันที่เลือกอยู่นอกขอบเขตชุดข้อมูล")

    results = []

    if best_model_name == "SVR":
        df_calc = df_model.copy()
        if target_dt > df_daily.index.max():
            last_known_dt = df_daily.index.max()
            curr_dt = last_known_dt + pd.Timedelta(days=1)
            while curr_dt <= target_dt + pd.Timedelta(days=7):
                df_calc.loc[curr_dt, 'Lag1'] = df_calc.loc[curr_dt - pd.Timedelta(days=1), 'PM25']
                df_calc.loc[curr_dt, 'Lag2'] = df_calc.loc[curr_dt - pd.Timedelta(days=2), 'PM25']
                df_calc.loc[curr_dt, 'Lag3'] = df_calc.loc[curr_dt - pd.Timedelta(days=3), 'PM25']
                df_calc.loc[curr_dt, 'Lag7'] = df_calc.loc[curr_dt - pd.Timedelta(days=7), 'PM25']
                df_calc.loc[curr_dt, 'PM25_Diff1'] = df_calc.loc[curr_dt, 'Lag1'] - df_calc.loc[curr_dt, 'Lag2']
                recent_pms = [df_calc.loc[curr_dt - pd.Timedelta(days=j), 'PM25'] for j in range(1, 8)]
                df_calc.loc[curr_dt, 'Rolling7_Mean'] = np.nanmean(recent_pms)
                df_calc.loc[curr_dt, 'Rolling14_Mean'] = np.nanmean(recent_pms)
                df_calc.loc[curr_dt, 'EWMA7'] = np.nanmean(recent_pms)
                row_sim = df_calc.loc[[curr_dt], features]
                df_calc.loc[curr_dt, 'PM25'] = trained_models["SVR"]['t1'].predict(row_sim)[0]
                curr_dt += pd.Timedelta(days=1)

        row = df_calc.loc[[target_dt], features]
        actual_day0 = float(df_daily.loc[target_dt, 'PM25']) if target_dt in df_daily.index else None
        pred_day0 = actual_day0 if actual_day0 is not None else (float(df_calc.loc[target_dt, 'PM25']) if target_dt in df_calc.index and pd.notna(df_calc.loc[target_dt, 'PM25']) else None)

        results.append({
            "day_name": "วันนี้ (Day 0)",
            "date": target_dt.strftime('%d/%m/%Y'),
            "predicted": round(pred_day0, 1) if pred_day0 is not None else None,
            "actual": round(actual_day0, 1) if actual_day0 is not None else None
        })

        for i in range(1, 8):
            future_date = target_dt + pd.Timedelta(days=i)
            pred_val = float(trained_models["SVR"][f't{i}'].predict(row)[0])
            actual_val = float(df_daily.loc[future_date, 'PM25']) if future_date in df_daily.index else None
            
            results.append({
                "day_name": f"Day {i}",
                "date": future_date.strftime('%d/%m/%Y'),
                "predicted": round(pred_val, 1),
                "actual": round(actual_val, 1) if actual_val is not None else None
            })

    else:  # ARIMA
        arima_fit = trained_models["ARIMA"]
        forecast_res = arima_fit.forecast(steps=8)
        
        for i in range(8):
            future_date = target_dt + pd.Timedelta(days=i)
            actual_val = float(df_daily.loc[future_date, 'PM25']) if future_date in df_daily.index else None
            pred_val = float(forecast_res.iloc[i]) if i < len(forecast_res) else float(forecast_res.iloc[-1])

            results.append({
                "day_name": "วันนี้ (Day 0)" if i == 0 else f"Day {i}",
                "date": future_date.strftime('%d/%m/%Y'),
                "predicted": round(actual_val if (i == 0 and actual_val is not None) else pred_val, 1),
                "actual": round(actual_val, 1) if actual_val is not None else None
            })

    return {
        "base_date": target_dt.strftime('%d/%m/%Y'),
        "selected_model": best_model_name,
        "forecast": results,
        "metrics": {
            "mae": best_mae,
            "accuracy": best_accuracy,
            "comparison": model_results
        }
    }

@app.get("/api/model-evaluation")
def get_model_evaluation():
    return {
        "best_model": best_model_name,
        "comparison": model_results
    }

@app.get("/api/pm25-map-tile")
def get_pm25_map_tile(date: str):
    try:
        start_dt = pd.to_datetime(date)
        end_dt = start_dt + pd.Timedelta(days=1)
        
        cams = ee.ImageCollection('ECMWF/CAMS/NRT') \
            .filterDate(start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')) \
            .select('particulate_matter_d_less_than_25_um_surface')

        pm25_img = cams.mean().multiply(1e9)
        vis_params = {
            'min': 5.0,
            'max': 60.0,
            'palette': ['00e400', 'ffff00', 'ff7e00', 'ff0000', '99004c', '7e0023']
        }
        map_id_dict = pm25_img.getMapId(vis_params)
        return {"tile_url": map_id_dict['tile_fetcher'].url_format}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))