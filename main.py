from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import ee
import pandas as pd
from datetime import datetime, timedelta
import os

app = FastAPI(title="PM2.5 & GEE Layer API")

# ตั้งค่า CORS เพื่ออนุญาตให้ Frontend (index.html) เรียกใช้ API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 🛰️ ส่วนที่ 1: Google Earth Engine (GEE) API
# ==============================================================================
SERVICE_ACCOUNT_EMAIL = 'gee-pm25-service@neat-episode-505714-g6.iam.gserviceaccount.com'
KEY_FILE_PATH = 'gee-key.json'
PROJECT_ID = 'neat-episode-505714-g6'

try:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, KEY_FILE_PATH)
    ee.Initialize(credentials, project=PROJECT_ID)
    print("✅ [GEE] ยืนยันตัวตนและเชื่อมต่อสำเร็จ!")
except Exception as e:
    print("❌ [GEE] ไม่สามารถเชื่อมต่อได้:", e)


@app.get("/api/gee-tile")
def get_gee_tile():
    try:
        # ขอบเขตพื้นที่ภาคเหนือของประเทศไทย (เชียงราย พะเยา น่าน เชียงใหม่ ฯลฯ)
        roi = ee.Geometry.BBox(97.3, 15.0, 101.5, 20.5)

        # ดึงข้อมูลความหนาแน่น PM2.5 จาก CAMS NRT แบบกระจายตัวทั่วพื้นที่
        dataset = (ee.ImageCollection('ECMWF/CAMS/NRT')
                   .filterBounds(roi)
                   .filterDate('2024-01-01', '2024-12-31')
                   .select('particulate_matter_d_less_than_2_5_um_surface')
                   .mean()
                   .multiply(1e9)  # แปลงหน่วยเป็น µg/m³
                   .clip(roi))

        # การกำหนดช่วงสี (เขียว -> เหลือง -> ส้ม -> แดง)
        vis_params = {
            'min': 0,
            'max': 75,
            'palette': ['00e400', 'ffff00', 'ff7e00', 'ff0000']
        }

        map_id = ee.Image(dataset).getMapId(vis_params)
        return {"status": "success", "tile_url": map_id['tile_fetcher'].url_format}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==============================================================================
# 📊 ส่วนที่ 2: ประมวลผลและพยากรณ์ฝุ่น PM2.5 จาก Excel (2022 - 2026)
# ==============================================================================
EXCEL_FILES = [
    'สนามกีฬาจังหวัดพะเยา ปี 2022.xlsx',
    'สนามกีฬาจังหวัดพะเยา ปี 2023.xlsx',
    'สนามกีฬาจังหวัดพะเยา ปี 2024.xlsx',
    'สนามกีฬาจังหวัดพะเยา ปี 2025.xlsx',
    'สนามกีฬาจังหวัดพะเยา ปี 2026.xlsx'
]

def load_all_excel_data():
    all_dfs = []
    for filepath in EXCEL_FILES:
        if os.path.exists(filepath):
            df = pd.read_excel(filepath, skiprows=3, names=['date_str', 'pm25'])
            df['date'] = pd.to_datetime(df['date_str'], format='%d/%m/%Y', errors='coerce')
            all_dfs.append(df)
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.dropna(subset=['date']).sort_values('date')
    return pd.DataFrame()

def get_risk_level(val: float) -> str:
    if val <= 15.0: return "ดีมาก"
    elif val <= 37.0: return "ปานกลาง"
    elif val <= 75.0: return "เริ่มมีผลกระทบ"
    else: return "มีผลกระทบ"


@app.get("/predict")
def predict_pm25(date: str = Query(...)):
    try:
        df = load_all_excel_data()
        if df.empty:
            return {"status": "error", "message": "ไม่พบไฟล์ข้อมูล Excel ในระบบ"}

        selected_dt = pd.to_datetime(date)
        pm_dict = df.set_index('date')['pm25'].to_dict()

        forecast_list = []
        labels = ["วันนี้", "พรุ่งนี้", "ล่วงหน้า 2 วัน", "ล่วงหน้า 3 วัน", "ล่วงหน้า 4 วัน", "ล่วงหน้า 5 วัน", "ล่วงหน้า 6 วัน", "ล่วงหน้า 7 วัน"]
        predicted_values = []

        for i in range(8):
            current_dt = selected_dt + timedelta(days=i)
            actual_val = pm_dict.get(current_dt, None)

            if actual_val is not None and pd.notnull(actual_val):
                actual_res = round(float(actual_val), 1)
                pred_res = round(float(actual_val), 1)
            else:
                actual_res = "N/A"
                past_vals = df[(df['date'].dt.month == current_dt.month) & (df['date'].dt.day == current_dt.day)]['pm25'].dropna()
                pred_res = round(float(past_vals.mean()), 1) if not past_vals.empty else 12.0

            predicted_values.append(pred_res)
            forecast_list.append({
                "period": labels[i],
                "date": current_dt.strftime('%Y-%m-%d'),
                "actual": actual_res,
                "predicted": pred_res,
                "full_label": f"{labels[i]} ({current_dt.strftime('%d/%m')})"
            })

        avg_val = round(sum(predicted_values) / len(predicted_values), 1)
        max_val = round(max(predicted_values), 1)

        return {
            "status": "success",
            "summary": {
                "average": avg_val,
                "max_val": max_val,
                "risk_level": get_risk_level(max_val)
            },
            "forecast": forecast_list
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}