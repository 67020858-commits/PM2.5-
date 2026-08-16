ขั้นตอนที่คนเปิดต้องทำ (พิมพ์ใน Terminal ของ VS Code):
ขั้นที่ 1: สร้างและเปิดใช้งาน Virtual Environment ใหม่

Bash
# สร้าง .venv ในเครื่องผู้รับ
python -m venv .venv

# เปิดใช้งาน (Windows)
.venv\Scripts\activate

ขั้นที่ 2: ติดตั้ง Library ทั้งหมดจากไฟล์ requirements.txt

Bash
pip install -r requirements.txt

ขั้นที่ 3: สั่งรันโปรเจกต์

Bash
uvicorn main:app --reload