# การตั้งค่าให้คนในมหาวิทยาลัยสามารถเข้าถึงได้

## ✅ การเปลี่ยนแปลง

ผมได้ปรับแก้โค้ดแล้วให้รองรับการเข้าถึงจากคนอื่นในเครือข่ายมหาวิทยาลัย:

1. **docker-compose.yml** - เปลี่ยนจาก `localhost` เป็น `0.0.0.0:8005` (รับการเชื่อมต่อจากทั้งเครือข่ายท้องถิ่น)
2. **backend/main.py** - เพิ่ม CORS Middleware (อนุญาตให้ frontend ที่อยู่ต่าง IP เข้าถึง API ได้)

## 🚀 วิธีเข้าถึงจากเครือข่ายมหาวิทยาลัย

### ขั้นตอนที่ 1: เริ่มต้นระบบบนเซิร์ฟเวอร์

```bash
docker-compose up --build
```

### ขั้นตอนที่ 2: ค้นหา IP Address ของเซิร์ฟเวอร์

**บน Windows:**
```bash
ipconfig
```
มองหา "IPv4 Address" ที่อยู่ในรูป `192.168.x.x` หรือ `10.x.x.x`

**บน Linux/Mac:**
```bash
ifconfig
# หรือ
hostname -I
```

### ขั้นตอนที่ 3: คนอื่นในมหาวิทยาลัยเข้าถึง

บนเครื่องคอมพิวเตอร์ใดๆ ที่อยู่ในเครือข่ายมหาวิทยาลัยเดียวกัน ให้เปิด Browser และไปที่:

```
http://<SERVER_IP>:8005
```

**ตัวอย่าง:**
```
http://192.168.1.100:8005
```

## 🔒 ความปลอดภัย (Security)

- CORS ได้เปิด `allow_origins=["*"]` ซึ่งหมายความว่าอนุญาตให้ผู้ใช้ที่มี access ไปที่ IP นั้น ใช้งาน API ได้
- หากต้องการจำกัดให้มีแต่ IP ที่ระบุใหม่ ให้แก้ไขไฟล์ `backend/main.py` ส่วน CORS

### ตัวอย่างการจำกัด IP ที่อนุญาต (หากต้องการ):

แก้ไขใน `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # เปลี่ยนจาก "*" เป็น list ของ IP/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📋 Troubleshooting

### Q: ไม่สามารถเข้าถึงได้จากเครื่องอื่น
**A:** ตรวจสอบว่า:
- 1. Docker container ยังทำงานอยู่ (`docker ps`)
- 2. Firewall ไม่ได้บล็อก port 8005 (ลองไปที่ Server เซ็ตอัพ Firewall)
- 3. IP Address ถูกต้อง (ping ลองดู IP นั้น)

### Q: ได้ข้อ error เกี่ยวกับ CORS
**A:** ถ้ายังเกิด error ให้ลองรีสตาร์ท Docker:
```bash
docker-compose restart backend
```

## 📝 Notes

- Frontend ใช้ `window.location.origin` ซึ่งหมายว่ามันจะเชื่อมต่อไปยังเซิร์ฟเวอร์ที่บริการ frontend นั้น โดยอัตโนมัติ
- เมื่อคนอื่นเข้าถึง `http://<SERVER_IP>:8005` frontend จะเชื่อมต่อกับ backend API ที่ URL เดียวกัน
- ไม่จำเป็นต้องแก้ไขอะไรอีกแล้ว!

---

**Happy Sharing! 🎉**
