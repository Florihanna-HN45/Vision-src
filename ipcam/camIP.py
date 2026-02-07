import cv2
import torch
import pathlib

# --- ĐOẠN FIX LỖI POSIXPATH (Giữ nguyên của bạn) ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 1. Load model YOLOv5
# Lưu ý: Đảm bảo đường dẫn này đúng trên máy của bạn
model_path = r"D:/1. STUDY/ROBOCON 2026/Weapons1/29th.pt" 

try:
    # force_reload=False để tận dụng cache, load nhanh hơn
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    print("Model loaded successfully on Windows!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

pathlib.PosixPath = temp

# 2. Cấu hình
model.conf = 0.8

# --- THAY ĐỔI Ở ĐÂY ---
# Thay dòng này bằng địa chỉ hiển thị trên app (ví dụ: IP Webcam trên Android)
# Lưu ý: Phải có đuôi "/video" nếu dùng IP Webcam để lấy luồng mượt nhất
ip_url = "http://192.168.1.129:8080/video" 

cap = cv2.VideoCapture(ip_url)
# ---------------------

print(f"Connecting to IP Camera at {ip_url}...")

if not cap.isOpened():
    print("Không thể kết nối tới Camera IP. Hãy kiểm tra lại Wifi và địa chỉ IP.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Mất kết nối stream.")
        break

    # Suy luận
    results = model(frame)
    
    # Vẽ và hiển thị
    annotated_frame = results.render()[0]
    cv2.imshow("YOLOv5 IP Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()