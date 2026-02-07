import cv2
import torch
import time
import pathlib

# --- ĐOẠN FIX LỖI POSIXPATH ---
# Mẹo: Ép hệ thống Windows hiểu PosixPath (đường dẫn Linux)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# ------------------------------

# 1. Load model YOLOv5
model_path = r"D:/Unarrage/4Feb.pt"
# model_path = r"D:/Unarrage/hoa.pt"
try:
    # Thêm force_reload=True nếu bạn nghi ngờ cache bị lỗi
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    print("Model loaded successfully on Windows!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Phục hồi lại pathlib sau khi load xong (để tránh lỗi các hàm khác)
pathlib.PosixPath = temp

# 2. Cấu hình
model.conf = 0.5
cap = cv2.VideoCapture(0)

print("Camera is running...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Suy luận
    results = model(frame)
    
    # Vẽ và hiển thị
    annotated_frame = results.render()[0]
    cv2.imshow("YOLOv5 Windows Fix", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()