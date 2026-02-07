import cv2
from ultralytics import YOLO

# 1. Load Model
model_path = r"D:/Unarrage/4Feb.pt"
model = YOLO(model_path)

# 2. Cấu hình Camera IP
# Thử thêm cv2.CAP_FFMPEG để OpenCV xử lý luồng video tốt hơn
ip_url = "http://192.168.1.129:8080/video"
cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)

# KIỂM TRA KẾT NỐI NGAY LẬP TỨC
if not cap.isOpened():
    print("--- KHÔNG THỂ KẾT NỐI CAMERA ---")
    print("Vui lòng kiểm tra:")
    print("1. Điện thoại và máy tính chung Wi-Fi chưa?")
    print("2. URL đã đúng chưa (thử dán vào trình duyệt)?")
    exit()

print("Kết nối thành công! Đang mở cửa sổ hiển thị...")

# Tạo cửa sổ trước để tránh bị lỗi "not responding"
cv2.namedWindow("YOLO_Detect", cv2.WINDOW_NORMAL) 

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Mất kết nối với camera...")
        break

    # 3. Inference
    # Lưu ý: conf=0.8 của bạn khá cao, mình hạ xuống 0.5 để dễ thấy kết quả
    results = model(frame, imgsz=640, conf=0.5, verbose=False)

    # 4. Vẽ kết quả
    annotated_frame = results[0].plot() 

    # 5. Hiển thị
    cv2.imshow("YOLO_Detect", annotated_frame)

    # BẮT BUỘC phải có waitKey để cửa sổ window cập nhật hình ảnh
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()