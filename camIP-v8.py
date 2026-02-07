import cv2
from ultralytics import YOLO  # Thư viện chuẩn của YOLOv8

# 1. Load model YOLOv8
# Lưu ý: File .pt này tốt nhất nên được train bằng YOLOv8. 
# Nếu là file của YOLOv5 cũ, thư viện vẫn có thể đọc được nhưng đôi khi cần convert.
model_path = r"D:/Unarrage/4Feb.pt"  # Thay đường dẫn tới model của bạn

try:
    # Load model trực tiếp, không cần torch.hub
    model = YOLO(model_path)
    print("✅ Đã load Model YOLOv8 thành công!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    print("👉 Gợi ý: Hãy đảm bảo bạn đã cài thư viện: pip install ultralytics")
    exit()

# 2. Cấu hình Camera IP
# Thay đúng địa chỉ IP Webcam của bạn vào đây
ip_url = "http://10.56.48.240:8080/video" 

cap = cv2.VideoCapture(ip_url)
# print(f"📡 Đang kết nối tới Camera tại: {ip_url}...")
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể kết nối tới Camera IP. Hãy kiểm tra lại Wifi/4G.")
    exit()

# 3. Vòng lặp xử lý
while True:
    ret, frame = cap.read()
    if not ret: 
        print("⚠️ Mất kết nối stream.")
        break

    # --- PHẦN KHÁC BIỆT NHẤT SO VỚI YOLOV5 ---
    # Thay vì model.conf = 0.8, ta truyền trực tiếp vào hàm predict
    # conf=0.6: Chỉ hiện vật thể có độ tin cậy > 60%
    # verbose=False: Để đỡ bị spam log đầy màn hình console
    results = model(frame, conf=0.6, verbose=False)
    
    # Lấy kết quả vẽ (Plot) từ YOLOv8
    # results[0] là kết quả của frame đầu tiên (vì ta chỉ đưa vào 1 ảnh)
    annotated_frame = results[0].plot() 

    # Hiển thị
    cv2.imshow("YOLOv8 - Robocon 2026", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()