#weapon3-tflite
import cv2
from ultralytics import YOLO

# 1. Load model TFLITE
# Đảm bảo đường dẫn file .tflite chính xác
model_path = r"C:\Users\This pc\Downloads\pdisease1_int8 (2).tflite"

try:
    # Ultralytics hỗ trợ chạy trực tiếp .tflite thông qua interpreter
    model = YOLO(model_path, task='detect')
    print(f"✅ Đã load model TFLite thành công: {model_path}")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()

# 2. Cấu hình Camera
#ip_url = "http://192.168.2.74:8080/video" 
ip_url = 0  
cap = cv2.VideoCapture(ip_url)

# Thiết lập độ phân giải để đồng bộ với imgsz của model (giúp tăng FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 Camera đang chạy... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận luồng video từ IP Camera.")
        break

    # 3. Suy luận (Inference)
    # Với TFLite int8, bạn nên giữ imgsz cố định (thường là 640)
    # conf=0.25 là mức tiêu chuẩn, bạn có thể tăng lên nếu muốn lọc nhiễu
    results = model(frame, imgsz=640, conf=0.5, iou=0.45, verbose=False)

    # Lấy kết quả từ frame đầu tiên
    result = results[0]
    num_boxes = len(result.boxes)

    # Debug: In ra nếu phát hiện vật thể
    if num_boxes > 0:
        print(f"-> Phát hiện {num_boxes} vật thể!")

    # 4. Vẽ và hiển thị
    try:
        # Sử dụng hàm plot() mặc định của YOLO để vẽ bounding box và label
        annotated_frame = result.plot()
        
        # Ghi thêm thông tin FPS/Số lượng lên màn hình
        info_text = f"TFLite int8 | Objects: {num_boxes}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO TFLite Inference", annotated_frame)
    except Exception as e:
        print(f"Lỗi hiển thị: {e}")
        cv2.imshow("YOLO TFLite Inference", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()