import cv2
from ultralytics import YOLO

# 1. Load model ONNX
model_path = r"D:/Unarrage/4Feb-box.pt"

try:
    model = YOLO(model_path, task='detect')
    print(f"Đã load model thành công: {model_path}")

    # Gán tên class (Quan trọng để không bị lỗi hiển thị)
    # MY_CLASS_NAMES = {0: 'fist', 1: 'giao', 2: 'hand'}

except Exception as e:
    print(f"Lỗi load model: {e}")
    exit()

# 2. Cấu hình Camera
# ip_url = "http://192.168.1.129:8080/video/video" 

# cap = cv2.VideoCapture(ip_url)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera đang chạy... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Suy luận (Inference)
    # --- THAY ĐỔI QUAN TRỌNG: conf=0.25 (hạ thấp xuống để test) ---
    results = model(frame, imgsz=640, conf=0.6, iou=0.45, verbose=False)

    # Gán tên class vào kết quả
    # results[0].names = MY_CLASS_NAMES

    # --- ĐOẠN DEBUG: In ra số lượng vật thể tìm thấy ---
    num_boxes = len(results[0].boxes)
    if num_boxes > 0:
        print(f"-> Tìm thấy {num_boxes} vật thể!") # Nếu dòng này hiện, chắc chắn sẽ có hình
    # --------------------------------------------------

    # 4. Vẽ và hiển thị
    try:
        annotated_frame = results[0].plot()
        
        # Hiển thị thông số lên màn hình
        info_text = f"Objects: {num_boxes} | Conf: 0.25"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO ONNX Inference", annotated_frame)
    except Exception as e:
        print(f"Lỗi khi vẽ: {e}")
        cv2.imshow("YOLO ONNX Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()