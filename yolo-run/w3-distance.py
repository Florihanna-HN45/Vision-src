import cv2
from ultralytics import YOLO

model_path = r"D:\Unarrage\kfs1.pt"
try:
    model = YOLO(model_path, task='detect')
    print(f"Đã load model thành công: {model_path}")
except Exception as e:
    print(f"Lỗi load model: {e}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera đang chạy... Nhấn 'q' để thoát.")

# 1. THIẾT LẬP THAM SỐ ƯỚC LƯỢNG KHOẢNG CÁCH (tạm tính)
# Ví dụ: biết vật thể "hand" có chiều cao thật khoảng 0.15 m
KNOWN_HEIGHT_M = 0.35          # thực tế (thay tên lớp nếu cần)
FOCAL_LENGTH_PX = 600          # ước tính tiêu cự camera (px), có thể hiệu chỉnh sau
# Nếu bạn có 1 vật tham chiếu (biết kích thước thật), dùng công thức:
#   FOCAL = (width_in_pixel * distance_real) / real_width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2
    annotated_frame = frame.copy()

    results = model(frame, imgsz=640, conf=0.8, iou=0.45, verbose=False)
    num_boxes = len(results[0].boxes)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]}: {conf:.2f}"

        # 2. Tính trọng tâm bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 3. Vẽ bbox + tâm bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

        # 4. Kẻ đường từ tâm ảnh đến tâm bbox
        cv2.line(annotated_frame, (img_cx, img_cy), (cx, cy), (255, 200, 0), 2)

        # 5. Khoảng cách tương đối trên màn (pixel)
        dist_px = int(((cx - img_cx)**2 + (cy - img_cy)**2)**0.5)
        cv2.putText(annotated_frame, f"Dist: {dist_px}px", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 6. Khoảng cách thực tế tạm tính (tạm theo chiều cao bbox)
        #   distance = (KNOWN_HEIGHT_M * FOCAL_LENGTH_PX) / bbox_height_px
        bbox_h_px = y2 - y1
        if bbox_h_px > 0:
            distance_real = (KNOWN_HEIGHT_M * FOCAL_LENGTH_PX) / bbox_h_px
            cv2.putText(annotated_frame, f"Real: {distance_real:.2f}m", (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7. Thông tin tổng số vật thể
    info_text = f"Objects: {num_boxes} | Conf: 0.8"
    cv2.putText(annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO + Center Lines + Distance", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()