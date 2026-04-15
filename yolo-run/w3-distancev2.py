import cv2
from ultralytics import YOLO

# 1. LOAD MODEL YOLO
model_path = r"D:\Unarrage\kfs1.pt"  # sửa đường dẫn model của bạn
try:
    model = YOLO(model_path, task="detect")
    print(f"Đã load model: {model_path}")
except Exception as e:
    print(f"Lỗi load model: {e}")
    exit()

# 2. CẤU HÌNH CAMERA USB
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera đang chạy... Nhấn 'q' để thoát.")

# 3. THAM SỐ ƯỚC LƯỢNG KHOẢNG CÁCH
# Vật: hộp vuông 350x350 mm = 0.35 m
REAL_HEIGHT_M = 0.35  # kích thước thật của hộp (chiều cao)
REAL_WIDTH_M  = 0.35  # kích thước thật của hộp (chiều rộng)

# TIÊU CỰ (pixel) – thay bằng giá trị hiệu chuẩn của bạn
# Nếu chưa hiệu chuẩn, thử 400–800 px, điều chỉnh sau cho phù hợp
FOCAL_LENGTH_PX = 750  # focal length theo pixel (ước lượng)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2
    annotated_frame = frame.copy()

    # 4. INFER YOLO
    results = model(frame, imgsz=640, conf=0.8, iou=0.45, verbose=False)
    num_boxes = len(results[0].boxes)

    for box in results[0].boxes:
        # Lấy bbox và label
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]}: {conf:.2f}"

        # 5. TÍNH TÂM BBOX
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Vẽ bbox và tâm
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

        # 6. KẺ ĐƯỜNG TỪ TÂM ẢNH ĐẾN TÂM BBOX
        cv2.line(annotated_frame, (img_cx, img_cy), (cx, cy), (255, 200, 0), 2)

        # 7. KHOẢNG CÁCH TƯƠNG ĐỐI (pixel)
        dist_px = int(((cx - img_cx)**2 + (cy - img_cy)**2)**0.5)
        cv2.putText(annotated_frame, f"Dist (px): {dist_px}px", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 8. KHOẢNG CÁCH THỰC TẾ ƯỚC LƯỢNG (từ chiều cao bbox)
        bbox_h_px = y2 - y1
        if bbox_h_px > 10:  # tránh bbox quá nhỏ
            distance_real = (REAL_HEIGHT_M * FOCAL_LENGTH_PX) / bbox_h_px
            cv2.putText(annotated_frame, f"Est. dist: {distance_real:.2f} m", (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 9. GHI NHÃN
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 10. THÔNG TIN TỔNG
    info_text = f"Objects: {num_boxes} | Box: 35x35 cm"
    cv2.putText(annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("USB Camera + Distance Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()