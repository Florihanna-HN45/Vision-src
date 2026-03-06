import cv2
import numpy as np
from ultralytics import YOLO
import time

# 1. Cấu hình
model = YOLO(r"D:\1. STUDY\ROBOCON 2026\Weapons1\src\model\7Feb.pt")
ip_url = "http://192.168.1.117:8080/video"
cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
frame_skip = 2
frame_count = 0

# --- KHỞI TẠO BIẾN LOGIC ---
all_detections = []    # Lưu tất cả để vẽ BBox
targets = []           # Lưu danh sách 3 loại ưu
picked_count = 0
completed_classes = set() 
prev_time = 0
fps = 0

EXIT_THRESHOLD = 50 

cv2.namedWindow("Robocon_Logic_Control", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret or frame is None: break
    
    frame_count += 1
    h_frame, w_frame, _ = frame.shape
    base_point = (int(w_frame / 2), h_frame)

    # 2. Xử lý AI
    if frame_count % frame_skip == 0:
        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        
        current_frame_detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                current_frame_detections.append({
                    'box': (x1, y1, x2, y2),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'class': cls,
                    'x_min': x1,
                    'y_max': y2
                })

        # Lưu lại tất cả để vẽ BBox ở Bước 4
        all_detections = current_frame_detections.copy()

        # Sắp xếp từ trái qua phải để xử lý logic gắp
        current_frame_detections.sort(key=lambda x: x['x_min'])
        
        # Lọc tìm 3 loại vũ khí khác nhau mà CHƯA PICK
        temp_targets = []
        seen_classes_in_frame = set()
        for det in current_frame_detections:
            if det['class'] not in completed_classes and det['class'] not in seen_classes_in_frame:
                if len(seen_classes_in_frame) < 3:
                    seen_classes_in_frame.add(det['class'])
                    temp_targets.append(det)
        targets = temp_targets

    # 3. Logic Đếm (Xét vật thể trái nhất trong danh sách mục tiêu)
    if targets:
        primary_target = targets[0]
        # Nếu vật thể đang được trỏ line bị kéo ra khỏi mép dưới
        if primary_target['y_max'] > h_frame - EXIT_THRESHOLD:
            completed_classes.add(primary_target['class'])
            picked_count += 1
            print(f"Đã gắp xong loại: {primary_target['class']}")

    # 4. Vẽ hiển thị
    # A. VẼ TẤT CẢ BBOX VÀ CLASS (Cho mọi vật thể nhận diện được)
    for det in all_detections:
        x1, y1, x2, y2 = det['box']
        cls_id = det['class']
        # Màu sắc: Nếu đã pick rồi thì vẽ màu xám, chưa thì màu xanh dương
        color = (100, 100, 100) if cls_id in completed_classes else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Class:{cls_id}", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # B. CHỈ KẺ LINE VÀO VẬT THỂ TRÁI NHẤT (Chưa pick)
    if targets:
        target_0 = targets[0]
        tx, ty = target_0['center']
        # Kẻ line xanh lá nổi bật
        cv2.line(frame, base_point, (tx, ty), (0, 255, 0), 3)
        cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)
        
        dist = np.sqrt((tx - base_point[0])**2 + (ty - base_point[1])**2)
        cv2.putText(frame, f"TARGETING: {int(dist)}px", (tx + 10, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # C. Hiển thị thông số tổng quát
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 1, 1.5, (0, 255, 255), 2)
    cv2.putText(frame, f"PICKED: {picked_count}/3", (20, 80), 1, 1.5, (0, 0, 255), 2)
    
    cv2.imshow("Robocon_Logic_Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()