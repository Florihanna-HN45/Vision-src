import cv2
import numpy as np
from ultralytics import YOLO
import time

# 1. Cấu hình
model = YOLO(r"D:\1. STUDY\ROBOCON 2026\Weapons1\src\model\7Feb.pt")
ip_url = "http://192.168.1.114:8080/video"
cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)

# Cấu hình tối ưu tốc độ
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
frame_skip = 2  
frame_count = 0

# --- KHỞI TẠO BIẾN TRƯỚC VÒNG LẶP (QUAN TRỌNG) ---
targets = [] 
prev_time = 0
fps = 0

cv2.namedWindow("YOLO_Fast_Detect", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    frame_count += 1
    h_frame, w_frame, _ = frame.shape
    # Điểm gốc: Giữa cạnh dưới màn hình
    base_point = (int(w_frame / 2), h_frame)

    # 2. Xử lý logic AI (Chỉ chạy khi đến lượt skip)
    if frame_count % frame_skip == 0:
        # Giảm imgsz xuống 320 để tăng FPS đáng kể
        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        
        current_detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                current_detections.append({
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)), 
                    'class': cls, 
                    'x_min': x1
                })

        # Sắp xếp từ trái qua phải
        current_detections.sort(key=lambda x: x['x_min'])
        
        # Lọc lấy 3 loại vũ khí khác nhau đầu tiên từ trái sang
        temp_targets = []
        picked_classes = []
        for det in current_detections:
            if det['class'] not in picked_classes and len(picked_classes) < 3:
                picked_classes.append(det['class'])
                temp_targets.append(det)
        
        # Cập nhật danh sách targets để bước vẽ sử dụng
        targets = temp_targets

    # 3. Tính toán FPS
    curr_time = time.time()
    if (curr_time - prev_time) > 0:
        fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 4. Vẽ kết quả (Luôn chạy cho mọi frame để không bị nhấp nháy)
    for target in targets:
        tx, ty = target['center']
        # Vẽ đường kẻ từ base tới vật thể
        cv2.line(frame, base_point, (tx, ty), (34, 122, 34), 2)
        
        # Tính khoảng cách Euclidean
        dist = np.sqrt((tx - base_point[0])**2 + (ty - base_point[1])**2)
        
        # Hiển thị ID và Khoảng cách
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{target['class']} Dist:{int(dist)}px", 
                    (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị FPS lên màn hình
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("YOLO_Fast_Detect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()