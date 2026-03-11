import cv2
import numpy as np
from ultralytics import YOLO
import time
from threading import Thread

# --- LỚP ĐỌC CAMERA ĐA LUỒNG TỐI ƯU ---
class WebCamStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        # Kiểm tra kết nối ngay lập tức
        if not self.stream.isOpened():
            print(f"LỖI: Không thể kết nối tới {src}. Kiểm tra IP/Wifi!")
            self.stopped = True
            return
            
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
            else:
                self.frame = frame

    def read(self):
        return self.frame if not self.stopped else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- 1. CẤU HÌNH ---
model_path = r"D:\1. STUDY\ROBOCON 2026\Weapons1\src\model\7Feb.tflite"
# Tối ưu: Dùng trực tiếp task 'detect' để YOLO bỏ qua các bước tự đoán task
model = YOLO(model_path, task='detect')

ip_url = "http://192.168.1.113:8080/video"
vs = WebCamStream(ip_url).start()

# Chờ 2 giây hoặc thoát nếu lỗi -138
time.sleep(2.0)
if vs.read() is None:
    print("Dừng chương trình do lỗi kết nối Camera.")
    exit()

# Biến logic
picked_count = 0
completed_classes = set()
prev_time = time.time()
EXIT_THRESHOLD = 50
all_detections = []
targets = []

print("Bắt đầu xử lý với TFLite...")

while True:
    frame = vs.read()
    if frame is None: break

    h_frame, w_frame = frame.shape[:2]
    base_point = (w_frame // 2, h_frame)

    # --- 2. XỬ LÝ AI TỐI ƯU ---
    # verbose=False: Tắt in log terminal (tiết kiệm CPU)
    # imgsz=320: Phải khớp với lúc export để đạt tốc độ cao nhất
    results = model(frame, imgsz=320, conf=0.45, verbose=False)
    
    current_frame_detections = []
    
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        # Chuyển dữ liệu sang CPU/Numpy một lần duy nhất
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        for i in range(len(xyxy)):
            cls = int(classes[i])
            # Nếu class đã gắp rồi thì bỏ qua bước tính toán logic, chỉ để vẽ
            det = {
                'box': xyxy[i].astype(int),
                'center': (int((xyxy[i][0] + xyxy[i][2]) / 2), int((xyxy[i][1] + xyxy[i][3]) / 2)),
                'class': cls,
                'y_max': int(xyxy[i][3])
            }
            current_frame_detections.append(det)

        all_detections = current_frame_detections.copy()
        
        # Sắp xếp theo x_min để gắp từ trái sang phải
        current_frame_detections.sort(key=lambda x: x['box'][0])
        
        # Tìm 3 loại mục tiêu ưu tiên
        temp_targets = []
        seen_in_frame = set()
        for det in current_frame_detections:
            c = det['class']
            if c not in completed_classes and c not in seen_in_frame:
                seen_in_frame.add(c)
                temp_targets.append(det)
                if len(temp_targets) == 3: break
        targets = temp_targets

    # --- 3. LOGIC GẮP ---
    if targets:
        p_target = targets[0]
        if p_target['y_max'] > h_frame - EXIT_THRESHOLD:
            completed_classes.add(p_target['class'])
            picked_count += 1
            print(f"GẮP THÀNH CÔNG: Class {p_target['class']} | Tổng: {picked_count}/3")

    # --- 4. VẼ HIỂN THỊ (Tối giản để giữ FPS) ---
    for det in all_detections:
        x1, y1, x2, y2 = det['box']
        clr = (80, 80, 80) if det['class'] in completed_classes else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 1)

    if targets:
        tx, ty = targets[0]['center']
        cv2.line(frame, base_point, (tx, ty), (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)

    # Tính FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 1, 1, (0, 255, 255), 1)
    cv2.putText(frame, f"PICKED: {picked_count}/3", (10, 60), 1, 1, (0, 0, 255), 1)
    
    cv2.imshow("Robocon_TFLite_Optimized", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.stop()
cv2.destroyAllWindows()