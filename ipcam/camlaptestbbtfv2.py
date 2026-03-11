import cv2
import numpy as np
from ultralytics import YOLO
import time
from threading import Thread

class WebCamStream:
    def __init__(self, src=0): # Mặc định là 0 cho Webcam laptop
        # Sử dụng cv2.CAP_DSHOW trên Windows để khởi động Camera nhanh hơn
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW) 
        if not self.stream.isOpened():
            print("LỖI: Không thể mở Webcam laptop.")
            self.stopped = True
            return
            
        # Tối ưu cho Camera Laptop
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
            else:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- 1. KHỞI TẠO ---
model_path = r"D:\1. STUDY\ROBOCON 2026\Weapons1\src\model\7Feb.tflite"
model = YOLO(model_path, task='detect')
MODEL_LABELS = model.names 

# THAY ĐỔI TẠI ĐÂY: src=0 là camera mặc định của laptop
vs = WebCamStream(src=0).start()
time.sleep(1.0) # Webcam laptop khởi động nhanh hơn IP Cam

completed_classes = set()
picked_count = 0
EXIT_THRESHOLD = 50
prev_time = time.time()
FONT = cv2.FONT_HERSHEY_PLAIN

print("Đang đọc dữ liệu từ Webcam Laptop...")

while True:
    frame = vs.read()
    if frame is None: break

    # Để đảm bảo không bị ngược hình (nếu muốn soi gương), có thể dùng:
    # frame = cv2.flip(frame, 1) 

    h_f, w_f = frame.shape[:2]
    base_x, base_y = w_f // 2, h_f
    
    # Inference với kích thước cố định của model
    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    
    current_dets = []
    
    if results and results[0].boxes:
        boxes = results[0].boxes
        xyxy_all = boxes.xyxy.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(xyxy_all)):
            c_id = cls_all[i]
            if c_id in completed_classes: continue
                
            x1, y1, x2, y2 = xyxy_all[i].astype(int)
            det = {
                'box': (x1, y1, x2, y2),
                'class': c_id,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'y_max': y2
            }
            current_dets.append(det)

            # Vẽ box và nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame, MODEL_LABELS[c_id], (x1, y1 - 5), FONT, 1, (255, 0, 0), 1)

        if current_dets:
            # Sắp xếp từ trái qua phải
            current_dets.sort(key=lambda d: d['box'][0])
            targets = []
            seen = set()
            for d in current_dets:
                if d['class'] not in seen:
                    seen.add(d['class'])
                    targets.append(d)
                    if len(targets) == 3: break
            
            if targets:
                t0 = targets[0]
                tx, ty = t0['center']
                cv2.line(frame, (base_x, base_y), (tx, ty), (0, 255, 0), 2)
                
                # Logic giả lập gắp vật phẩm khi nó sát mép dưới camera
                if t0['y_max'] > h_f - EXIT_THRESHOLD:
                    completed_classes.add(t0['class'])
                    picked_count += 1
                    print(f"Đã gắp: {MODEL_LABELS[t0['class']]}")

    # Tính toán FPS
    c_time = time.time()
    dt = c_time - prev_time
    fps = 1 / dt if dt > 0 else 0
    prev_time = c_time

    # Dashboard hiển thị
    cv2.putText(frame, f"FPS:{int(fps)} | PICKED:{picked_count}/3", (10, 25), FONT, 1.2, (0, 255, 255), 1)
    
    cv2.imshow("Laptop_Camera_Robocon", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.stop()
cv2.destroyAllWindows()