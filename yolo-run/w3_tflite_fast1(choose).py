"""
weapon3-tflite OPTIMIZED — Tối ưu FPS cho phần cứng yếu
=========================================================
Fix: imgsz=640 cố định (TFLite int8 shape bị đóng băng lúc export)
Tăng FPS bằng: threading | skip frame | adaptive skip | max_det | conf cao hơn
"""

import cv2
import threading
import queue
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CẤU HÌNH — Chỉnh tại đây
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\This pc\Downloads\pdisease1_int8 (2).tflite"
CAMERA_INDEX = 1                # 0 = webcam mặc định, 1 = camera phụ

# Camera input resolution (nhỏ hơn → đọc frame nhanh hơn)
CAP_WIDTH    = 480
CAP_HEIGHT   = 360

# ⚠️ PHẢI = 640 với TFLite int8 — không được đổi
IMGSZ        = 640

# Inference settings
CONF_THRESH  = 0.5
IOU_THRESH   = 0.45
MAX_DET      = 10               # giới hạn số box → NMS nhanh hơn

# Adaptive skip frame
INITIAL_SKIP = 3                # inference mỗi (SKIP+1) frame, tăng nếu lag
TARGET_FPS   = 20               # FPS mục tiêu để adaptive tự điều chỉnh
# ─────────────────────────────────────────────


class CameraReader(threading.Thread):
    """Thread đọc camera liên tục, chỉ giữ frame MỚI NHẤT."""

    def __init__(self, source, width, height):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class InferenceWorker(threading.Thread):
    """Thread inference riêng: nhận frame, trả annotated_frame."""

    def __init__(self, model, imgsz, conf, iou, max_det):
        super().__init__(daemon=True)
        self.model   = model
        self.imgsz   = imgsz
        self.conf    = conf
        self.iou     = iou
        self.max_det = max_det
        self.in_q    = queue.Queue(maxsize=1)
        self.out_q   = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            results = self.model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
            )
            result    = results[0]
            annotated = result.plot()
            num_boxes = len(result.boxes)

            if num_boxes > 0:
                print(f"  → Phát hiện {num_boxes} vật thể!")

            # Luôn giữ kết quả MỚI NHẤT
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
            self.out_q.put((annotated, num_boxes))

    def submit(self, frame):
        if self.in_q.full():
            try:
                self.in_q.get_nowait()
            except queue.Empty:
                pass
        self.in_q.put_nowait(frame)

    def get_result(self):
        try:
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False


def draw_overlay(frame, fps, num_boxes, skip):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    fps_color = (0, 255, 0) if fps >= TARGET_FPS else \
                (0, 165, 255) if fps >= 10 else (0, 0, 255)

    cv2.putText(frame, f"FPS  : {fps:5.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)
    cv2.putText(frame, f"Skip : every {skip + 1} frames",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Objs : {num_boxes}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return frame


def main():
    # ── Load model ───────────────────────────────────────────────────────────
    print(f"⏳ Đang load model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH, task="detect")
        print("✅ Load model thành công")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # ── Warmup ───────────────────────────────────────────────────────────────
    print("🔥 Warmup model (imgsz=640)...")
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype="uint8")
    model(dummy, imgsz=IMGSZ, conf=CONF_THRESH, iou=IOU_THRESH,
          max_det=MAX_DET, verbose=False)
    print("✅ Warmup xong\n")

    # ── Khởi động threads ────────────────────────────────────────────────────
    cam    = CameraReader(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
    worker = InferenceWorker(model, IMGSZ, CONF_THRESH, IOU_THRESH, MAX_DET)
    cam.start()
    worker.start()
    print("🚀 Đang chạy... Nhấn 'q' để thoát.")

    frame_count    = 0
    skip_frames    = INITIAL_SKIP
    fps_deque      = deque(maxlen=30)
    last_time      = time.perf_counter()
    last_annotated = None
    last_num_boxes = 0
    fps_display    = 0.0
    adjust_counter = 0

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        # Gửi frame cho inference theo lịch skip
        if frame_count % (skip_frames + 1) == 0:
            worker.submit(frame)

        # Lấy kết quả mới nhất nếu có
        result = worker.get_result()
        if result is not None:
            last_annotated, last_num_boxes = result

        # Tính FPS
        now = time.perf_counter()
        fps_deque.append(1.0 / max(now - last_time, 1e-6))
        last_time   = now
        fps_display = sum(fps_deque) / len(fps_deque)

        # Adaptive skip mỗi 60 frame display
        adjust_counter += 1
        if adjust_counter >= 60:
            adjust_counter = 0
            if fps_display < TARGET_FPS * 0.7 and skip_frames < 6:
                skip_frames += 1
            elif fps_display > TARGET_FPS * 1.2 and skip_frames > 0:
                skip_frames -= 1

        # Hiển thị
        display_frame = last_annotated if last_annotated is not None else frame
        display_frame = draw_overlay(display_frame, fps_display,
                                     last_num_boxes, skip_frames)
        cv2.imshow("YOLO TFLite | Optimized", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Đang dừng...")
            break

    cam.stop()
    worker.stop()
    cv2.destroyAllWindows()
    print("✅ Đã dừng hoàn toàn.")


if __name__ == "__main__":
    main()