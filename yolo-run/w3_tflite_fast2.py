# -*- coding: utf-8 -*-
"""
weapon3-tflite ULTRA -- Tối ưu cực mạnh cho CPU không GPU
=========================================================
Mục tiêu: gắn robot, chạy liên tục, giảm lag + giật tối đa
---------------------------------------------------------
Kỹ thuật:
  1. Camera thread riêng + MJPG codec -> đọc frame cực nhanh
  2. KHÔNG dùng result.plot() -> tự vẽ box thủ công (3-5× nhanh hơn)
  3. Pre-resize frame bằng cv2 trước khi feed model (không để YOLO resize)
  4. Skip frame cứng + adaptive -- không bao giờ queue tích tụ
  5. TFLite interpreter gọi trực tiếp (bypass ultralytics overhead)
  6. Warmup đủ 3 lần để JIT ổn định
  7. FPS display tách khỏi inference timing
"""

import cv2
import threading
import time
import numpy as np
from collections import deque

# -----------------------------------------------------------------
# 
# -----------------------------------------------------------------
MODEL_PATH    = r"C:\Users\This pc\Downloads\pdisease1_int8 (2).tflite"
CAMERA_INDEX  = 1

CAP_WIDTH     = 640
CAP_HEIGHT    = 480
IMGSZ         = 640          

CONF_THRESH   = 0.5
IOU_THRESH    = 0.45
MAX_DET       = 10

SKIP_FRAME    = 2            
TARGET_FPS    = 25
# -----------------------------------------------------------------

BOX_COLOR   = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)
LABEL_BG    = (0, 255, 0)


# -- ----------
def load_tflite(model_path):
    try:
        # 
        from tflite_runtime.interpreter import Interpreter
        print("[OK] Dung tflite_runtime (nhe hon)")
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
        print("[OK] Dung tensorflow.lite")

    interp = Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    out  = interp.get_output_details()
    return interp, inp, out

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        iou = (w * h) / np.maximum(areas[order[1:]] + areas[i] - w * h, 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def postprocess(outputs, orig_h, orig_w, conf_thresh, iou_thresh, max_det):
    
    pred = outputs[0].squeeze()          # 

    # 
    if pred.ndim == 1:
        return []
    if pred.shape[0] < pred.shape[1]:    # shape (4+nc, anchors) -> transpose
        pred = pred.T

    # pred: [anchors, 4 + num_classes]
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]

    class_ids  = np.argmax(scores_all, axis=1)
    confs      = scores_all[np.arange(len(class_ids)), class_ids]

    mask  = confs >= conf_thresh
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    class_ids  = class_ids[mask]

    # cx,cy,w,h -> x1,y1,x2,y2 (normalized 0-1 -> pixel)
    cx = boxes_xywh[:, 0] / IMGSZ * orig_w
    cy = boxes_xywh[:, 1] / IMGSZ * orig_h
    bw = boxes_xywh[:, 2] / IMGSZ * orig_w
    bh = boxes_xywh[:, 3] / IMGSZ * orig_h
    x1 = cx - bw / 2;  y1 = cy - bh / 2
    x2 = cx + bw / 2;  y2 = cy + bh / 2
    boxes_px = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_px, confs, iou_thresh)
    keep = keep[:max_det]

    results = []
    for i in keep:
        results.append((
            int(boxes_px[i, 0]), int(boxes_px[i, 1]),
            int(boxes_px[i, 2]), int(boxes_px[i, 3]),
            float(confs[i]), int(class_ids[i])
        ))
    return results


def draw_detections(frame, detections, class_names=None):
  
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label = f"{class_names[cls_id] if class_names else cls_id} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), LABEL_BG, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps, num_det, skip, inference_ms):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    fps_c = (0,255,0) if fps >= TARGET_FPS else (0,165,255) if fps >= 12 else (0,0,255)
    cv2.putText(frame, f"FPS      : {fps:5.1f}",     (8,24),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, fps_c,          2)
    cv2.putText(frame, f"Infer ms : {inference_ms:.0f}ms", (8,48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f"Skip     : 1/{skip+1}",     (8,72),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f"Objects  : {num_det}",      (8,96),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,255),   2)
    return frame


# -- Camera thread ------------------------------------------------------------
class CameraReader(threading.Thread):
    def __init__(self, src, w, h):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
       
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
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


# -- Main ---------------------------------------------------------------------
def main():
    print(f"[LOAD] Load TFLite interpreter: {MODEL_PATH}")
    try:
        interp, inp_detail, out_details = load_tflite(MODEL_PATH)
    except Exception as e:
        print(f"[ERR] Loi load model: {e}")
        return

    inp_dtype  = inp_detail["dtype"]
    inp_scale, inp_zero = (1.0, 0)
    if inp_detail.get("quantization_parameters"):
        qp = inp_detail["quantization_parameters"]
        if len(qp.get("scales", [])) > 0:
            inp_scale = qp["scales"][0]
            inp_zero  = qp["zero_points"][0]
    print(f"   Input dtype : {inp_dtype}, scale={inp_scale:.4f}, zero={inp_zero}")


    print("[...] Warmup (3 lan)...")
    dummy = np.zeros((1, IMGSZ, IMGSZ, 3), dtype=inp_dtype)
    for _ in range(3):
        interp.set_tensor(inp_detail["index"], dummy)
        interp.invoke()
    print("[OK] Warmup xong")

   
    cam = CameraReader(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
    cam.start()
    print("[RUN] Robot mode -- Nhan q de thoat.")

    frame_count    = 0
    skip           = SKIP_FRAME
    fps_deque      = deque(maxlen=60)
    last_t         = time.perf_counter()
    last_dets      = []
    last_frame_disp = None
    inference_ms   = 0.0
    adjust_cnt     = 0

    while True:
        raw = cam.read()
        if raw is None:
            time.sleep(0.003)
            continue

        frame_count += 1
        h0, w0 = raw.shape[:2]

        # -- -----------------------------------
        if frame_count % (skip + 1) == 0:
            # 
            resized = cv2.resize(raw, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
            inp_arr = resized[np.newaxis]          # (1, 640, 640, 3)

            # Qu
            if inp_dtype == np.int8:
                inp_arr = (inp_arr.astype(np.float32) / 255.0 / inp_scale + inp_zero
                           ).clip(-128, 127).astype(np.int8)
            elif inp_dtype == np.uint8:
                inp_arr = inp_arr.astype(np.uint8)
            else:
                inp_arr = inp_arr.astype(np.float32) / 255.0

            t0 = time.perf_counter()
            interp.set_tensor(inp_detail["index"], inp_arr)
            interp.invoke()
            outputs = [interp.get_tensor(o["index"]) for o in out_details]
            inference_ms = (time.perf_counter() - t0) * 1000

            last_dets = postprocess(outputs, h0, w0, CONF_THRESH, IOU_THRESH, MAX_DET)
            if last_dets:
                print(f"  -> {len(last_dets)} vật thể | {inference_ms:.0f}ms")

        # -- -------------------------------
        display = raw.copy()
        if last_dets:
            draw_detections(display, last_dets)

        # -- FPS --------------------------------------------------------------
        now = time.perf_counter()
        fps_deque.append(1.0 / max(now - last_t, 1e-6))
        last_t  = now
        fps_avg = sum(fps_deque) / len(fps_deque)

        # -- Adapti------------------------------------
        adjust_cnt += 1
        if adjust_cnt >= 90:
            adjust_cnt = 0
            if fps_avg < TARGET_FPS * 0.75 and skip < 7:
                skip += 1
            elif fps_avg > TARGET_FPS * 1.25 and skip > 0:
                skip -= 1

        draw_hud(display, fps_avg, len(last_dets), skip, inference_ms)
        cv2.imshow("YOLO TFLite | Robot Mode", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("[OK] Dung.")


if __name__ == "__main__":
    main()