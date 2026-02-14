import cv2
import numpy as np
import tensorflow as tf
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = r"D:\Unarrage\7Feb.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(f"Model input shape: {input_shape}")  # [1, 320, 320, 3]
input_h, input_w = input_shape[1], input_shape[2]  # 320, 320
print(f"Target: {input_h}x{input_w}")
class_names = ["class0", "class1", "class2"]
cap = cv2.VideoCapture(0)
conf_thres = 0.3

def letterbox(img, size=(320, 320), stride=32, color=(114, 114, 114)):
    """Fix letterbox cho square input 320x320 [web:18]."""
    h0, w0 = img.shape[:2]
    r = min(size[0] / h0, size[1] / w0)  # scale giữ aspect
    new_h, new_w = int(h0 * r), int(w0 * r)
    
    # Align stride
    new_h = round(new_h / stride) * stride
    new_w = round(new_w / stride) * stride
    r = min(new_h / h0, new_w / w0)  # update scale
    
    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    
    # Pad to exact size
    dh, dw = size[0] - img.shape[0], size[1] - img.shape[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (left, top)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    orig_h, orig_w = frame.shape[:2]

    # Letterbox đúng 320x320
    img_rgb, scale, pad = letterbox(frame, size=(input_w, input_h))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    # Normalize & expand_dims -> [1,320,320,3]
    img = img_rgb.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    print(f"Input shape now: {img.shape}")  # Confirm [1,320,320,3]

    # Inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    inf_time = time.time() - start
    fps = 1 / inf_time

    outputs = interpreter.get_tensor(output_details[0]['index'])[0]
    print(f"Output shape: {outputs.shape}")  # Debug 1 lần

    for detection in outputs:
        conf = detection[4]
        if conf < conf_thres:
            continue
        scores = detection[5:]
        cls_id = np.argmax(scores)
        score = conf * scores[cls_id]
        if score < conf_thres:
            continue

        # Bbox decode (normalized xywh)
        cx, cy, bw, bh = detection[:4]
        x1 = (cx - bw/2) * orig_w
        y1 = (cy - bh/2) * orig_h
        x2 = (cx + bw/2) * orig_w
        y2 = (cy + bh/2) * orig_h

        # Remove pad offset
        x1 = (x1 - pad[0]) / scale
        y1 = (y1 - pad[1]) / scale
        x2 = (x2 - pad[0]) / scale
        y2 = (y2 - pad[1]) / scale

        # Clamp
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))

        label = class_names[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(y1-10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("YOLO Fixed 320x320", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
