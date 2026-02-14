import cv2
import numpy as np
import tensorflow as tf
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letterbox(img, size=(320, 320), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(size[0] / h0, size[1] / w0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    img_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size[1] - new_w, size[0] - new_h
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img_pad = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_pad, r, (left, top)

# --- CONFIG ---
model_path = r"D:\Unarrage\7Feb.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = input_details[0]['shape'][1], input_details[0]['shape'][2] 

# Danh sach class (Tam thoi de nua de tranh crash)
class_names = ["class0", "class1", "class2"]
url ="http://192.168.1.123:8080/video"
cap = cv2.VideoCapture(url)
conf_thres = 0.5 
nms_thres = 0.4 

print("--- He thong bat dau ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    orig_h, orig_w = frame.shape[:2]

    # Preprocessing
    img_pad, scale, (pad_w, pad_h) = letterbox(frame, size=(input_w, input_h))
    gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.merge([gray, gray, gray]) 
    img = img_rgb.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Auto-shape
    output = np.squeeze(raw_output)
    if output.shape[0] < output.shape[1]: 
        output = output.T
    
    # Check thuc te model co bao nhieu class
    actual_num_classes = output.shape[1] - 4
    if len(class_names) < actual_num_classes:
        # Tu dong dien them ten class de khong bi IndexError
        for i in range(len(class_names), actual_num_classes):
            class_names.append(f"Unknown_{i}")

    if np.max(output[:, 4:]) > 1.0:
        output[:, 4:] = sigmoid(output[:, 4:])

    boxes, confs, class_ids = [], [], []
    for detection in output:
        scores = detection[4:]
        cls_id = np.argmax(scores)
        score = scores[cls_id]

        if score > conf_thres:
            cx, cy, bw, bh = detection[:4]
            x1 = (cx - bw/2 - pad_w) / scale
            y1 = (cy - bh/2 - pad_h) / scale
            boxes.append([int(x1), int(y1), int(bw/scale), int(bh/scale)])
            confs.append(float(score))
            class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thres, nms_thres)

    if len(indices) > 0:
        # Fix cho mot so phien ban OpenCV tra ve list hoac numpy array khac nhau
        idx_list = indices.flatten() if hasattr(indices, 'flatten') else indices
        for i in idx_list:
            x, y, w, h = boxes[i]
            # Kiem tra index mot lan nua truoc khi hien thi
            cid = class_ids[i]
            label = class_names[cid] if cid < len(class_names) else f"ID_{cid}"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confs[i]:.2f}", (x, max(y-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Debug Robocon", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()