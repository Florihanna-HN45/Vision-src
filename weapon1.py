import cv2
import numpy as np
import tensorflow as tf
import time

# 1. Load model TFLite
model_path = "best-int8.tflite"
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# Kiem tra kieu du lieu dau vao (UINT8 cho INT8 model)
is_int8 = input_details[0]['dtype'] == np.uint8

# 2. Mo Webcam
cap = cv2.VideoCapture(0)
print("Camera is running... Press 'q' to quit.") 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- TIEN XU LY ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height))
    input_data = np.expand_dims(img_resized, axis=0)

    if is_int8:
        input_data = input_data.astype(np.uint8) # Khong chia 255
    else:
        input_data = input_data.astype(np.float32) / 255.0

    # --- SUY LUAN ---
    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Giai nen dau ra neu la model INT8
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    t2 = time.time()

    # --- XU LY KET QUA ---
    conf_threshold = 0.4
    # Loc box co do tin cay > threshold
    detections = output[output[:, 4] > conf_threshold]

    for det in detections:
        # Lay class_id va tinh toan toa do
        classes = det[5:]
        class_id = np.argmax(classes)
        conf = det[4]
        
        # YOLOv5 output: center_x, center_y, width, height (0-1)
        cx, cy, w, h = det[0:4]
        
        # Chuyen sang pixel
        x1 = int((cx - w/2) * frame.shape[1])
        y1 = int((cy - h/2) * frame.shape[0])
        x2 = int((cx + w/2) * frame.shape[1])
        y2 = int((cy + h/2) * frame.shape[0])

        # Ve len man hinh
        label = f"ID:{class_id} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hien thi FPS
    fps = 1 / (t2 - t1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Robocon 2026 - Weapon Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()