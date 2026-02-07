import cv2
import numpy as np
import tensorflow as tf
import time

model_path = r"D:\Unarrage\7Feb.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h, input_w = input_details[0]['shape'][1:3]
print("TFLite model loaded!")
class_names = [
    "class0",
    "class1",
    "class2"
]
#GIF
gif_path = r"C:\Downloads\7.gif"
cap = cv2.VideoCapture(gif_path)
#URL for IP camera
# url =""
# cap = cv2.VideoCapture(url)
print("Camera is running...")
conf_thres = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0, _ = frame.shape

    # --- Preprocess ---
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    if input_details[0]['dtype'] == np.uint8:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32) / 255.0

    # --- Inference ---
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    fps = 1 / (time.time() - start)

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    for p in output:
        obj = p[4]
        if obj < conf_thres:
            continue

        cls_id = np.argmax(p[5:])
        score = obj * p[5 + cls_id]
        if score < conf_thres:
            continue

        x, y, w, h = p[:4]

        x1 = int((x - w/2) * w0)
        y1 = int((y - h/2) * h0)
        x2 = int((x + w/2) * w0)
        y2 = int((y + h/2) * h0)

        label = class_names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLO TFLite Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
