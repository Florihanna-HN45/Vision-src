import cv2
import numpy as np
import onnxruntime as ort
import time

# 1. Khởi tạo phiên làm việc với ONNX Runtime
model_path = r"D:\Unarrage\model_int8_qdq.onnx"
# Sử dụng 'CPUExecutionProvider' hoặc 'CUDAExecutionProvider' nếu có GPU
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Lấy thông tên và thông số đầu vào của mô hình
input_info = session.get_inputs()[0]
input_shape = input_info.shape
input_name = input_info.name
width = 512
height = 512
# 2. Mở Camera
cap = cv2.VideoCapture(0)

print(f"Model input: {input_name}, Shape: {input_shape}")
prev_frame_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- BƯỚC 3: PRE-PROCESSING (Rất quan trọng) ---
    # Chuyển BGR (OpenCV) sang RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize về kích thước model yêu cầu
    img = cv2.resize(img, (width, height))
    # Chuyển từ HWC (Height, Width, Channel) sang CHW (Channel, Height, Width)
    img = img.transpose(2, 0, 1)
    # Thêm batch dimension và chuyển sang float32 (hoặc tùy model)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    # Normalize (ví dụ chia 255)
    img /= 255.0

    # --- BƯỚC 4: INFERENCE VỚI ONNX RUNTIME ---
    outputs = session.run(None, {input_name: img})	
    new_frame_time = time.time()
    
    # Tính toán thời gian chênh lệch
    time_diff = new_frame_time - prev_frame_time
    if time_diff > 0:
        fps = 1 / time_diff
    else:
        fps = 0
        
    prev_frame_time = new_frame_time
	
    
    # --- BƯỚC 5: HẬU XỬ LÝ ---
    # Kết quả trả về là một list các mảng numpy
    result = outputs[0]
    class_id = np.argmax(result)
    
    # Hiển thị lên màn hình
    cv2.putText(frame, f"fps: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ORT Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()