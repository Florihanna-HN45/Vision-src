import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

def main():
    print("[INFO] Khởi động hệ thống Vision cho AMR trên Raspberry Pi 4...")
    
    # ---------------------------------------------------------
    # 1. KHỞI TẠO TFLITE VỚI TỐI ƯU HÓA CPU THREADING
    # Pi 4 có 4 nhân CPU. Ta dùng 3 nhân cho AI, chừa 1 nhân cho Camera/OS/LiDAR
    # ---------------------------------------------------------
    model_path = "yolov8n_int8.tflite"
    try:
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=3)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"[ERROR] Không thể load model. Hãy chắc chắn file {model_path} đang ở cùng thư mục.")
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # ---------------------------------------------------------
    # 2. KHỞI TẠO CAMERA VỚI CHUẨN V4L2
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(0)
    
    # Ép Pi 4 chỉ đọc độ phân giải mặc định (thường là 640x480) để tiết kiệm RAM.
    # BÍ QUYẾT: Giới hạn buffer = 1 để chống Delay (luôn lấy ảnh tươi nhất)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    if not cap.isOpened():
        print("[ERROR] Không thể kết nối với Camera USB.")
        return

    # ---------------------------------------------------------
    # 3. CẤU HÌNH THÔNG SỐ VẬN HÀNH
    # ---------------------------------------------------------
    FRAMES_TO_SKIP = 5       # Chạy AI 1 lần, lướt 5 lần (Giúp màn hình đạt ~20-25 FPS)
    CONF_THRESHOLD = 0.3     # Hạ ngưỡng tự tin vì Int8 bị nén
    NMS_THRESHOLD = 0.4      # Ngưỡng lọc Bbox trùng lặp
    
    frame_counter = 0

    # Các biến lưu trữ trạng thái của AI để vẽ mượt mà
    last_boxes = []
    last_scores = []
    last_class_ids = []

    print("[INFO] Hệ thống sẵn sàng. Nhấn 'q' trên cửa sổ video để thoát.")

    while True:
        start_time = time.time()
        frame_counter += 1

        # Lấy ảnh từ Camera
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.01) # Nghỉ 10ms nếu cam chớp tắt thay vì văng app
            continue

        # Lấy kích thước thực tế của khung hình (thường là w=640, h=480)
        h, w = frame.shape[:2]

        if frame_counter % (FRAMES_TO_SKIP + 1) != 0:
            # ==========================================
            # LUỒNG 1: FRAME BỎ QUA (HIỂN THỊ UI MƯỢT MÀ)
            # ==========================================
            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
            
        else:
            # ==========================================
            # LUỒNG 2: FRAME CHẠY AI (XỬ LÝ TFLITE)
            # ==========================================
            
            # --- Tiền xử lý (Pre-processing) ---
            # Chỉ resize bản nháp (img_ai) để AI đọc, KHÔNG đụng vào biến 'frame' hiển thị
            img_ai = cv2.resize(frame, (640, 640))
            img_ai = img_ai.astype(np.float32) / 255.0
            img_ai = np.expand_dims(img_ai, axis=0) # Chuẩn NHWC: [1, 640, 640, 3]

            # Lượng tử hóa input nếu model yêu cầu thuần Int8
            if input_details['dtype'] == np.int8:
                scale, zero_point = input_details['quantization']
                img_ai = (img_ai / scale + zero_point).astype(np.int8)

            # --- Gọi AI Inference ---
            interpreter.set_tensor(input_details['index'], img_ai)
            interpreter.invoke()

            # --- Hậu xử lý (Post-processing) ---
            output_data = interpreter.get_tensor(output_details['index'])

            # Giải lượng tử (De-quantize) từ Int8 về Float32 để tính toán Bbox không bị méo
            if output_details['dtype'] == np.int8:
                scale, zero_point = output_details['quantization']
                output_data = (output_data.astype(np.float32) - zero_point) * scale

            # Ma trận [1, 84, 8400] chuyển vị thành [8400, 84]
            outputs = output_data[0].T 
            
            temp_boxes, temp_scores, temp_ids = [], [], []

            # Tỷ lệ để ánh xạ tọa độ từ 640x640 về lại kích thước màn hình thật (w, h)
            x_factor = w / 640.0
            y_factor = h / 640.0

            for row in outputs:
                classes_scores = row[4:] # Từ cột 4 trở đi là xác suất của các class
                max_score = np.max(classes_scores)
                
                if max_score >= CONF_THRESHOLD:
                    class_id = np.argmax(classes_scores)
                    cx, cy, box_w, box_h = row[0], row[1], row[2], row[3]
                    
                    # Giải mã tọa độ Tâm (cx, cy) thành Góc trên cùng bên trái (x_min, y_min)
                    # và nhân tỷ lệ trả về màn hình thực
                    x_min = int((cx - (box_w / 2)) * x_factor)
                    y_min = int((cy - (box_h / 2)) * y_factor)
                    actual_w = int(box_w * x_factor)
                    actual_h = int(box_h * y_factor)
                    
                    temp_boxes.append([x_min, y_min, actual_w, actual_h])
                    temp_scores.append(float(max_score))
                    temp_ids.append(class_id)

            # Lọc Bbox trùng lặp bằng thuật toán NMS của OpenCV C++ Core
            indices = cv2.dnn.NMSBoxes(temp_boxes, temp_scores, CONF_THRESHOLD, NMS_THRESHOLD)
            
            # Lưu trữ lại kết quả để dùng cho 5 frame tiếp theo
            last_boxes = [temp_boxes[i] for i in indices] if len(indices) > 0 else []
            last_scores = [temp_scores[i] for i in indices] if len(indices) > 0 else []
            last_class_ids = [temp_ids[i] for i in indices] if len(indices) > 0 else []

            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
            
            frame_counter = 0 # Reset bộ đếm

        # ==========================================
        # 4. VẼ KẾT QUẢ & HIỂN THỊ LÊN MÀN HÌNH
        # ==========================================
        for i in range(len(boxes_to_draw)):
            x, y, box_w, box_h = boxes_to_draw[i]
            
            # Vẽ hộp chữ nhật (Màu Cyan nổi bật)
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 255, 0), 2)
            
            # Gắn nhãn
            label = f"ID:{ids_to_draw[i]} {scores_to_draw[i]:.2f}"
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Tính toán và hiển thị thông số hệ thống
        fps = 1.0 / (time.time() - start_time)
        status = "AI Computing" if frame_counter == 0 else f"Tracking {frame_counter}/5"
        color = (0, 0, 255) if frame_counter == 0 else (0, 255, 0) # Đỏ khi AI chạy, Xanh khi Tracking
        
        cv2.putText(frame, f"FPS: {fps:.1f} | {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Hiển thị
        cv2.imshow("AMR Vision Module - Pi 4", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã tắt hệ thống Vision.")

if __name__ == "__main__":
    main()