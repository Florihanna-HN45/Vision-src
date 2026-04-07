import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

def main():
    # 1. Khởi tạo Model (Giữ nguyên như cũ)
    model_path = "yolov8n_int8.tflite"
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=3) # Dùng 3 luồng, chừa 1 luồng cho Camera/LiDAR
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 2. Khởi tạo Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 3. Cấu hình Frame Skipping
    FRAMES_TO_SKIP = 3  # Bỏ qua 3 frame, chạy AI ở frame thứ 4
    frame_counter = 0

    # Các biến lưu lại kết quả của lần chạy AI gần nhất để vẽ cho mượt
    last_boxes = []
    last_scores = []
    last_class_ids = []

    print("Bắt đầu chạy luồng Camera Real-time...")

    while True:
        start_time = time.time()
        
        # Đếm số frame
        frame_counter += 1

        if frame_counter % (FRAMES_TO_SKIP + 1) != 0:
            # === CÁC FRAME BỊ BỎ QUA AI ===
            # Vẫn đọc ảnh ra để hiển thị cho video mượt, nhưng KHÔNG giải mã toàn bộ nếu không cần thiết
            ret, frame = cap.read()
            if not ret:
                break
            
            # Khúc này KHÔNG chạy TFLite, chỉ lấy Bbox cũ ra vẽ lại
            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
            
        else:
            # === FRAME ĐƯỢC CHỌN ĐỂ CHẠY AI ===
            # Bước này rất quan trọng: Dọn dẹp sạch buffer bị dồn ứ trong lúc chạy AI vòng trước
            # grab() cực kỳ nhẹ vì nó chỉ "bỏ qua" frame ở mức hardware
            for _ in range(5): 
                cap.grab() 
            
            # Lấy frame hiện tại (mới nhất)
            ret, frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame.shape[:2]

            # --- Tiền xử lý & Chạy TFLite (Y hệt code cũ) ---
            img = cv2.resize(frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            if input_details['dtype'] == np.int8:
                scale, zero_point = input_details['quantization']
                img = (img / scale + zero_point).astype(np.int8)

            interpreter.set_tensor(input_details['index'], img)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details['index'])

            if output_details['dtype'] == np.int8:
                scale, zero_point = output_details['quantization']
                output_data = (output_data.astype(np.float32) - zero_point) * scale

            outputs = output_data[0].T
            
            # --- Hậu xử lý (Lưu vào biến tạm) ---
            temp_boxes, temp_scores, temp_ids = [], [], []
            x_factor = orig_w / 640.0
            y_factor = orig_h / 640.0

            for row in outputs:
                classes_scores = row[4:]
                max_score = np.max(classes_scores)
                if max_score >= 0.5:
                    class_id = np.argmax(classes_scores)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    x_min = int((cx - (w / 2)) * x_factor)
                    y_min = int((cy - (h / 2)) * y_factor)
                    temp_boxes.append([x_min, y_min, int(w * x_factor), int(h * y_factor)])
                    temp_scores.append(float(max_score))
                    temp_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(temp_boxes, temp_scores, 0.5, 0.4)
            
            # Cập nhật kết quả AI mới nhất vào biến toàn cục
            last_boxes = [temp_boxes[i] for i in indices] if len(indices) > 0 else []
            last_scores = [temp_scores[i] for i in indices] if len(indices) > 0 else []
            last_class_ids = [temp_ids[i] for i in indices] if len(indices) > 0 else []

            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids

            # Reset counter để tránh tràn số
            frame_counter = 0

        # 4. Vẽ Bbox lên frame (dù là ảnh mới chạy AI hay ảnh skip)
        for i in range(len(boxes_to_draw)):
            x, y, w, h = boxes_to_draw[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {ids_to_draw[i]}: {scores_to_draw[i]:.2f}", 
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tính toán và in ra FPS thực tế của màn hình
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("AMR Camera - Frame Skipping", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()