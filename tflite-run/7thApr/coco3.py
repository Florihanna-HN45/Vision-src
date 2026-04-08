import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

################## PHAN DINH NGHIA FUNCTION #######################

def process_tflite_output(output_data, output_details, w, h, conf_threshold=0.3):
    """Hàm xử lý ma trận thô của YOLOv8 TFLite thành tọa độ Bbox"""
    # Giải lượng tử nếu model là Int8
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    outputs = output_data[0].T
    boxes, scores, class_ids = [], [], []

    # Tỷ lệ ánh xạ (do ảnh AI là 640x640, ảnh gốc có thể khác)
    x_factor = w / 640.0
    y_factor = h / 640.0

    for row in outputs:
        classes_scores = row[4:]
        max_score = np.max(classes_scores)
        
        if max_score >= conf_threshold:
            class_id = np.argmax(classes_scores)
            cx, cy, box_w, box_h = row[0], row[1], row[2], row[3]
            
            x_min = int((cx - (box_w / 2)) * x_factor)
            y_min = int((cy - (box_h / 2)) * y_factor)
            
            boxes.append([x_min, y_min, int(box_w * x_factor), int(box_h * y_factor)])
            scores.append(float(max_score))
            class_ids.append(class_id)

    # Lọc Bbox trùng lặp
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.4)
    
    final_boxes = [boxes[i] for i in indices] if len(indices) > 0 else []
    final_scores = [scores[i] for i in indices] if len(indices) > 0 else []
    final_ids = [class_ids[i] for i in indices] if len(indices) > 0 else []
    
    return final_boxes, final_scores, final_ids

################ CHUONG TRINH CHINH ##############################

def main():
    # 1. Tải mô hình TFLite (YOLOv8 của bài toán AMR)
    interpreter = tflite.Interpreter(model_path="yolov8n_int8.tflite", num_threads=3)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Lấy ảnh từ Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Chống lag buffer

    # Cấu hình thông số
    FRAMES_TO_SKIP = 3
    frame_counter = 0
    last_boxes, last_scores, last_class_ids = [], [], []

    print("[INFO] Hệ thống Object Detection khởi động. Nhấn 'q' để thoát.")

    while True:
        start_time = time.time()
        
        # Đọc ảnh từ webcam
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        orig_h, orig_w = frame.shape[:2]
        frame_counter += 1

        if frame_counter % (FRAMES_TO_SKIP + 1) != 0:
            # Dùng lại kết quả cũ để khung hình mượt mà
            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
        else:
            # --- Tiền xử lý cho YOLOv8 ---
            img_ai = cv2.resize(frame, (640, 640))
            img_ai = img_ai.astype(np.float32) / 255.0
            img_ai = np.expand_dims(img_ai, axis=0)

            # Ép kiểu Int8 nếu model yêu cầu
            if input_details[0]['dtype'] == np.int8:
                scale, zero_point = input_details[0]['quantization']
                img_ai = (img_ai / scale + zero_point).astype(np.int8)

            # Chạy dự đoán
            interpreter.set_tensor(input_details[0]["index"], img_ai)
            interpreter.invoke()

            # Lấy và xử lý kết quả đầu ra
            output_data = interpreter.get_tensor(output_details[0]["index"])
            last_boxes, last_scores, last_class_ids = process_tflite_output(
                output_data, output_details[0], orig_w, orig_h
            )
            
            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
            frame_counter = 0

        # --- Vẽ Bbox lên vật thể ---
        for i in range(len(boxes_to_draw)):
            x, y, w, h = boxes_to_draw[i]
            
            # Vẽ khung chữ nhật xanh lá
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Hiển thị ID class và độ tự tin
            label = f"ID:{ids_to_draw[i]} {scores_to_draw[i]:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Chú ý: Đã bỏ cv2.flip() vì trên Robot AMR, trái phải phải chuẩn xác.
        cv2.imshow("AMR Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()