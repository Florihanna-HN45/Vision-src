import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Khởi tạo model bằng PyTorch gốc cho lẹ (không cần TFLite trên PC)
    print("Đang load model YOLOv8n coco...")
    model = YOLO(r"D:\Unarrage\coco_yolov8n_int8.tflite") 

    # 2. Khởi tạo Webcam (0 là camera mặc định của PC/Laptop)
    cap = cv2.VideoCapture(0)
    
    # 3. Cấu hình Frame Skipping
    FRAMES_TO_SKIP = 3  # Bỏ qua 3 frame, chạy AI ở frame thứ 4
    frame_counter = 0

    # Biến lưu trữ Bbox của lần chạy AI gần nhất
    last_boxes = []

    print("Bắt đầu test. Nhấn 'q' để thoát.")

    while True:
        start_time = time.time()
        frame_counter += 1

        if frame_counter % (FRAMES_TO_SKIP + 1) != 0:
            # === FRAME BỎ QUA AI ===
            # Vẫn đọc frame để video trôi chảy, nhưng KHÔNG đưa vào model
            ret, frame = cap.read()
            if not ret: break
            
            # Tái sử dụng Bbox cũ
            boxes_to_draw = last_boxes
            
        else:
            # === FRAME CHẠY AI ===
            # Kỹ thuật dọn dẹp Buffer: Quét sạch các frame cũ bị kẹt trong hàng đợi
            for _ in range(5):
                cap.grab()
            
            ret, frame = cap.read()
            if not ret: break

            # Chạy AI (set verbose=False để terminal không bị spam chữ)
            results = model.predict(frame, imgsz=640, verbose=False)
            
            # Trích xuất dữ liệu Bbox, Confidence, và Tên Class
            temp_boxes = []
            for box in results[0].boxes:
                # Ép kiểu tọa độ về số nguyên
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Lấy tên class từ ID
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                
                temp_boxes.append((x1, y1, x2, y2, conf, name))
            
            # Cập nhật kết quả mới nhất
            last_boxes = temp_boxes
            boxes_to_draw = last_boxes
            
            # Reset biến đếm
            frame_counter = 0

        # 4. Vẽ Bounding Box và Thông tin lên ảnh
        for (x1, y1, x2, y2, conf, name) in boxes_to_draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tính toán FPS của luồng Video
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"Video FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Hiển thị trạng thái hoạt động của thuật toán góc trái bên dưới
        status_text = "AI COMPUTING" if frame_counter == 0 else f"SKIPPING ({frame_counter}/{FRAMES_TO_SKIP})"
        status_color = (0, 0, 255) if frame_counter == 0 else (255, 0, 0)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 5. Show kết quả
        cv2.imshow("PC Test - Frame Skipping", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()