import cv2
from ultralytics import YOLO

# 1. Load file TFLite vừa export
model = YOLO(r"D:\Unarrage\coco_yolov8n_int8.tflite")

# 2. TỰ ĐỘNG LẤY LABEL TỪ FILE MODEL
# TFLite do Ultralytics export đã lưu sẵn mảng dictionary này.
# Cấu trúc sẽ là dạng: {0: 'person', 1: 'bicycle', 2: 'car', ...}
class_names = model.names 
print("Đã lấy được danh sách labels từ model:", class_names)

# 3. Chạy inference lấy kết quả thô
# Cậu có thể truyền đường dẫn file ảnh thật của cậu vào đây
image_path = "https://ultralytics.com/images/bus.jpg"
results = model.predict(source=image_path, imgsz=640)

# 4. Đọc ảnh bằng OpenCV để tự vẽ
# Nếu dùng ảnh trên mạng như URL kia, OpenCV cần tải về, 
# nhưng ở đây giả sử tớ dùng file lưu trên máy cho thực tế:
# img = cv2.imread("bus.jpg") 
img = results[0].orig_img # Lấy luôn mảng ảnh numpy gốc từ biến results cho tiện

# 5. Bóc tách dữ liệu và vẽ tay bằng OpenCV
boxes = results[0].boxes # Lấy object chứa bounding box

for box in boxes:
    # Lấy tọa độ [x_min, y_min, x_max, y_max] và ép về số nguyên
    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
    
    # Lấy độ tự tin (Confidence score)
    conf = float(box.conf[0])
    
    # Lấy ID của class và tra ngược lại tên class từ dictionary
    class_id = int(box.cls[0])
    label_name = class_names[class_id]
    
    # Tạo chuỗi text hiển thị (VD: "person 0.89")
    display_text = f"{label_name} {conf:.2f}"
    
    # --- BẮT ĐẦU VẼ BẰNG OPENCV ---
    
    # Vẽ khung chữ nhật (Màu xanh lá, viền dày 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Tính toán kích thước chữ để vẽ background cho text dễ nhìn
    (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
    
    # Ghi chữ lên trên background vừa vẽ
    cv2.putText(img, display_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# 6. Hiển thị kết quả ra màn hình
cv2.imshow("Custom TFLite Inference", img)
cv2.waitKey(0)
cv2.destroyAllWindows()