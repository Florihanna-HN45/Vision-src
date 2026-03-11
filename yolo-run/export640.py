from ultralytics import YOLO

# Load model gốc .pt (model bạn đã train 640)
model = YOLO(r"C:\Users\This pc\Downloads\pdisease1 (2).pt")

# Export sang tflite với định dạng int8 và ép kích thước 640
# 'data' là file .yaml bạn dùng khi train (cần thiết để định lượng INT8 chuẩn xác)
model.export(format="tflite", imgsz=640, int8=True, data="path/to/your/data.yaml")