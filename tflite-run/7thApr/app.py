import cv2
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load model TFLite của bạn (Hoặc dùng "yolov8n.pt" nếu muốn chạy bản chuẩn)
model = YOLO("coco_yolov8n_int8.tflite")

# Biến toàn cục để nhận thông số từ thanh trượt trên Web
params = {"conf": 0.5, "iou": 0.45}

def generate_frames():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 0 là camera mặc định
    # BỔ SUNG: Ép camera USB chạy ở độ phân giải HD (1280x720) cho nét
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Gọi hàm predict của Ultralytics với thông số từ Web
        results = model.predict(
            source=frame, 
            conf=params["conf"], 
            iou=params["iou"], 
            stream=False,
            verbose=False # Tắt log thừa
        )
        
        # Ultralytics TỰ ĐỘNG VẼ BBOX SIÊU ĐẸP
        annotated_frame = results[0].plot()
        
        # Mã hóa frame thành JPEG để gửi lên Web
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield video stream liên tục
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # API phát video luồng
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_params', methods=['POST'])
def update_params():
    # API nhận thông số từ thanh trượt HTML gửi về
    data = request.json
    params["conf"] = float(data.get("conf", 0.5))
    params["iou"] = float(data.get("iou", 0.45))
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)