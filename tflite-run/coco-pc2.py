import cv2
import numpy as np
import time
import os

# Cài đặt TensorFlow
try:
    import tensorflow as tf
    print(" TensorFlow imported")
except ImportError:
    print(" TensorFlow not found. Install: pip install tensorflow")
    exit(1)

# ==========================================
# 1. HÀM LETTERBOX (Tạo viền xám, chống méo ảnh)
# ==========================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    
    shape = im.shape[:2]  # (H, W)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    print(f"[Letterbox] Shape: {shape} -> Padded: {im.shape[:2]}, Ratio: {r:.3f}, Pad: L={left} T={top}")
    
    return im, r, (left, top)

# ==========================================
# 2. HÀM XỬ LÝ OUTPUT TFLite
# ==========================================
def process_tflite_output_letterbox(output_data, output_details, ratio, pad_info, conf_threshold=0.3, debug=False):
    
    pad_left, pad_top = pad_info
    
    # === BƯỚC 1: Dequantize nếu là Int8 ===
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale
        if debug:
            print(f"[Dequant] Scale: {scale}, Zero-point: {zero_point}")

    # === BƯỚC 2: Parse output ===
    outputs = output_data[0].T  # Shape: (num_detections, 6+num_classes)
    
    if debug:
        print(f"[Output] Shape: {outputs.shape}, First row sample: {outputs[0][:6]}")
    
    boxes, scores, class_ids = [], [], []
    detections_raw = []

    for idx, row in enumerate(outputs):
        classes_scores = row[4:]
        max_score = np.max(classes_scores)
        
        if max_score >= conf_threshold:
            class_id = np.argmax(classes_scores)
            
            # Tọa độ từ model (center_x, center_y, width, height)
            cx, cy, box_w, box_h = row[0], row[1], row[2], row[3]
            
            detections_raw.append({
                'idx': idx,
                'cx_pad': cx, 'cy_pad': cy,
                'w_pad': box_w, 'h_pad': box_h,
                'score': max_score,
                'class_id': class_id
            })
            
            # === BƯỚC 3: Chuyển đổi tọa độ ===
            # 1. Bỏ viền xám
            cx_unpad = cx - pad_left
            cy_unpad = cy - pad_top
            
            # 2. Phóng to về kích thước gốc
            cx_orig = cx_unpad / ratio
            cy_orig = cy_unpad / ratio
            w_orig = box_w / ratio
            h_orig = box_h / ratio
            
            # 3. Từ center -> top-left
            x_min = int(cx_orig - (w_orig / 2))
            y_min = int(cy_orig - (h_orig / 2))
            w_final = int(w_orig)
            h_final = int(h_orig)
            
            if debug and len(detections_raw) <= 3:  # In 3 detection đầu tiên
                print(f"[Detection {idx}] Score: {max_score:.3f} | Class: {class_id}")
                print(f"  Padded: cx={cx:.1f} cy={cy:.1f} w={box_w:.1f} h={box_h:.1f}")
                print(f"  Unpadded: cx={cx_unpad:.1f} cy={cy_unpad:.1f}")
                print(f"  Original: cx={cx_orig:.1f} cy={cy_orig:.1f} w={w_orig:.1f} h={h_orig:.1f}")
                print(f"  Final Box: x={x_min} y={y_min} w={w_final} h={h_final}")
            
            # Lọc ngoài biên
            if x_min >= 0 and y_min >= 0 and w_final > 0 and h_final > 0:
                boxes.append([x_min, y_min, w_final, h_final])
                scores.append(float(max_score))
                class_ids.append(class_id)

    if debug:
        print(f"[Summary] Raw detections: {len(detections_raw)}, After filtering: {len(boxes)}")

    # === BƯỚC 4: NMS (Non-Maximum Suppression) ===
    if len(boxes) > 0:
        try:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.4)
            final_boxes = [boxes[i] for i in indices] if len(indices) > 0 else []
            final_scores = [scores[i] for i in indices] if len(indices) > 0 else []
            final_ids = [class_ids[i] for i in indices] if len(indices) > 0 else []
            
            if debug:
                print(f"[NMS] Before: {len(boxes)}, After: {len(final_boxes)}")
        except Exception as e:
            print(f"[NMS Error] {e}")
            final_boxes, final_scores, final_ids = boxes, scores, class_ids
    else:
        final_boxes, final_scores, final_ids = [], [], []
    
    return final_boxes, final_scores, final_ids

# ==========================================
# 3. LOAD MODEL
# ==========================================
def load_model(model_path):
    
    if not os.path.exists(model_path):
        print(f" Model not found: {model_path}")
        print(f"  Current dir: {os.getcwd()}")
        print(f"  Available files: {os.listdir('.')}")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path="./7thApr/coco_yolov8n_int8.tflite")
        interpreter.allocate_tensors()
        print(f" Model loaded: {model_path}")
        return interpreter
    except Exception as e:
        print(f" Model load error: {e}")
        return None

# ==========================================
# 4. CHUỖI COCO CLASS NAMES
# ==========================================
COCO_CLASSES = {
    0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle',
    4: 'Airplane', 5: 'Bus', 6: 'Train', 7: 'Truck',
    8: 'Boat', 9: 'Traffic light', 10: 'Fire hydrant', 11: 'Stop sign',
    12: 'Parking meter', 13: 'Bench', 14: 'Cat', 15: 'Dog',
    16: 'Horse', 17: 'Sheep', 18: 'Cow', 19: 'Elephant'
}

# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    print("=" * 60)
    print("  YOLO TFLite PC Test - Full Debug Version")
    print("=" * 60)
    
    # === CẤU HÌNH ===
    MODEL_PATH = "./7thApr/coco_yolov8n_int8.tflite"  # Thay bằng path của cậu
    INPUT_SIZE = 640
    CONF_THRESHOLD = 0.3
    FRAMES_TO_SKIP = 3  # Xử lý mỗi 4 frame (3 skip + 1 inference)
    DEBUG_MODE = True   # In chi tiết lần đầu
    
    # === LOAD MODEL ===
    interpreter = load_model(MODEL_PATH)
    if interpreter is None:
        print("\n💡model path")
        return
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n[Model Info]")
    print(f"  Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
    
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        print(f"  Input quantization: scale={scale}, zero_point={zero_point}")
    
    # === MỞ WEBCAM ===
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open webcam. Try: sudo modprobe bcm2835_v4l2 (on Pi)")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Webcam opened")
    
    # === BIẾN TOÀN CỤC ===
    frame_counter = 0
    last_boxes, last_scores, last_class_ids = [], [], []
    frame_idx = 0
    first_inference = True
    
    print("\n[Controls]")
    print("  Press 'Q' to quit")
    print("  Press 'D' to toggle debug mode")
    print("  Press 'S' to save frame")
    print("\nStarting...\n")
    
    while True:
        start_time = time.time()
        
        # === XỔ BUFFER LẦN ĐẦU ===
        if frame_counter == 0:
            for _ in range(5):
                cap.grab()

        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to read frame")
            break

        frame_counter += 1
        frame_idx += 1
        original_frame = frame.copy()

        # === QUY ĐỊNH: Chạy inference mỗi 4 frame ===
        if frame_counter % (FRAMES_TO_SKIP + 1) != 0:
            # Dùng kết quả cũ
            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
            is_new_inference = False
        else:
            # === TIỀN XỬ LÝ (Pre-processing) ===
            img_ai, ratio, pad_info = letterbox(frame, new_shape=(INPUT_SIZE, INPUT_SIZE))
            img_ai = cv2.cvtColor(img_ai, cv2.COLOR_BGR2RGB)
            img_ai = img_ai.astype(np.float32) / 255.0
            img_ai = np.expand_dims(img_ai, axis=0)

            # Quantize if needed
            if input_details[0]['dtype'] == np.int8:
                scale, zero_point = input_details[0]['quantization']
                img_ai = (img_ai / scale + zero_point).astype(np.int8)

            # === INFERENCE ===
            inference_start = time.time()
            interpreter.set_tensor(input_details[0]["index"], img_ai)
            interpreter.invoke()
            inference_time = time.time() - inference_start

            # === HẬU XỬ LÝ (Post-processing) ===
            output_data = interpreter.get_tensor(output_details[0]["index"])
            last_boxes, last_scores, last_class_ids = process_tflite_output_letterbox(
                output_data, output_details[0], ratio, pad_info, 
                conf_threshold=CONF_THRESHOLD, 
                debug=(DEBUG_MODE and first_inference)
            )
            
            boxes_to_draw = last_boxes
            scores_to_draw = last_scores
            ids_to_draw = last_class_ids
            frame_counter = 0
            is_new_inference = True
            first_inference = False

        # === VẼ BOUNDING BOX ===
        for i in range(len(boxes_to_draw)):
            x, y, w, h = boxes_to_draw[i]
            
            # Vẽ box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Vẽ label
            cls_id = ids_to_draw[i]
            name = COCO_CLASSES.get(cls_id, f"ID:{cls_id}")
            label = f"{name} {scores_to_draw[i]:.2f}"
            
            # Background cho text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # === THÔNG TIN TRÊN MÀN HÌNH ===
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time
        
        status = f"Inference (Frame {frame_counter})" if is_new_inference else f"Tracking ({frame_counter}/{FRAMES_TO_SKIP})"
        status_color = (0, 0, 255) if is_new_inference else (255, 0, 0)
        
        # Dòng 1: FPS & Status
        cv2.putText(frame, f"FPS: {fps:.1f} | {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Dòng 2: Detections
        cv2.putText(frame, f"Detections: {len(boxes_to_draw)} | Frame: {frame_idx}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dòng 3: Model input size
        cv2.putText(frame, f"Input: {INPUT_SIZE}x{INPUT_SIZE} | Conf: {CONF_THRESHOLD}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # === HIỂN THỊ ===
        cv2.imshow("YOLO TFLite - PC Test", frame)

        # === XỬ LÝ PHÍM ===
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[Exit] Quit by user")
            break
        elif key == ord('d') or key == ord('D'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"[Debug] Mode: {'ON' if DEBUG_MODE else 'OFF'}")
        elif key == ord('s') or key == ord('S'):
            filename = f"frame_{frame_idx}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[Save] Frame saved: {filename}")

    # === CLEANUP ===
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Cleanup completed")

if __name__ == "__main__":
    main()