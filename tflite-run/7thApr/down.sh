#!/bin/bash

# ============================================
# Fix: Download Model Chính Xác từ UltraYOLOv8
# ============================================

echo "🚀 Downloading correct YOLOv8n INT8 TFLite model..."
echo

# Method 1: Direct download from UltraYOLOv8 releases
echo "📥 Downloading from UltraYOLOv8 official release..."

# TFLite model URL (nano model, INT8 quantized)
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-int8.tflite"

# Alternative URLs (choose one if above fails)
# MODEL_URL="https://github.com/ultralytics/yolov8/releases/download/v8.3.0/yolov8n-int8.tflite"
# MODEL_URL="https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov8n-int8.tflite"

# Download
wget "$MODEL_URL" -O yolov8n_int8.tflite.tmp

if [ $? -eq 0 ]; then
  echo "✓ Download successful"
  
  # Verify file size
  FILE_SIZE=$(stat -f%z yolov8n_int8.tflite.tmp 2>/dev/null || stat -c%s yolov8n_int8.tflite.tmp 2>/dev/null)
  FILE_SIZE_MB=$(echo "scale=2; $FILE_SIZE / 1024 / 1024" | bc)
  echo "✓ File size: $FILE_SIZE_MB MB"
  
  # Verify magic bytes
  MAGIC=$(xxd -l 4 -p yolov8n_int8.tflite.tmp | head -c 8)
  echo "✓ Magic bytes: $MAGIC"
  
  if [ "$MAGIC" = "54464c33" ] || [ "$MAGIC" = "5446 4c33" ]; then
    echo "✓ Valid TFLite format (TFL3)"
    
    # Backup old file
    if [ -f "coco_yolov8n_int8.tflite" ]; then
      mv coco_yolov8n_int8.tflite coco_yolov8n_int8.tflite.bak
      echo "✓ Backed up old file"
    fi
    
    # Rename
    mv yolov8n_int8.tflite.tmp coco_yolov8n_int8.tflite
    echo "✓ Model renamed to coco_yolov8n_int8.tflite"
    echo
    echo "✅ SUCCESS! Model downloaded and verified"
    echo "   Ready to use in yolov8_detection_http_server.html"
  else
    echo "❌ Invalid TFLite format!"
    echo "   Magic bytes: $MAGIC (expected: 54464c33)"
    rm yolov8n_int8.tflite.tmp
    exit 1
  fi
else
  echo "❌ Download failed!"
  echo "   Check internet connection or try alternative URL"
  exit 1
fi

echo
echo "🎉 Model ready! You can now:"
echo "   1. Run server: python3 serve_model.py"
echo "   2. Open: http://localhost:8000/yolov8_detection_http_server.html"