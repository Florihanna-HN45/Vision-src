import cv2
import numpy as np
import os
from glob import glob

def preprocess_and_augment(image_path, output_folder):
    # 1. Load ảnh
    img = cv2.imread(image_path)
    if img is None:
        return

    # 2. Resizing: Đưa về kích thước chuẩn (ví dụ 224x224)
    img_resized = cv2.resize(img, (224, 224))

    # 3. Grayscale Conversion: Chuyển sang ảnh xám
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 4. Noise Reduction: Làm mịn bằng Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. Contrast Enhancement: Cân bằng biểu đồ nhiệt (Histogram Equalization)
    equalized = cv2.equalizeHist(blurred)

    # 6. Normalization: Chuẩn hóa pixel về đoạn [0, 1]
    # Lưu ý: Khi lưu ảnh, ta thường giữ lại định dạng 0-255. 
    # Bước này thường dùng ngay trước khi đưa vào Model.
    normalized = equalized.astype('float32') / 255.0

    # 7. Image Segmentation/Binarization: Chuyển về ảnh đen trắng (Otsu's threshold)
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. Geometric Transformations: Xoay ảnh 90 độ (Augmentation)
    rotated = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

    # --- Lưu kết quả ---
    base_name = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(f"{output_folder}/{base_name}_processed.jpg", equalized)
    cv2.imwrite(f"{output_folder}/{base_name}_aug_rotated.jpg", rotated)
    cv2.imwrite(f"{output_folder}/{base_name}_segmented.jpg", thresh)

# Cấu hình đường dẫn
input_dir = 'path/to/your/images'
output_dir = 'path/to/output' #vị trí lưu ảnh sau xử lý

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Chạy vòng lặp qua các file ảnh
image_files = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))

for img_path in image_files:
    preprocess_and_augment(img_path, output_dir)

print(f"Hoàn thành! Đã xử lý {len(image_files)} ảnh.")