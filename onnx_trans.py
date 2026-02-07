import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Đường dẫn file input và output
input_model_path = r'D:/1. STUDY/ROBOCON 2026/Weapons1/21th.onnx'
output_model_path = r'D:/1. STUDY/ROBOCON 2026/Weapons1/best_int8.onnx'

# Thực hiện lượng tử hóa Dynamic
quantize_dynamic(
    model_input=input_model_path,
    model_output=output_model_path,
    weight_type=QuantType.QUInt8  # Chuyển trọng số về Unsigned Int 8
)

print(f"Đã tạo file {output_model_path}")