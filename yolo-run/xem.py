import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path=r"C:\Users\This pc\Downloads\pdisease1_int8 (2).tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print("kich thuoc:", input_details[0]['shape'])