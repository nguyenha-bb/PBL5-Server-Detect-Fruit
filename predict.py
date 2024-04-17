import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.astype(np.float32)
    return img

def predict_and_display(image_path):
    img = load_and_preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]['index'], img.reshape(input_details[0]['shape']))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    label_ = np.where(output_data > 0.5, 1, 0)[0][0]
    confidence = output_data[0][0] if label_ == 1 else 1.0 - output_data[0][0]
    label = "Fresh Orange" if label_ == 0 else "Rotten Orange"

    img = cv2.imread(image_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text = f"{label} ({round(float(confidence) * 100, 2)}%)"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = 10
    text_y = 30 
    text_color = (0, 255, 0) if label == "Fresh Orange" else (0, 0, 255)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    cv2.imshow('Prediction', img)
    
    return {'image_path': image_path, 'state': label_}
