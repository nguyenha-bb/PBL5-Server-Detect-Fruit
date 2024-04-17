import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time
from random import randint
from datetime import datetime
import storage
import predict


CAMERA_DEVICE_ID = 0 
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
fps = 0

def saveResult(list_images):
    result = 0
    for image_info in list_images:
        if image_info['state'] == 1:
            result = 1
            break
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    input_data = {
        "list_images": list_images,
        "result": result,
        "time_predict": current_time,
    }
        
    storage.upload_image(input_data)

def visualize_fps(image, fps: int):
    if len(np.shape(image)) < 3:
        text_color = (255, 255, 255)  
    else:
        text_color = (0, 255, 0) 
    row_size = 20 
    left_margin = 24 

    font_size = 1
    font_thickness = 1

    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    return image

if __name__ == '__main__':
    try:
        cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)
        capture_image = False

        with open('coco.names', 'rt') as f:
            class_names = f.read().rstrip('\n').split('\n')

        model_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weight_path = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weight_path, model_path)
        net.setInputSize(320, 320)
        net.setInputScale(1.0/127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        
        list_images = []
        count = 0

        while True:
            start_time = time.time()
            _, frame = cap.read()

            classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if class_names[classId-1].lower() == 'orange':
                        capture_image = True
            
            frame_with_fps = visualize_fps(frame, fps)
            cv2.imshow('Object Detection', frame_with_fps)
                        
            if capture_image:
                count += 1
                cv2.imwrite(f'orange_detected_{count}.jpg', frame_with_fps)
                image_info = predict.predict_and_display(f'orange_detected_{count}.jpg')
                list_images.append(image_info)
                capture_image = False  
                time.sleep(3)
            

            end_time = time.time()
            seconds = end_time - start_time
            fps = 1.0 / seconds
            
            if count == 2:
                saveResult(list_images)
                break

            if cv2.waitKey(1) == 27:
                break
            
    except Exception as e:
        print(e)

    finally:
        cv2.destroyAllWindows()
        cap.release()

