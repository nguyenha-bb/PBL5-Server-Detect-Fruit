import pyrebase
import base64
from datetime import datetime

config = {
    "apiKey": "AIzaSyCud_uqSMsLqkKkEdTlDJf1UZw32mTkfN0",
    "authDomain": "raspfruit2024.firebaseapp.com",
    "databaseURL": "https://raspfruit2024-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "raspfruit2024",
    "storageBucket": "raspfruit2024.appspot.com",
    "messagingSenderId": "890319740534",
    "appId": "1:890319740534:web:0bdb3e0a63ed4195188957",
    "measurementId": "G-EWV1GGTZ4M"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()


def upload_image(input_data):
    images = input_data["list_images"]
    list_images = []
    for image in images:
        with open(image["image_path"], "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            list_images.append({
                "image_path": img_base64,
                "state": int(image["state"]),
            })
    data = {"list_images": list_images, "result": int(input_data["result"]), "time_predict": input_data["time_predict"]}
    db.child("images_info").push(data)
