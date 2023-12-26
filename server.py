from fastapi import FastAPI,File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response 
import io
from PIL import Image 
import tensorflow as tf
from uvicorn import run
import math as m
import numpy as np



app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=methods,
    allow_headers=headers,
)


class_names = ["Covid", "Normal"]
model = tf.keras.models.load_model("model.keras")


def get_prediction(image_np):
    image_np = tf.image.resize(image_np, [256, 256])
    image_np = np.expand_dims(image_np, axis=0)
    image_np = image_np/255
    prediction = model.predict(image_np)
    prediction = tf.nn.softmax(prediction[0])
    return {"class": class_names[np.argmax(prediction)], "score": m.floor(np.max(prediction)*100)}



@app.get("/")
def home():
    return {"message": "Hello World"}



@app.get("/predict")
def sendHtml():
    with open("index.html", "r") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file:
        return {"message": "No file"}
    img = Image.open(file.file).convert("RGB")
    image_np = np.array(img)
    result = get_prediction(image_np)
    return result



if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)