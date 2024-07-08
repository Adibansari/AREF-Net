from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import joblib
import io
import pathlib

app = FastAPI()

# Load the model
model_path = pathlib.Path(__file__).parent / "models" / "arefnet_model.pkl"
model = joblib.load(model_path)

# CIFAR-10 class names
class_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((32, 32))
        image = np.array(image)

        if image.shape != (32, 32, 3):
            raise HTTPException(status_code=400, detail="Image shape must be (32, 32, 3)")

        # Normalize the image
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        return JSONResponse(content={
            "predicted_class_index": int(predicted_class_index),
            "predicted_class_name": predicted_class_name
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AREF-Net image classification API. Use /predict to classify images."}
