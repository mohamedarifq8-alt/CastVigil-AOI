import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(
    title="Casting Quality Inspection API",
    description="Automated Optical Inspection API using MobileNetV3 Softmax model.",
    version="2.0.0"
)

MODEL_PATH = "casting_model.onnx"
CLASS_NAMES = ["Defective", "OK"] # Order derived during model training

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("✅ ONNX Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")

IMG_SIZE = 224

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32)
    # Add batch dimension: (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

@app.post("/predict", summary="Inspect Casting Product")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)
        
        # Run inference
        outputs = session.run(None, {input_name: processed_img})
        probabilities = outputs[0][0] # Returns [p_defective, p_ok]
        
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100
        
        return {
            "filename": file.filename,
            "status": predicted_label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "Defective": round(float(probabilities[0]) * 100, 2),
                "OK": round(float(probabilities[1]) * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
def root():
    return {"message": "Casting Inspection API is running successfully!"}