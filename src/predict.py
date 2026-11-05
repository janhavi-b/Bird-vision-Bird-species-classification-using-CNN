import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/birdvision_mobilenetv2.keras"
LABEL_PATH = "class_names.json"

model = None
class_names = None

def load_bird_model():
    global model, class_names

    if model is None:
        print("Loading BirdVision model...")
        model = tf.keras.models.load_model(MODEL_PATH)

    if class_names is None:
        with open(LABEL_PATH, "r") as f:
            class_names = json.load(f)

    return model, class_names


def predict_bird(image):
    model, class_names = load_bird_model()

    # Prepare uploaded image
    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)

    # Same preprocessing used during training
    image_array = (image_array / 127.5) - 1.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict bird species
    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-5:][::-1]

    top_predictions = [
        {
            "species": class_names[int(index)],
            "confidence": float(predictions[index] * 100)
        }
        for index in top_indices
    ]

    best_prediction = top_predictions[0]

    return (
        best_prediction["species"],
        best_prediction["confidence"],
        top_predictions
    )
