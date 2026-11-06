import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

MODEL_PATH = "models/birdvision_mobilenetv2.keras"
LABEL_PATH = "class_names.json"

model = None
class_names = None
bird_checker = None

# Common bird categories in the ImageNet classifier
BIRD_LABELS = {
    "hen", "cock", "ostrich", "brambling", "goldfinch",
    "house_finch", "junco", "indigo_bunting", "robin",
    "bulbul", "jay", "magpie", "chickadee", "water_ouzel",
    "kite", "bald_eagle", "vulture", "great_grey_owl",
    "albatross", "pelican", "king_penguin", "swan",
    "black_swan", "goose", "drake", "partridge", "quail",
    "peacock", "macaw", "sulphur_crested_cockatoo",
    "lorikeet", "coucal", "bee_eater", "hornbill",
    "hummingbird", "toucan", "African_grey", "flamingo",
    "crane", "stork", "spoonbill", "heron", "egret",
    "bittern", "bustard", "oystercatcher", "sandpiper",
    "woodpecker", "kingfisher", "pigeon", "dove",
    "parrot", "parakeet", "finch", "sparrow"
}


def load_bird_model():
    global model, class_names, bird_checker

    if model is None:
        print("Loading BirdVision model...")
        model = tf.keras.models.load_model(MODEL_PATH)

    if class_names is None:
        with open(LABEL_PATH, "r") as f:
            class_names = json.load(f)

    if bird_checker is None:
        print("Loading bird verification model...")
        bird_checker = MobileNetV2(weights="imagenet")

    return model, class_names, bird_checker


def is_bird_image(image):
    _, _, checker = load_bird_model()

    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = checker.predict(image_array, verbose=0)
    results = decode_predictions(predictions, top=3)[0]

    # Accept only when the top ImageNet prediction is a bird
    top_label = results[0][1]

    return top_label in BIRD_LABELS


def predict_bird(image):
    model, class_names, _ = load_bird_model()

    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)

    image_array = (image_array / 127.5) - 1.0
    image_array = np.expand_dims(image_array, axis=0)

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
