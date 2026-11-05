import os
import json
import random
import numpy as np
import tensorflow as tf
from datasets import load_dataset

# Keep CPU usage manageable
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

print("Loading bird dataset...")
ds = load_dataset("yashikota/birds-525-species-image-classification")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)

num_classes = len(class_names)
print("Number of bird species:", num_classes)

# Select a balanced training subset
print("Preparing balanced training images...")
labels = ds["train"]["label"]
random.seed(42)

selected_indices = []

for old_id in sorted(int(k) for k in label_mapping.keys()):
    class_indices = [
        i for i, label in enumerate(labels)
        if label == old_id
    ]
    selected_indices.extend(
        random.sample(class_indices, min(10, len(class_indices)))
    )

random.shuffle(selected_indices)
train_data = ds["train"].select(selected_indices)

# Exclude validation labels that do not exist in training
valid_ids = set(int(k) for k in label_mapping.keys())
val_labels = ds["validation"]["label"]

valid_val_indices = [
    i for i, label in enumerate(val_labels)
    if label in valid_ids
]

val_data = ds["validation"].select(valid_val_indices)

print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# Convert images lazily to avoid loading all images into RAM
def make_dataset(data, batch_size=8):
    def generator():
        for item in data:
            image = item["image"].convert("RGB").resize((224, 224))
            image = np.asarray(image, dtype=np.float32)
            image = (image / 127.5) - 1.0
            label = label_mapping[str(item["label"])]
            yield image, np.int32(label)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    return dataset.batch(batch_size).prefetch(1)

train_ds = make_dataset(train_data)
val_ds = make_dataset(val_data)

# Build MobileNetV2 transfer learning model
print("Building MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    num_classes, activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
)

# Save model and training history
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

model.save("models/birdvision_mobilenetv2.keras")

with open("results/training_history.json", "w") as f:
    json.dump(
        {key: [float(v) for v in values]
         for key, values in history.history.items()},
        f,
        indent=4
    )

print("\nTraining completed!")
print("Model saved in models/birdvision_mobilenetv2.keras")
print("Training history saved in results/training_history.json")
