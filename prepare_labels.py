from datasets import load_dataset
import json

print("Preparing correct species labels...")

ds = load_dataset("yashikota/birds-525-species-image-classification")

label_names = ds["train"].features["label"].names
train_ids = sorted(set(ds["train"]["label"]))

# Keep only classes that have training images
label_mapping = {
    old: new for new, old in enumerate(train_ids)
}

class_names = [label_names[i] for i in train_ids]

# Find validation images without training examples
validation_labels = ds["validation"]["label"]
excluded = [
    i for i, label in enumerate(validation_labels)
    if label not in label_mapping
]

print("\nExcluded validation images:", len(excluded))
print("Excluded species:", label_names[381])

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f, indent=4)

print("\nLabel mapping created successfully!")
print("Training classes:", len(class_names))
print("Saved: class_names.json")
print("Saved: label_mapping.json")
