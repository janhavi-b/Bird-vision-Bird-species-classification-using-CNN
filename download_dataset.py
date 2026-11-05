from datasets import load_dataset

print("Downloading Bird Species Dataset...")

ds = load_dataset("yashikota/birds-525-species-image-classification")

print("\nDataset downloaded successfully!")
print(ds)
print("\nTraining samples:", len(ds["train"]))
print("Validation samples:", len(ds["validation"]))
print("Test samples:", len(ds["test"]))
print("\nColumns:", ds["train"].column_names)
