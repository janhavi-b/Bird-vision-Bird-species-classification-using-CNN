from datasets import load_dataset

ds = load_dataset("yashikota/birds-525-species-image-classification")

missing_id = 381

for split in ["train", "validation", "test"]:
    labels = ds[split]["label"]
    count = labels.count(missing_id)
    print(split, "PARAKETT AUKLET images:", count)

print("\nOther validation/test label differences:")

train_ids = set(ds["train"]["label"])

for split in ["validation", "test"]:
    extra = set(ds[split]["label"]) - train_ids
    print(split, "labels missing from training:", extra)
