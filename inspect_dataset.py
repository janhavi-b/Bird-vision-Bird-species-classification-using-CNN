from datasets import load_dataset

ds = load_dataset("yashikota/birds-525-species-image-classification")

label_names = ds["train"].features["label"].names

print("Total bird species:", len(label_names))
print("\nFirst 20 bird species:")

for i, name in enumerate(label_names[:20]):
    print(i, ":", name)

print("\nSample image size:", ds["train"][0]["image"].size)
