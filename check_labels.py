from datasets import load_dataset

ds = load_dataset("yashikota/birds-525-species-image-classification")

labels = ds["train"].features["label"].names

print("Total labels:", len(labels))
print("\nLast 10 labels:")

for i in range(len(labels)-10, len(labels)):
    print(i, ":", labels[i])

print("\nUnique training labels:", len(set(ds["train"]["label"])))
