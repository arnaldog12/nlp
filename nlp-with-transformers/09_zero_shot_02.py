import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from transformers import pipeline

module = __import__("09_dataset")
ds = module.ds
mlb = module.mlb
all_labels = module.all_labels

np.random.seed(0)
all_indices = np.expand_dims(list(range(len(ds["train"]))), axis=1)
indices_pool = all_indices
labels = mlb.transform(ds["train"]["labels"])
train_samples = [8, 16, 32, 64, 128]
train_slices, last_k = [], 0
for i, k in enumerate(train_samples):
    indices_pool, labels, new_slice, _ = iterative_train_test_split(
        indices_pool, labels, (k - last_k) / len(labels)
    )
    last_k = k
    if i == 0:
        train_slices.append(new_slice)
    else:
        train_slices.append(np.concatenate((train_slices[-1], new_slice)))

train_slices.append(all_indices), train_samples.append(len(ds["train"]))
train_slices = [np.squeeze(train_slice) for train_slice in train_slices]


def prepare_labels(batch):
    batch["label_ids"] = mlb.transform(batch["labels"])
    return batch


ds = ds.map(prepare_labels, batched=True)
sample = ds["train"][0]
print("Labels:", f"{sample['labels']}")

pipe = pipeline("zero-shot-classification")
output = pipe(sample["text"], all_labels, multi_label=True)
print(output["sequence"][:400], end="\n\n")
print("Predictions")

for label, score in zip(output["labels"], output["scores"]):
    print(f"{label}: {score:.2f}")


def zero_shot_pipeline(example):
    output = pipe(example["text"], all_labels, multi_label=True)
    example["predicted_labels"] = output["labels"]
    example["scores"] = output["scores"]
    return example


ds_zero_shot = ds["valid"].map(zero_shot_pipeline)
print(ds_zero_shot)
