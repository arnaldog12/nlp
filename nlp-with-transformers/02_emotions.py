from datasets import load_dataset

# from datasets import list_datasets
# all_datasets = list_datasets()
# print(f"Number of datasets: {len(all_datasets)}")
# print(f"First 10 datasets: {all_datasets[:10]}")

emotions = load_dataset("dair-ai/emotion")
print(emotions)

train_ds = emotions["train"]
print(len(train_ds))
print(train_ds[0])
print(train_ds.column_names)
print(train_ds.features)
print(train_ds[:3])
print(train_ds["text"][:3])

emotions.set_format("pandas")
df = emotions["train"][:]


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)
df.head()
print(df.head())
print(df["label_name"].value_counts(ascending=True))

emotions.reset_format()
