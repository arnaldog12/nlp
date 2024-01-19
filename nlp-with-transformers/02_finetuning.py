import numpy as np
import tensorflow as tf
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_labels = 6
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=num_labels,
).to(device)

emotions = load_dataset("dair-ai/emotion")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return dict(accuracy=acc, f1=f1)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

batch_size = 64

# pytorch
training_args = TrainingArguments(
    output_dir=f"{model_ckpt}-finetuned-emotion",
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=len(emotions_encoded["train"]) // batch_size,
    push_to_hub=False,
    # log_level="error"
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer,
)
trainer.train()

preds_outputs = trainer.predict(emotions_encoded["validation"])
print(preds_outputs.metrics)
y_preds = np.argmax(preds_outputs.predictions, axis=1)


# tensorflow
tf_model = TFAutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
)

tokenizer_columns = tokenizer.model_input_names

tf_train_dataset = emotions_encoded["train"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols=["label"],
    shuffle=True,
    batch_size=batch_size,
)

tf_eval_dataset = emotions_encoded["validation"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols=["label"],
    shuffle=False,
    batch_size=batch_size,
)

tf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

tf_model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=1)
