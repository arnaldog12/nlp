import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

module = __import__("09_dataset")
ds = module.ds
all_labels = module.all_labels
train_slices = module.train_slices


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)


ds_enc = ds.map(tokenize, batched=True)
ds_enc = ds_enc.remove_columns(["labels", "text"])
ds_enc.set_format("torch")
ds_enc = ds_enc.map(
    lambda x: {"label_ids_f": x["label_ids"].to(torch.float)},
    remove_columns=["label_ids"],
)
ds_enc = ds_enc.rename_column("label_ids_f", "label_ids")

training_args_fine_tune = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    learning_rate=3e-5,
    lr_scheduler_type="constant",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    weight_decay=0.0,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro f1",
    save_total_limit=1,
    log_level="error",
)


def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = sigmoid(pred.predictions)
    y_pred = (y_pred > 0.5).astype(float)

    clf_dict = classification_report(
        y_true, y_pred, target_names=all_labels, zero_division=0, output_dict=True
    )
    return {
        "micro f1": clf_dict["micro avg"]["f1-score"],
        "macro f1": clf_dict["macro avg"]["f1-score"],
    }


config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = len(all_labels)
config.problem_type = "multi_label_classification"

for train_slice in train_slices:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, config=config
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=ds_enc["train"].select(train_slice),
        eval_dataset=ds_enc["valid"],
    )
    trainer.train()

    pred = trainer.predict(ds_enc["test"])
    metrics = compute_metrics(pred)
    print(metrics)
