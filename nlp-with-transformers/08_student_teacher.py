import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)
id2label = pipe.model.config.id2label
label2id = pipe.model.config.label2id

student_ckpt = "distilbert-base-uncased"
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)

accuracy_score = load_metric("accuracy")

module = __import__("08_knowledge_distillation")


def tokenize_text(batch):
    return student_tokenizer(batch["text"], truncation=True)


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


clinc = load_dataset("clinc_oos", "plus")
intents = clinc["test"].features["intent"]
num_labels = intents.num_classes

clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=["text"])
clinc_enc = clinc_enc.rename_column("intent", "labels")

batch_size = 48

finetuned_ckpt = "distilbert-base-uncased-finetuned-clinc"
student_training_args = module.DistillationTrainingArguments(
    output_dir=finetuned_ckpt,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    alpha=0.125,
    weight_decay=0.01,
    push_to_hub=False,
)


student_config = AutoConfig.from_pretrained(
    student_ckpt,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)


def student_init():
    return AutoModelForSequenceClassification.from_pretrained(
        student_ckpt, config=student_config
    ).to(device)


teacher_model = AutoModelForSequenceClassification.from_pretrained(
    bert_ckpt, num_labels=num_labels
).to(device)

distilbert_trainer = module.DistillationTrainer(
    model_init=student_init,
    teacher_model=teacher_model,
    args=student_training_args,
    train_dataset=clinc_enc["train"],
    eval_dataset=clinc_enc["validation"],
    compute_metrics=compute_metrics,
    tokenizer=student_tokenizer,
)
distilbert_trainer.train()
