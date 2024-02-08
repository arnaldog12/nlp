import pandas as pd
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

module = __import__("09_dataset")
ds = module.ds


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,
    )


ds_mlm = ds.map(tokenize, batched=True)
ds_mlm = ds_mlm.remove_columns(["labels", "text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)


set_seed(3)
data_collator.return_tensors = "np"
inputs = tokenizer("Transformers are awesome!", return_tensors="np")
outputs = data_collator([{"input_ids": inputs["input_ids"][0]}])

original_input_ids = inputs["input_ids"][0]
masked_input_ids = outputs["input_ids"][0]

df_sample = pd.DataFrame(
    {
        "Original tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        "Masked tokens": tokenizer.convert_ids_to_tokens(outputs["input_ids"][0]),
        "Original input_ids": original_input_ids,
        "Masked input_ids": masked_input_ids,
        "Labels": outputs["labels"][0],
    }
).T
print(df_sample)

data_collator.return_tensors = "pt"

training_args = TrainingArguments(
    output_dir=f"{model_ckpt}-issues-128",
    per_device_train_batch_size=32,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=16,
    push_to_hub=False,
    log_level="error",
    report_to="none",
)

trainer = Trainer(
    model=AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds_mlm["unsup"],
    eval_dataset=ds_mlm["train"],
)
trainer.train()
