import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

subjqa = load_dataset("subjqa", name="electronics")
print(subjqa)
print(subjqa["train"]["answers"][1])

dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['id'].nunique()}")


model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

question = "How much music can this hold?"
context = "An MP3 is about 1 MB/minute, so about 6000 hours depending on file size."
inputs = tokenizer(question, context, return_tensors="pt")
print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))

model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(f"Input IDs shape: {inputs.input_ids.size()}")
print(f"Start logits shape: {start_logits.size()}")
print(f"End logits shape: {end_logits.size()}")


start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits)
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print("Question:", question)
print("Answer:", answer)


pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
print(pipe(question=question, context=context, topk=3))
