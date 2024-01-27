from time import time

import nltk
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

nltk.download("punkt")

dataset = load_dataset("ccdv/cnn_dailymail", version="3.0.0")
print(dataset)

sample = dataset["train"][1]
print(
    f"""
Article (excerpt of 500 characters, total length: {len(sample["article"])}): """
)
print(sample["article"][:500])
print(f'\nSummary (length: {len(sample["highlights"])}):')
print(sample["highlights"])

sample_text = dataset["train"][1]["article"][:2000]


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


summaries = {}
summaries["baseline"] = three_sentence_summary(sample_text)

pipe = pipeline("summarization", model="t5-small")
start = time()
pipe_out = pipe(sample_text)
end = time()
print("t5:", end - start)
summaries["t5"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))

pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")
start = time()
pipe_out = pipe(sample_text)
end = time()
print("pegasus:", end - start)
summaries["pegasus"] = pipe_out[0]["summary_text"].replace(" .<n>", ".\n")


print("GROUND TRUTH")
print(dataset["train"][1]["highlights"], end="\n\n")
for model_name in summaries:
    print(model_name.upper())
    print(summaries[model_name])
    print("")


bleu_metric = load_metric("sacrebleu")
rouge_metric = load_metric("rouge")

reference = dataset["train"][1]["highlights"]
records = []
rouge_names = [
    "rouge1",
    "rouge2",
    "rougeL",  # calculates scores per sentence and averages it
    "rougeLsum",  # calculates directly over the whole summary
]

for model_name in summaries:
    rouge_metric.add(prediction=summaries[model_name], reference=reference)
    score = rouge_metric.compute()
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    records.append(rouge_dict)

df = pd.DataFrame.from_records(records, index=summaries.keys())
print(df)


device = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def evaluate_summaries_pegasus(
    dataset,
    metric,
    model,
    tokenizer,
    batch_size=16,
    device=device,
    column_text="article",
    column_summary="highlights",
):
    article_batches = list(chunks(dataset[column_text], batch_size))
    target_batches = list(chunks(dataset[column_summary], batch_size))
    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)
    ):
        inputs = tokenizer(
            article_batch,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        summaries = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            length_penalty=0.8,
            num_beams=8,
            max_length=128,
        )
        decoded_summaries = [
            tokenizer.decode(
                s, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for s in summaries
        ]
        decoded_summaries = [d.replace("<n>", "") for d in decoded_summaries]
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
    score = metric.compute()
    return score


test_sampled = dataset["test"].shuffle(seed=42).select(range(100))


model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
score = evaluate_summaries_pegasus(
    test_sampled, rouge_metric, model, tokenizer, batch_size=8
)
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
print(pd.DataFrame(rouge_dict, index=["pegasus"]))
