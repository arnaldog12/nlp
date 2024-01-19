from datasets import load_dataset
from transformers import AutoTokenizer

emotions = load_dataset("dair-ai/emotion")
print(emotions)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    # padding=True: pad the examples with zeros to the size of the longest one in a batch
    # truncation=True: truncate the examples to the model's maximum context size
    return tokenizer(batch["text"], padding=True, truncation=True)


print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(
    tokenize,
    batched=True,  # if False, map operates individually on each example
    batch_size=None,  # All examples at once. Ensures input_tensors and attetion_masks have the same shape globally
)
print(emotions_encoded)
