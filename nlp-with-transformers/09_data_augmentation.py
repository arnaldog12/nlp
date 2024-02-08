import nlpaug.augmenter.word as naw
from transformers import set_seed

set_seed(3)
aug = naw.ContextualWordEmbsAug(
    model_path="distilbert-base-uncased", device="cpu", action="substitute"
)

text = "Transformers are the most popular toys"
print("Original text:", text)
print("Augmented text:", aug.augment(text))
