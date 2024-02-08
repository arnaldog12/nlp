import keyword

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

python_code = r"""def say_hello():
        print("Hello, World!")
# Print it
say_hello()
"""

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer(python_code).tokens())
print("normalizer:", tokenizer.backend_tokenizer.normalizer)
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))

a, e = "a", "€"
byte = ord(a.encode("utf-8"))
print(f'`{a}` is encoded as `{a.encode("utf-8")}` with a single byte: {byte}')
byte = [ord(chr(i)) for i in e.encode("utf-8")]
print(f'`{e}` is encoded as `{e.encode("utf-8")}` with three bytes: {byte}')

byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())
print(f"Size of our base vocabulary: {len(base_vocab)}")
print(f"First element: `{base_vocab[0]}`, last element: `{base_vocab[-1]}`")

# longest words in tokenizer (can show if they make sense or not to be in vocab)
tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
print([f"{tokenizer.convert_tokens_to_string([t])}" for t, _ in tokens[:8]])

# last words added to vocab
# can be and indication that target vocab is too large or contains unnecessary tokens
tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True)
print([f"{tokenizer.convert_tokens_to_string([t])}" for t, _ in tokens[:8]])

# train a tokenizer in a corpus
length = 200000
dataset_name = "transformersbook/codeparrot-train"
dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)


def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)["content"] for _ in range(batch_size)]


new_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=32768,  # multiples of 8 works better on GPUs
    initial_alphabet=base_vocab,
)
tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)
print([f"{new_tokenizer.convert_tokens_to_string([t])}" for t, _ in tokens[-12:]])
print(new_tokenizer(python_code).tokens())


print(f"There are in total {len(keyword.kwlist)} Python keywords.")
for keyw in keyword.kwlist:
    if keyw not in new_tokenizer.vocab:
        print(f"No, keyword `{keyw}` is not in the vocabulary")
