import torch
from datasets import load_dataset
from torch import nn
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

torch.backends.quantized.engine = "qnnpack"  # mac os and windows only

clinc = load_dataset("clinc_oos", "plus")
intents = clinc["test"].features["intent"]

# dynamic quantization
model_ckpt = "transformersbook/distilbert-base-uncased-distilled-clinc"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to("cpu")

model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

pipe = pipeline("text-classification", model=model_quantized, tokenizer=tokenizer)
optim_type = "Distilation + quantization"

module = __import__("08_performance_benchmark")
pb = module.PerformanceBenchmark(pipe, clinc["test"])
perf_metrics = pb.run_benchmark(intents)
