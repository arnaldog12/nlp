import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from psutil import cpu_count
from scipy.special import softmax
from transformers import AutoTokenizer
from transformers.convert_graph_to_onnx import convert

os.environ["OMP_NUM_THREADS"] = str(cpu_count)
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"


def tokenize_text(batch):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(batch["text"], truncation=True)


model_ckpt = "transformersbook/distilbert-base-uncased-distilled-clinc"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
clinc = load_dataset("clinc_oos", "plus")
clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=["text"])
clinc_enc = clinc_enc.rename_column("intent", "labels")
intents = clinc["test"].features["intent"]

onnx_model_path = Path("onnx/model.onnx")
convert(
    framework="pt",
    model=model_ckpt,
    tokenizer=tokenizer,
    output=onnx_model_path,
    opset=12,  # onnx version
    pipeline_name="text-classification",
)


def create_model_for_provider(model_path, provider="CPUExecutionProvider"):
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session


onnx_model = create_model_for_provider(onnx_model_path)

inputs = clinc_enc["test"][:1]
del inputs["labels"]
logits_onnx = onnx_model.run(None, inputs)[0]
print(logits_onnx.shape)
print(np.argmax(logits_onnx))
print(clinc_enc["test"][0]["labels"])


class OnnxPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [{"label": intents.int2str(pred_idx), "score": probs[pred_idx]}]


query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van"""

pipe = OnnxPipeline(onnx_model, tokenizer)
pipe(query)


module = __import__("08_performance_benchmark")
pb = module.OnnxPerformanceBenchmark(
    "onnx/model.onnx",
    pipe,
    clinc["test"],
    "Distillation + ORT",
)
perf_metrics = pb.run_benchmark(intents)


model_input = "onnx/model.onnx"
model_output = "onnx/model.quant.onnx"
quantize_dynamic(model_input, model_output, weight_type=QuantType.QInt8)
onnx_quantized_model = create_model_for_provider(model_output)
pipe = OnnxPipeline(onnx_quantized_model, tokenizer)
pb = module.OnnxPerformanceBenchmark(
    model_output,
    pipe,
    clinc["test"],
    "Distillation + ORT (quantized)",
)
pb.run_benchmark(intents)
