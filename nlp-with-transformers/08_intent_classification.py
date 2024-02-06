from datasets import load_dataset
from transformers import pipeline

bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)

query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van"""
print(pipe(query))


clinc = load_dataset("clinc_oos", "plus")
sample = clinc["test"][42]
print(sample)

intents = clinc["test"].features["intent"]
print(intents.int2str(sample["intent"]))

module = __import__("08_performance_benchmark")
pb = module.PerformanceBenchmark(pipe, clinc["test"])
perf_metrics = pb.run_benchmark(intents)
