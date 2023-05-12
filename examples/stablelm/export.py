from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-3b")
print(model)

prompt = "<|ASSISTANT|>"

inputs = tokenizer(prompt, return_tensors="pt")
torch.onnx.export(
    model,
    (inputs["input_ids"].cpu(), inputs["attention_mask"].cpu()),
    "output/stability-lm-tuned-3b.onnx",
    input_names=["input_ids", "attention_mask"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    },
)
