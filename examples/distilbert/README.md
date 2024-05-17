# DistilBert exported to ONNX with HuggingFace transformers

### Running

Run `python export.py` to create the ONNX model for distilbert/distilbert-base-uncased-finetuned-sst-2-english, then `mix run` the `distilbert_classification.exs` script.

### Labels

When exporting the model from huggingface transformers to ONNX, a `config.json` file is added to the chosen directory. This file has the id to label mappings and you can extract them directly to give a label to the input, as shwon in `distilbert_classification.exs`. 
