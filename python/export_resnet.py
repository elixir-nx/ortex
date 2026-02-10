from pathlib import Path
import importlib

import torch
from torchvision.models import ResNet50_Weights, resnet50


def require_module(module_name: str, pip_name: str | None = None) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        install_name = pip_name or module_name
        raise SystemExit(
            f"Missing dependency: {module_name}. "
            f"Install it with `python3 -m pip install {install_name}`."
        )


require_module("onnx")

USE_DYNAMO = True
try:
    importlib.import_module("onnxscript")
except ModuleNotFoundError:
    USE_DYNAMO = False

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

onnx_input = torch.randn(1, 3, 224, 224)
repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
output_path = models_dir / "resnet50.onnx"

if not USE_DYNAMO:
    print("onnxscript not available; exporting with legacy ONNX exporter.")

with torch.inference_mode():
    torch.onnx.export(
        model,
        (onnx_input,),
        output_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True,
        opset_version=19,
        dynamo=USE_DYNAMO,
    )

print(f"Wrote {output_path}")
