import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)

model.eval()
onnx_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    onnx_input,
    "resnet50.onnx",
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    export_params=True,
    opset_version=19,
)
