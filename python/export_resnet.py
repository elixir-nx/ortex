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
)

"""
onnx.load("./resnet50.onnx")
...
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: -1
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 224
          }
          dim {
            dim_value: 224
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: -1
          }
          dim {
            dim_value: 1000
          }
        }
      }
    }
  }
}
"""
