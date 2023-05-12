import torch


class MultiInputModel(torch.nn.Module):
    """
    A simple model for testing Ortex multi-input and multi-output
    with different dtypes
    """

    def __init__(self):
        super(MultiInputModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 10)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        x = self.linear1(x.float())
        y = self.linear2(y)
        return x + y, x, y


tinymodel = MultiInputModel()
print(tinymodel)

x = torch.zeros([100], dtype=torch.int32).unsqueeze(0)
y = torch.zeros([100], dtype=torch.float32).unsqueeze(0)

tinymodel(x, y)

torch.onnx.export(
    tinymodel,
    (x, y),
    "tinymodel.onnx",
    input_names=["x", "y"],
    output_names=["output1", "output2", "output3"],
    dynamic_axes={
        "x": {0: "batch_size"},
        "y": {0: "batch_size"},
    },
)
