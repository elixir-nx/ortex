# Ortex

`Ortex` is a wrapper around [ONNX Runtime](https://onnxruntime.ai/) implemented as a
(limited) `Nx.Backend` using `Rustler` and [`ort`](https://github.com/pykeio/ort).

ONNX models are a standard machine learning model format that can be exported from most ML
libraries like PyTorch and TensorFlow. Ortex allows for easy loading and fast inference of
ONNX models using different backends available to ONNX Runtime such as CUDA, TensorRT, Core
ML, and ARM Compute Library.

## Examples

TL;DR
```elixir
iex> model = Ortex.load("./models/resnet50.onnx")
#Ortex.Model<
  inputs: [{"input", "Float32", [nil, 3, 224, 224]}]
  outputs: [{"output", "Float32", [nil, 1000]}]>
iex> {output} = Ortex.run(model, Nx.broadcast(0.0, {1, 3, 224, 224}))
iex> output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.argmax
#Nx.Tensor<
  s64
  499
>
```
Inspecting a model shows the expected inputs, outputs, data types, and shapes. Axes with
`nil` represent a dynamic size.

To see more real world examples see `examples`.

### Serving
`Ortex` also implements `Nx.Serving` behaviour. To use it in your application's
supervision tree consult the `Nx.Serving` docs.

```elixir
iex> serving = Nx.Serving.new(Ortex.Serving, model)
iex> batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
iex> {result} = Nx.Serving.run(serving, batch)
iex> result |> Nx.backend_transfer |> Nx.argmax(axis: 1)
#Nx.Tensor<
  s64[1]
  [499]
>
```

## Installation

`Ortex` can be installed by adding `ortex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ortex, "~> 0.1.4"}
  ]
end
```
