# Ortex

`Ortex` is a wrapper around [ONNX Runtime](https://onnxruntime.ai/) (implemented as
bindings to [`ort`](https://github.com/pykeio/ort)). Ortex leverages
[`Nx.Serving`](https://hexdocs.pm/nx/Nx.Serving.html) to easily deploy ONNX models
that run concurrently and distributed in a cluster. Ortex also provides a storage-only
tensor implementation for ease of use.

ONNX models are a standard machine learning model format that can be exported from most ML
libraries like PyTorch and TensorFlow. Ortex allows for easy loading and fast inference of
ONNX models using different backends available to ONNX Runtime such as CUDA, TensorRT, Core
ML, and ARM Compute Library.

## Examples

TL;DR:

```elixir
iex> model = Ortex.load("./models/resnet50.onnx")
#Ortex.Model<
  inputs: [{"input", "Float32", [nil, 3, 224, 224]}]
  outputs: [{"output", "Float32", [nil, 1000]}]>
iex> {output} = Ortex.run(model, Nx.broadcast(0.0, {1, 3, 224, 224}))
iex> output |> Nx.backend_transfer() |> Nx.argmax
#Nx.Tensor<
  s64
  499
>
```

Inspecting a model shows the expected inputs, outputs, data types, and shapes. Axes with
`nil` represent a dynamic size.

To see more real world examples see the `examples` folder.

### Serving

`Ortex` also implements `Nx.Serving` behaviour. To use it in your application's
supervision tree consult the `Nx.Serving` docs.

```elixir
iex> serving = Nx.Serving.new(Ortex.Serving, model)
iex> batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
iex> {result} = Nx.Serving.run(serving, batch)
iex> result |> Nx.backend_transfer() |> Nx.argmax(axis: 1)
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
    {:ortex, "~> 0.1.10"}
  ]
end
```

You will need [Rust](https://www.rust-lang.org/tools/install) for compilation to succeed.

### Per-session CPU threading

`Ortex.load/4` accepts a keyword list of ONNX Runtime session options:

```elixir
model = Ortex.load("model.onnx", [:cpu], 3,
  intra_op_num_threads: 1,
  inter_op_num_threads: 1,
  execution_mode: :sequential,
  intra_op_allow_spinning: false,
  inter_op_allow_spinning: false
)
```

Omitting the fourth argument, or passing `[]`, retains the existing behavior.
Omitted individual settings leave ONNX Runtime defaults intact; `0` for either
thread count lets ONNX Runtime choose. Inter-op parallelism applies only to
`:parallel` execution. Settings belong to the loaded session, not the process or
node. Choose them using measurements for your model and hardware; a smaller
thread pool does not guarantee faster inference.

These settings use the existing `ort` 2.0.0-rc.8 APIs and session config entries;
no ONNX Runtime or Rust dependency upgrade is required. Build the Elixir code
**and** native library from the same revision (`mix deps.compile ortex --force`
for a dependency). A copied/precompiled old NIF cannot implement the new
`init_with_options/4` entry point. Ortex normally compiles Rust from source;
custom release pipelines must invalidate native caches, retain `Cargo.lock`,
build for each target OS/architecture, and package the matching ONNX Runtime
libraries as before. The Hex package includes the new Elixir source through its
existing `lib` entry.

`Ortex.run/2` still uses a `DirtyIo` NIF. Tensor transfers use other NIFs, including
`DirtyCpu` operations. Timing the Elixir `run/2` call includes transfer and
scheduler wait, not just ONNX graph execution. Scheduler classification deserves
a separate investigation and is deliberately unchanged here.

References: [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[ERTS dirty NIFs](https://www.erlang.org/doc/apps/erts/erl_nif.html#dirty-nifs).
