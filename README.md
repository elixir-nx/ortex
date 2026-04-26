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
    {:ortex, "~> 0.1.11"}
  ]
end
```

### Precompiled NIFs

Starting with `0.1.11`, Ortex ships **precompiled NIF artifacts** for common platforms. If you're on one of the supported targets you do **not** need a Rust toolchain — `mix deps.get` downloads a prebuilt binary directly.

Supported targets:

* `aarch64-apple-darwin` (Apple Silicon macOS)
* `x86_64-apple-darwin` (Intel macOS)
* `x86_64-unknown-linux-gnu` (Linux x86_64, glibc >= 2.35)
* `aarch64-unknown-linux-gnu` (Linux ARM64, glibc >= 2.35)
* `x86_64-pc-windows-msvc` (Windows x86_64, MSVC ABI)

`onnxruntime` is statically linked into the precompiled NIF, so the artifact is a single self-contained shared library — no separate runtime install required.

### Source build

You'll need [Rust](https://www.rust-lang.org/tools/install) and a working C toolchain to build from source. This happens automatically when:

* Your target tuple isn't in the precompiled list above (e.g. musl Linux, 32-bit ARM, FreeBSD).
* You explicitly request it with `ORTEX_BUILD=true`:

  ```bash
  ORTEX_BUILD=true mix deps.compile ortex --force
  ```

The source path is the same flow that's shipped since 0.1.0 — `cargo build` downloads `onnxruntime` and the resulting NIF is dropped into `priv/native/`. Useful for development on the NIF crate itself, or when you need a feature flag (`cuda`, `tensorrt`, `coreml`, `directml`) that the precompiled CPU-only artifact doesn't include.
