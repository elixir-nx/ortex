defmodule Ortex do
  @moduledoc """
  Documentation for `Ortex`.

  `Ortex` is an Elixir wrapper around [ONNX Runtime](https://onnxruntime.ai/) using
  [Rustler](https://hexdocs.pm/rustler) and [ORT](https://github.com/pykeio/ort).
  """

  @doc """
  Load an `Ortex.Model` from disk. Optionally pass the execution providers as a list
  of descending priority and graph optimization level 1-3. Any graph optimization level
  beyond the range of 1-3 will disable graph optimization.

  By default, `Ortex` only includes some of the supported execution providers of ONNX Runtime.
  To enable others, first ensure you have downloaded or compiled a version of
  `libonnxruntime` that includes them, then set the environment variable `ORT_LIB_LOCATION`
  to its location. Then add `config :ortex, Ortex.Native, features: [EXECUTION_PROVIDERS]` to your
  `config.exs` where `EXECUTION_PROVIDERS` is a list of strings of which execution providers
  to enable.

  The optional fourth argument configures this session only:

    * `:intra_op_num_threads` / `:inter_op_num_threads` — non-negative integers
      (0 lets ONNX Runtime choose). Inter-op threads apply to parallel execution.
    * `:execution_mode` — `:sequential` or `:parallel`.
    * `:intra_op_allow_spinning` / `:inter_op_allow_spinning` — booleans.

  Omitted options preserve ONNX Runtime defaults. For example,
  `Ortex.load(path, [:cpu], 3, intra_op_num_threads: 1, inter_op_num_threads: 1,
  execution_mode: :sequential, intra_op_allow_spinning: false,
  inter_op_allow_spinning: false)`. Settings cannot be changed after loading.
  This API requires rebuilding the native library along with the Elixir code.

  ## Examples

      iex> Ortex.load("./models/tinymodel.onnx")
      iex> Ortex.load("./models/tinymodel.onnx", [:cuda, :cpu])
      iex> Ortex.load("./models/tinymodel.onnx", [:cpu], 0)

  """
  defdelegate load(path, eps \\ [:cpu], opt \\ 3, session_options \\ []), to: Ortex.Model

  @doc """
  Run a forward pass through a model.

  This takes a model and tuple of `Nx.Tensors`,
  optionally transfers them to the `Ortex.Backend` if they aren't there already,
  and runs a forward pass through the model. This will return a tuple of `Ortex.Backend`
  tensors, it's up to the user to transfer these back to another backend if additional
  ops are required.

  If there is only one input you can optionally pass a bare tensor rather than a tuple.

  ## Examples

      iex> model = Ortex.load("./models/tinymodel.onnx")
      iex> {%Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}}} = Ortex.run(
      ...>    model, {
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32),
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:f32)
      ...>    })

  """
  defdelegate run(model, tensors), to: Ortex.Model
end
