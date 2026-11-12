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

  Session config entries can be passed as a fourth argument. These are handed
  straight to ONNX Runtime as
  [session configuration keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h),
  so any key ONNX Runtime understands works here. Values may be given as strings,
  booleans (`true`/`false` become `"1"`/`"0"`), or numbers.

  ## Examples

      iex> Ortex.load("./models/tinymodel.onnx")
      iex> Ortex.load("./models/tinymodel.onnx", [:cuda, :cpu])
      iex> Ortex.load("./models/tinymodel.onnx", [:cpu], 0)
      iex> Ortex.load("./models/tinymodel.onnx", [:cpu], 3,
      ...>   "session.intra_op.allow_spinning": false,
      ...>   "session.inter_op.allow_spinning": false
      ...> )

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
