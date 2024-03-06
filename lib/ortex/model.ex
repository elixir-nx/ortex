defmodule Ortex.Model do
  @moduledoc """
  A model for running Ortex inference with.

  Implements a human-readable representation of a model including the name, dimension, and
  type of each input and output

  ```
  #Ortex.Model<
  inputs: [{"x", "Int32", [nil, 100]}, {"y", "Float32", [nil, 100]}]
  outputs: [
    {"9", "Float32", [nil, 10]},
    {"onnx::Add_7", "Float32", [nil, 10]},
    {"onnx::Add_8", "Float32", [nil, 10]}
  ]>
  ```

  `nil` values represent dynamic dimensions
  """
  alias Ortex.Native

  @enforce_keys [:reference]
  defstruct [:reference]

  @doc false
  def load(path, eps \\ [:cpu], opt \\ 3) do
    case Ortex.Native.init(path, map_eps(eps), opt) do
      {:error, msg} ->
        raise msg

      model ->
        %Ortex.Model{reference: model}
    end
  end

  @doc false
  def run(%Ortex.Model{} = model, tensor) when not is_tuple(tensor) do
    run(model, {tensor})
  end

  @doc false
  def run(%Ortex.Model{reference: model}, tensors) do
    # Move tensors into Ortex backend and pass the reference to the Ortex NIF
    output =
      case Ortex.Native.run(
             model,
             tensors
             |> Tuple.to_list()
             |> Enum.map(fn x -> x |> Nx.backend_transfer(Ortex.Backend) end)
             |> Enum.map(fn %Nx.Tensor{data: %Ortex.Backend{ref: x}} -> x end)
           ) do
        {:error, msg} -> raise msg
        output -> output
      end

    # Pack the output into new Ortex.Backend tensor(s)
    output
    |> Enum.map(fn {ref, shape, dtype_atom, dtype_bits} ->
      %Nx.Tensor{
        data: %Ortex.Backend{ref: ref},
        shape: shape |> List.to_tuple(),
        type: {dtype_atom, dtype_bits},
        names: List.duplicate(nil, length(shape))
      }
    end)
    |> List.to_tuple()
  end

  defp map_eps(eps) do
    Enum.map(eps, fn ep -> map_ep(ep) end)
  end

  defp map_ep(:cpu) do
    Native.make_cpu_ep(%Ortex.CPUExecutionProvider{})
  end

  defp map_ep(:cuda) do
    Native.make_cuda_ep(%Ortex.CUDAExecutionProvider{})
  end

  defp map_ep(:tensorrt) do
    Native.make_tensorrt_ep(%Ortex.TensorRTExecutionProvider{})
  end

  defp map_ep(%Ortex.CPUExecutionProvider{} = ep) do
    Native.make_cpu_ep(ep)
  end

  defp map_ep(%Ortex.CUDAExecutionProvider{} = ep) do
    Native.make_cuda_ep(ep)
  end

  defp map_ep(%Ortex.TensorRTExecutionProvider{} = ep) do
    Native.make_tensorrt_ep(ep)
  end
end

defimpl Inspect, for: Ortex.Model do
  import Inspect.Algebra

  def inspect(%Ortex.Model{reference: model}, inspect_opts) do
    case Ortex.Native.show_session(model) do
      {:error, msg} ->
        raise msg

      {inputs, outputs} ->
        force_unfit(
          concat([
            color("#Ortex.Model<", :map, inspect_opts),
            line(),
            nest(concat(["  inputs: ", Inspect.List.inspect(inputs, inspect_opts)]), 2),
            line(),
            nest(concat(["  outputs: ", Inspect.List.inspect(outputs, inspect_opts)]), 2),
            color(">", :map, inspect_opts)
          ])
        )
    end
  end
end
