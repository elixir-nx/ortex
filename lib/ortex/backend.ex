defmodule Ortex.Backend do
  @moduledoc """
  Documentation for `Ortex.Backend`.

  This implements the `Nx.Backend` behaviour for `Ortex` tensors. Most `Nx` operations
  are not implemented for this (although they may be in the future). This is mainly
  for ergonomic tensor construction and deconstruction from Ortex inputs and outputs.

  Since this does not implement most `Nx` operations, it's best *NOT* to set this as
  the default backend.
  """

  @behaviour Nx.Backend
  @enforce_keys [:ref]

  @derive {Nx.Container, containers: [:ref]}

  defstruct [:ref]

  alias Ortex.Backend, as: B
  alias Nx.Tensor, as: T

  @impl true
  def init(opts) do
    if opts != [] do
      raise ArgumentError, "Ortex.Backend accepts no options"
    end

    opts
  end

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, _backend_options) do
    data =
      case Ortex.Native.from_binary(binary, shape, type) do
        {:error, msg} -> raise msg
        res -> res
      end

    put_in(tensor.data, %Ortex.Backend{ref: data})
  end

  @impl true
  def to_binary(%T{data: %B{ref: ref}, type: {_, size}}, limit) do
    case Ortex.Native.to_binary(ref, size, limit) do
      {:error, msg} -> raise msg
      res -> res
    end
  end

  @impl true
  def backend_transfer(tensor, Nx.Tensor, _opts) do
    tensor
  end

  @impl true
  def backend_transfer(tensor, Ortex.Backend, _opts) do
    tensor
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    backend.from_binary(tensor, to_binary(tensor), opts)
  end

  defp to_binary(%T{data: %{ref: tensor}}) do
    # filling the bits and limits with 0 since we aren't using them right now
    Ortex.Native.to_binary(tensor, 0, 0)
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1
    # IO.inspect(limit)

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  @impl true
  def slice(out, %T{data: %B{ref: tensor_ref}}, start_indicies, lengths, strides) do
    r = Ortex.Native.slice(tensor_ref, start_indicies, lengths, strides)
    put_in(out.data, %Ortex.Backend{ref: r})
  end

  @impl true
  def reshape(out, _tensor) do
    out
  end

  if Application.compile_env(:ortex, :add_backend_on_inspect, true) do
    defp maybe_add_signature(result, %T{data: %B{ref: _mat_ref}}) do
      Inspect.Algebra.concat([
        "Ortex.Backend",
        Inspect.Algebra.line(),
        result
      ])
    end
  else
    defp maybe_add_signature(result, _tensor) do
      result
    end
  end

  funs = Nx.Backend.behaviour_info(:callbacks) -- Module.definitions_in(__MODULE__, :def)

  @doc false
  def __unimplemented__, do: unquote(funs)

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(args)) do
      raise "operation #{unquote(fun)} is not yet supported on Ortex.Backend."
    end
  end
end
