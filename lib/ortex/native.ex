defmodule Ortex.Native do
  @moduledoc false

  # We have to compile the crate before `use Rustler` compiles the crate since
  # cargo downloads the onnxruntime shared libraries and they are not available
  # to load or copy into Elixir's during the on_load or Elixir compile steps.
  # In the future, this may be configurable in Rustler.
  Rustler.Compiler.compile_crate(__MODULE__, otp_app: :ortex, crate: :ortex)
  Ortex.Util.copy_ort_libs()

  use Rustler,
    otp_app: :ortex,
    crate: :ortex

  # When loading a NIF module, dummy clauses for all NIF function are required.
  # NIF dummies usually just error out when called when the NIF is not loaded, as that should never normally happen.
  def init(_model_path, _execution_providers, _optimization_level),
    do: :erlang.nif_error(:nif_not_loaded)

  def run(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def from_binary(_bin, _shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def to_binary(_reference, _bits, _limit), do: :erlang.nif_error(:nif_not_loaded)
  def show_session(_model), do: :erlang.nif_error(:nif_not_loaded)

  def slice(_tensor, _start_indicies, _lengths, _strides),
    do: :erlang.nif_error(:nif_not_loaded)

  def reshape(_tensor, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def concatenate(_tensors_refs, _type, _axis), do: :erlang.nif_error(:nif_not_loaded)
end
