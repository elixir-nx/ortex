defmodule Ortex.Native do
  @moduledoc """
  Documentation for `Ortex.Native`.

  Stubs for `Rustler` NIFs. These should never be called directly.
  """
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
end
