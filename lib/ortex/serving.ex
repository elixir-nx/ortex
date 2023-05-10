defmodule Ortex.Serving do
  @moduledoc """
  `Ortex.Serving` Documentation

  This is a light wrapper for using `Nx.Serving` behaviour with `Ortex`. Using `jit` and
  `defn` functions in this are not supported, it is strictly for serving batches to
  an `Ortex.Model` for inference.
  """

  @behaviour Nx.Serving

  @impl true
  def init(_inline_or_process, model, [_defn_options]) do
    func = fn x -> Ortex.run(model, x) end
    {:ok, func}
  end

  @impl true
  def handle_batch(batch, _partition, function) do
    # A hack to move the back into a tensor for Ortex
    out = function.(Nx.Defn.jit_apply(&Function.identity/1, [batch]))
    {:execute, fn -> {out, :server_info} end, function}
  end
end
