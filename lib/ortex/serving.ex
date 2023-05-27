defmodule Ortex.Serving do
  @moduledoc """
  `Ortex.Serving` Documentation

  This is a lightweight wrapper for using `Nx.Serving` behaviour with `Ortex`. Using `jit` and
  `defn` functions in this are not supported, it is strictly for serving batches to
  an `Ortex.Model` for inference.

  ## Examples

  ### Inline/serverless workflow 

  To quickly create an `Ortex.Serving` and run it

  ```elixir
  iex> model = Ortex.load("./models/resnet50.onnx")
  iex> serving = Nx.Serving.new(Ortex.Serving, model)
  iex> batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
  iex> {result} = Nx.Serving.run(serving, batch)
  iex> result |> Nx.backend_transfer |> Nx.argmax(axis: 1)
  #Nx.Tensor<
    s64[1]
    [499]
  >
  ```

  ### Stateful/process workflow

  An `Ortex.Serving` can also be started in your Application's supervision tree
  ```elixir
  model = Ortex.load("./models/resnet50.onnx")
  children = [
      {Nx.Serving,
       serving: Nx.Serving.new(Ortex.Serving, model),
       name: MyServing,
       batch_size: 10,
       batch_timeout: 100}
    ]
  opts = [strategy: :one_for_one, name: OrtexServing.Supervisor]
  Supervisor.start_link(children, opts)
  ```

  With the application started, batches can now be sent to the `Ortex.Serving` process

  ```elixir
  iex> Nx.Serving.batched_run(MyServing, Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}]))
  ...> {#Nx.Tensor<
  f32[1][1000]
  Ortex.Backend
   [
     [...]
   ]
  >}

  ```

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
