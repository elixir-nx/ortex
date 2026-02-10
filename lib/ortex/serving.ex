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
  def init(_inline_or_process, model, defn_options) when is_list(defn_options) do
    defn_options =
      Enum.map(defn_options, fn opts ->
        opts = if is_list(opts), do: opts, else: []
        Keyword.put_new(opts, :compiler, Nx.Defn.Evaluator)
      end)

    func = fn x -> Ortex.run(model, x) end
    {:ok, {func, defn_options}}
  end

  @impl true
  def handle_batch(batch, partition, {function, defn_options}) do
    opts = Enum.at(defn_options, partition) || []

    materialized =
      case batch do
        %Nx.Batch{} -> Nx.Defn.jit_apply(&Function.identity/1, [batch], opts)
        _ -> batch
      end

    out = function.(materialized)
    {:execute, fn -> {out, :server_info} end, {function, defn_options}}
  end
end
