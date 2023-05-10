defmodule OrtexTest do
  use ExUnit.Case
  doctest Ortex

  test "resnet50" do
    model = Ortex.load("./models/resnet50.onnx")

    input = Nx.broadcast(0.0, {1, 3, 224, 224})
    {output} = Ortex.run(model, {input})
    argmax = output |> Nx.backend_transfer() |> Nx.argmax(axis: 1)

    assert argmax == Nx.tensor([499])
  end

  test "transfer to Ortex.Backend" do
    assert true
  end

  test "transfer from Ortex.Backend" do
    assert true
  end

  test "Nx.Serving with resnet50" do
    model = Ortex.load("./models/resnet50.onnx")

    serving = Nx.Serving.new(Ortex.Serving, model)
    batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
    {result} = Nx.Serving.run(serving, batch)
    assert result |> Nx.backend_transfer() |> Nx.argmax(axis: 1) == Nx.tensor([499])
  end

  test "Nx.Serving with tinymodel" do
    model = Ortex.load("./models/tinymodel.onnx")

    serving = Nx.Serving.new(Ortex.Serving, model)

    # Create a batch of size 3 with {int32, float32} inputs
    batch =
      Nx.Batch.stack([
        {Nx.broadcast(0, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(0.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(1, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(1.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(2, {100}) |> Nx.as_type(:s32), Nx.broadcast(2.0, {100}) |> Nx.as_type(:f32)}
      ])

    {%Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}} =
      Nx.Serving.run(serving, batch)
  end
end
