defmodule Ortex.TestSlice do
  use ExUnit.Case

  {tensor1d, _} = Nx.Random.uniform(Nx.Random.key(42), 0, 256, shape: {10})
  {tensor2d, _} = Nx.Random.uniform(Nx.Random.key(42), 0, 256, shape: {10, 10})

  @tensor1d tensor1d
  @tensor2d tensor2d

  defp tensor_binary(tensor, dtype) do
    tensor |> Nx.as_type(dtype)
  end

  defp tensor_ortex(tensor, dtype) do
    tensor
    |> Nx.as_type(dtype)
    |> Nx.backend_transfer(Ortex.Backend)
  end

  test "1d slice f32" do
    bin = tensor_binary(@tensor1d, :f32) |> Nx.slice([0], [4])

    ort = tensor_ortex(@tensor1d, :f32) |> Nx.slice([0], [4]) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d slice f32" do
    bin = tensor_binary(@tensor2d, :f32) |> Nx.slice([0, 2], [4, 6])

    ort =
      tensor_ortex(@tensor2d, :f32)
      |> Nx.slice([0, 2], [4, 6])
      |> Nx.backend_transfer()

    assert bin == ort
  end

  test "1d slice u8" do
    bin = tensor_binary(@tensor1d, :u8) |> Nx.slice([0], [4])

    ort = tensor_ortex(@tensor1d, :u8) |> Nx.slice([0], [4]) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d slice u8" do
    bin = tensor_binary(@tensor2d, :u8) |> Nx.slice([0, 2], [4, 6])

    ort =
      tensor_ortex(@tensor2d, :u8)
      |> Nx.slice([0, 2], [4, 6])
      |> Nx.backend_transfer()

    assert bin == ort
  end

  test "1d slice f32 strided" do
    bin = tensor_binary(@tensor1d, :f32) |> Nx.slice([0], [4], strides: [2])

    ort =
      tensor_ortex(@tensor1d, :f32) |> Nx.slice([0], [4], strides: [2]) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d slice f32 strided" do
    bin = tensor_binary(@tensor2d, :f32) |> Nx.slice([0, 2], [4, 6], strides: [2, 1])

    ort =
      tensor_ortex(@tensor2d, :f32)
      |> Nx.slice([0, 2], [4, 6], strides: [2, 1])
      |> Nx.backend_transfer()

    assert bin == ort
  end

  test "1d slice u8 strided" do
    bin = tensor_binary(@tensor1d, :u8) |> Nx.slice([0], [4], strides: [2])

    ort =
      tensor_ortex(@tensor1d, :u8) |> Nx.slice([0], [4], strides: [2]) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d slice u8 strided" do
    bin = tensor_binary(@tensor2d, :u8) |> Nx.slice([0, 2], [4, 6], strides: [2, 1])

    ort =
      tensor_ortex(@tensor2d, :u8)
      |> Nx.slice([0, 2], [4, 6], strides: [2, 1])
      |> Nx.backend_transfer()

    assert bin == ort
  end
end
