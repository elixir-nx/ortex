defmodule Ortex.TestDtypes do
  use ExUnit.Case

  {tensor, _} = Nx.Random.uniform(Nx.Random.key(42), 0, 256, shape: {100, 100})
  @tensor tensor

  defp bin_binary(dtype) do
    %{data: %{state: bin}} = @tensor |> Nx.as_type(dtype)
    bin
  end

  defp bin_ortex(dtype) do
    %{data: %{state: bin}} =
      @tensor
      |> Nx.as_type(dtype)
      |> Nx.backend_transfer(Ortex.Backend)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    bin
  end

  test "size 0 tensor" do
    %{data: %{state: bin1}} = Nx.tensor(0)

    %{data: %{state: bin2}} =
      Nx.tensor(0)
      |> Nx.backend_transfer(Ortex.Backend)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    assert bin1 == bin2
  end

  test "u8 conversion" do
    assert bin_binary(:u8) == bin_ortex(:u8)
  end

  test "u16 conversion" do
    assert bin_binary(:u16) == bin_ortex(:u16)
  end

  test "u32 conversion" do
    assert bin_binary(:u32) == bin_ortex(:u32)
  end

  test "u64 conversion" do
    assert bin_binary(:u64) == bin_ortex(:u64)
  end

  test "s8 conversion" do
    assert bin_binary(:s8) == bin_ortex(:s8)
  end

  test "s16 conversion" do
    assert bin_binary(:s16) == bin_ortex(:s16)
  end

  test "s32 conversion" do
    assert bin_binary(:s32) == bin_ortex(:s32)
  end

  test "s64 conversion" do
    assert bin_binary(:s64) == bin_ortex(:s64)
  end

  test "f16 conversion" do
    assert bin_binary(:f16) == bin_ortex(:f16)
  end

  test "bf16 conversion" do
    assert bin_binary(:bf16) == bin_ortex(:bf16)
  end

  test "f32 conversion" do
    assert bin_binary(:f32) == bin_ortex(:f32)
  end

  test "f64 conversion" do
    assert bin_binary(:f64) == bin_ortex(:f64)
  end
end
