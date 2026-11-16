defmodule Ortex.TestStack do
  use ExUnit.Case

  defp to_ortex(tensor), do: Nx.backend_transfer(tensor, Ortex.Backend)

  test "stack 1d tensors" do
    t1 = Nx.tensor([1, 2, 3])
    t2 = Nx.tensor([4, 5, 6])

    stacked = Nx.stack([to_ortex(t1), to_ortex(t2)]) |> Nx.backend_transfer()
    assert stacked == Nx.stack([t1, t2])
  end

  test "stack 2d tensors on each axis" do
    t1 = Nx.iota({2, 3}, type: :f32)
    t2 = Nx.iota({2, 3}, type: :f32) |> Nx.add(10)

    for axis <- 0..2 do
      stacked = Nx.stack([to_ortex(t1), to_ortex(t2)], axis: axis) |> Nx.backend_transfer()
      assert stacked == Nx.stack([t1, t2], axis: axis)
    end
  end

  test "stack three tensors" do
    tensors = for i <- 1..3, do: Nx.iota({2, 2}, type: :s64) |> Nx.multiply(i)

    stacked = tensors |> Enum.map(&to_ortex/1) |> Nx.stack(axis: 1) |> Nx.backend_transfer()
    assert stacked == Nx.stack(tensors, axis: 1)
  end
end
