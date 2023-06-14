defmodule Ortex.TestSqueeze do
  use ExUnit.Case

  test "1d squeeze" do
    t = Nx.tensor([[[1, 2, 3, 4]]])
    bin = t |> Nx.squeeze()

    ort = t |> Nx.backend_copy(Ortex.Backend) |> Nx.squeeze() |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d squeeze" do
    t = Nx.tensor([[[[1, 2]], [[3, 4]]]])
    bin = t |> Nx.squeeze()

    ort = t |> Nx.backend_copy(Ortex.Backend) |> Nx.squeeze() |> Nx.backend_transfer()

    assert bin == ort
  end

  test "axis squeeze" do
    t = Nx.tensor([[[[1, 2]], [[3, 4]]]])
    bin = t |> Nx.squeeze(axes: [0])

    ort = t |> Nx.backend_copy(Ortex.Backend) |> Nx.squeeze(axes: [0]) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "named squeeze" do
    t = Nx.tensor([[[[1, 2]], [[3, 4]]]], names: [:w, :x, :y, :z])
    bin = t |> Nx.squeeze(axes: [:w])

    ort = t |> Nx.backend_copy(Ortex.Backend) |> Nx.squeeze(axes: [:w]) |> Nx.backend_transfer()

    assert bin == ort
  end
end
