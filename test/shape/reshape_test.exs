defmodule Ortex.TestReshape do
  use ExUnit.Case

  test "1d reshape" do
    t = Nx.tensor([1, 2, 3, 4])
    bin = t |> Nx.reshape({2, 2})

    ort = t |> Nx.backend_copy(Ortex.Backend) |> Nx.reshape({2, 2}) |> Nx.backend_transfer()

    assert bin == ort
  end

  test "2d reshape" do
    shape = Nx.tensor([[0], [0], [0], [0]])
    t = Nx.tensor([1, 2, 3, 4])
    bin = t |> Nx.reshape(shape)

    ort =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape(shape |> Nx.backend_copy(Ortex.Backend))
      |> Nx.backend_transfer()

    assert bin == ort
  end

  test "scalar reshape" do
    shape = {1, 1, 1}
    t = Nx.tensor(1)
    bin = t |> Nx.reshape(shape)

    ort =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape(shape)
      |> Nx.backend_transfer()

    assert bin == ort
  end

  test "auto reshape" do
    shape = {:auto, 2}
    t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
    bin = t |> Nx.reshape(shape)

    ort =
      t
      |> Nx.backend_copy(Ortex.Backend)
      |> Nx.reshape(shape)
      |> Nx.backend_transfer()

    assert bin == ort
  end
end
