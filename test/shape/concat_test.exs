defmodule Ortex.TestConcat do
  use ExUnit.Case

  # Testing each type, since there's a bunch of boilerplate that we want to 
  # check for errors on the Rust side
  %{
    "s8" => {:s, 8},
    "s16" => {:s, 16},
    "s32" => {:s, 16},
    "s64" => {:s, 16},
    "u8" => {:s, 16},
    "u16" => {:s, 16},
    "u32" => {:s, 16},
    "u64" => {:s, 16},
    "f16" => {:s, 16},
    "bf16" => {:s, 16},
    "f32" => {:s, 16},
    "f64" => {:s, 16}
  }
  |> Enum.each(fn {type_str, type_tuple} ->
    test "Concat 1d tensors #{type_str}" do
      t1 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      t2 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      concatted = Nx.concatenate([t1, t2]) |> Nx.backend_transfer()
      expected = Nx.tensor([1, 2, 3, 4, 1, 2, 3, 4], type: unquote(type_tuple))
      assert concatted == expected
    end

    test "Concat 3d tensors #{type_str}" do
      o1 = Nx.iota({2, 3, 5}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      o2 = Nx.iota({1, 3, 5}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      concatted = Nx.concatenate([o1, o2]) |> Nx.backend_transfer()
      expected = Nx.concatenate([o1 |> Nx.backend_transfer(), o2 |> Nx.backend_transfer()])
      assert concatted == expected
    end

    test "Concat 3 #{type_str} vectors" do
      t1 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      t2 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      t3 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      concatted = Nx.concatenate([t1, t2, t3]) |> Nx.backend_transfer()
      expected = Nx.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], type: unquote(type_tuple))
      assert concatted == expected
    end

    test "Concat axis #{type_str} 1" do
      o1 = Nx.iota({3, 5}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      o2 = Nx.iota({3, 5}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      concatted = Nx.concatenate([o1, o2], axis: 1) |> Nx.backend_transfer()

      n1 = Nx.iota({3, 5}, type: unquote(type_tuple))
      n2 = Nx.iota({3, 5}, type: unquote(type_tuple))

      expected = Nx.concatenate([n1, n2], axis: 1)
      assert concatted == expected
    end

    test "Concat axis 1 of three 3-dimensional #{type_str} vector" do
      t1 = Nx.iota({3, 5, 7}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      t2 = Nx.iota({3, 5, 7}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      t3 = Nx.iota({3, 5, 7}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      concatted = Nx.concatenate([t1, t2, t3], axis: 1) |> Nx.backend_transfer()

      expected =
        Nx.concatenate(
          [
            Nx.iota({3, 5, 7}, type: unquote(type_tuple)),
            Nx.iota({3, 5, 7}, type: unquote(type_tuple)),
            Nx.iota({3, 5, 7}, type: unquote(type_tuple))
          ],
          axis: 1
        )

      assert concatted == expected
    end

    test "Concat axis 2 of three 3-dimensional #{type_str} vector" do
      t1 = Nx.iota({3, 5, 7}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      t2 = Nx.iota({3, 5, 7}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)
      t3 = Nx.iota({3, 5, 7}, type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      concatted = Nx.concatenate([t1, t2, t3], axis: 2) |> Nx.backend_transfer()

      expected =
        Nx.concatenate(
          [
            Nx.iota({3, 5, 7}, type: unquote(type_tuple)),
            Nx.iota({3, 5, 7}, type: unquote(type_tuple)),
            Nx.iota({3, 5, 7}, type: unquote(type_tuple))
          ],
          axis: 2
        )

      assert concatted == expected
    end

    test "Concat doesn't alter component #{type_str} vectors" do
      t1 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      t2 =
        Nx.tensor([1, 2, 3, 4], type: unquote(type_tuple)) |> Nx.backend_transfer(Ortex.Backend)

      concatted = Nx.concatenate([t1, t2]) |> Nx.backend_transfer()
      second_concatted = Nx.concatenate([t1, t2]) |> Nx.backend_transfer()

      assert concatted == second_concatted
    end
  end)

  test "Concat fails to concat vectors of differing types" do
    assert_raise RuntimeError,
                 "Ortex does not currently support concatenation of vectors with differing types.",
                 fn ->
                   t1 = Nx.tensor([1, 2, 3], type: {:s, 16}) |> Nx.backend_transfer(Ortex.Backend)
                   t2 = Nx.tensor([1, 2, 3], type: {:s, 32}) |> Nx.backend_transfer(Ortex.Backend)
                   _err = Nx.concatenate([t1, t2])
                 end
  end

  # Ignoring these tests, as Nx.Shape takes care of determining if the shape is valid

  # test "Concat fails to concat vectors with invalid default axis" do
  #   assert_raise ArgumentError, "expected all shapes to match {*, 5, 7}, got unmatching shape: {2, 4, 7}", fn() -> 
  #     t1 = Nx.iota({3, 5, 7}) |> Nx.backend_transfer(Ortex.Backend)
  #     t2 = Nx.iota({2, 4, 7}) |> Nx.backend_transfer(Ortex.Backend)
  #     _err = Nx.concatenate([t1, t2])
  #   end
  # end

  # test "Concat fails to concat vectors with invalid provided axis" do
  #   assert_raise ArgumentError, "different dims, given axis" do 
  #     t1 = Nx.iota({3, 5, 7}) |> Nx.backend_transfer(Ortex.Backend)
  #     t2 = Nx.iota({2, 4, 7}) |> Nx.backend_transfer(Ortex.Backend)
  #     _err = Nx.concatenate([t1, t2], axis: 2)
  #   end
  # end
end
