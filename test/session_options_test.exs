defmodule Ortex.SessionOptionsTest do
  use ExUnit.Case, async: true

  @tuned [
    intra_op_num_threads: 1,
    inter_op_num_threads: 1,
    execution_mode: :sequential,
    intra_op_allow_spinning: false,
    inter_op_allow_spinning: false
  ]

  test "legacy arities, empty options, and configured sessions produce equivalent results" do
    reference = run(Ortex.load("models/tinymodel.onnx"))

    models = [
      Ortex.load("models/tinymodel.onnx", [:cpu]),
      Ortex.load("models/tinymodel.onnx", [:cpu], 3),
      Ortex.load("models/tinymodel.onnx", [:cpu], 3, []),
      Ortex.load("models/tinymodel.onnx", [:cpu], 3, @tuned),
      Ortex.load(
        "models/tinymodel.onnx",
        [:cpu],
        3,
        Keyword.merge(@tuned,
          execution_mode: :parallel,
          intra_op_allow_spinning: true,
          inter_op_allow_spinning: true
        )
      )
    ]

    for model <- models, do: assert(run(model) == reference)
  end

  test "each option is independently optional and zero thread counts retain runtime choice" do
    for options <- Enum.map(@tuned, &[&1]) ++ [[intra_op_num_threads: 0, inter_op_num_threads: 0]] do
      assert %Ortex.Model{} = Ortex.load("models/tinymodel.onnx", [:cpu], 3, options)
    end
  end

  test "invalid options fail before native loading rather than silently using defaults" do
    for options <- [
          [unknown: 1],
          [intra_op_num_threads: -1],
          [inter_op_num_threads: 1.5],
          [intra_op_num_threads: 2_147_483_648],
          [execution_mode: :typo],
          [intra_op_allow_spinning: 0],
          [inter_op_allow_spinning: nil],
          [intra_op_num_threads: 1, intra_op_num_threads: 2],
          %{}
        ] do
      assert_raise ArgumentError, fn -> Ortex.load("missing.onnx", [:cpu], 3, options) end
    end
  end

  defp run(model) do
    Ortex.run(model, {Nx.broadcast(1, {1, 100}) |> Nx.as_type(:s32), Nx.broadcast(1.0, {1, 100})})
    |> Tuple.to_list()
    |> Enum.map(&Nx.to_binary/1)
  end
end
