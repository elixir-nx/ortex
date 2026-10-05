defmodule Ortex.SessionOptions do
  @moduledoc false

  @defaults %{
    intra_op_num_threads: nil,
    inter_op_num_threads: nil,
    execution_mode: nil,
    intra_op_allow_spinning: nil,
    inter_op_allow_spinning: nil
  }

  def normalize(options) do
    unless Keyword.keyword?(options),
      do: raise(ArgumentError, "session options must be a keyword list")

    if length(Keyword.keys(options)) != length(Enum.uniq(Keyword.keys(options))) do
      raise ArgumentError, "duplicate session options"
    end

    Enum.reduce(options, @defaults, fn {key, value}, acc ->
      unless valid?(key, value),
        do: raise(ArgumentError, "invalid session option #{inspect(key)}: #{inspect(value)}")

      Map.put(acc, key, value)
    end)
  end

  defp valid?(key, value) when key in [:intra_op_num_threads, :inter_op_num_threads],
    do: is_integer(value) and value >= 0 and value <= 2_147_483_647

  defp valid?(:execution_mode, value), do: value in [:sequential, :parallel]

  defp valid?(key, value) when key in [:intra_op_allow_spinning, :inter_op_allow_spinning],
    do: is_boolean(value)

  defp valid?(_, _), do: false
end
