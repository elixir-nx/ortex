defmodule Ortex.Util do
  def copy_ort_libs() do
    rust_env =
      case Mix.env() do
        :prod -> "release"
        _ -> "debug"
      end

    case :os.type() do
      {:win32, _} ->
        Path.wildcard(
          Path.join([
            "./_build",
            Mix.env() |> Atom.to_string(),
            "lib/ortex/native/ortex",
            rust_env,
            "libonnxruntime*.dll*"
          ])
        )

      {:unix, :darwin} ->
        Path.wildcard(
          Path.join([
            "./_build/",
            Mix.env() |> Atom.to_string(),
            "lib/ortex/native/ortex",
            rust_env,
            "libonnxruntime*.dylib*"
          ])
        )

      {:unix, _} ->
        Path.wildcard(
          Path.join([
            "./_build/",
            Mix.env() |> Atom.to_string(),
            "lib/ortex/native/ortex",
            rust_env,
            "libonnxruntime*.so*"
          ])
        )
    end
    |> Enum.map(fn x ->
      File.cp!(
        x,
        Path.join([
          "./_build",
          Mix.env() |> Atom.to_string(),
          "lib/ortex/priv/native/",
          Path.basename(x)
        ])
      )
    end)
  end
end
