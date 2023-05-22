defmodule Ortex.Util do
  def copy_ort_libs() do
    build_root = Path.absname(:code.priv_dir(:ortex)) |> Path.dirname()

    rust_env =
      case Path.join([build_root, "native/ortex/release"]) |> File.ls() do
        {:ok, _} -> "release"
        _ -> "debug"
      end

    case :os.type() do
      {:win32, _} ->
        Path.wildcard(
          Path.join([
            build_root,
            "native/ortex",
            rust_env,
            "libonnxruntime*.dll*"
          ])
        )

      {:unix, :darwin} ->
        Path.wildcard(
          Path.join([
            build_root,
            "native/ortex",
            rust_env,
            "libonnxruntime*.dylib*"
          ])
        )

      {:unix, _} ->
        Path.wildcard(
          Path.join([
            build_root,
            "native/ortex",
            rust_env,
            "libonnxruntime*.so*"
          ])
        )
    end
    |> Enum.map(fn x ->
      File.cp!(
        x,
        Path.join([
          :code.priv_dir(:ortex),
          "native",
          Path.basename(x)
        ])
      )
    end)
  end
end
