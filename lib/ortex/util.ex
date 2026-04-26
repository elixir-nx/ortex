defmodule Ortex.Util do
  @moduledoc false

  require Logger

  @doc """
    Copies the libonnxruntime libraries downloaded by the `ort` crate
    during cargo build into the application's `priv/native/` directory,
    so the NIF's rpath (`$ORIGIN` / `@loader_path`) can resolve them at
    load time on platforms where onnxruntime is dynamically linked.

    The cargo target directory's location depends on the Rustler version:

    * **Rustler ≥ 0.30** builds in the natural cargo location at
      `<crate-source>/target/release/`. For Ortex itself, that resolves
      to `<project>/native/ortex/target/release/`. For consumers, it
      resolves to `<project>/deps/ortex/native/ortex/target/release/`.

    * **Rustler < 0.30** staged artifacts at
      `<build-root>/native/<crate>/release/`. Kept as a fallback for
      anyone still on that path.

    The function tries each known layout in order, copies whatever
    libonnxruntime files it finds, and warns if no candidate directory
    contained any. Static-link platforms (notably macOS with the
    `ort` crate's default download strategy) legitimately produce no
    files to copy — the warning is informational rather than fatal.
  """
  def copy_ort_libs() do
    destination_dir = Path.join([:code.priv_dir(:ortex), "native"])
    pattern = "libonnxruntime*.#{shared_lib_ext()}*"

    case find_runtime_dir() do
      nil ->
        Logger.debug(
          "Ortex.Util.copy_ort_libs/0: no cargo target directory found in any " <>
            "expected location. This is normal on platforms where onnxruntime " <>
            "is statically linked into the NIF (e.g. macOS)."
        )

        :ok

      rust_path ->
        rust_path
        |> Path.join(pattern)
        |> Path.wildcard()
        |> Enum.each(fn src ->
          dest = Path.join([destination_dir, Path.basename(src)])
          File.cp!(src, dest)
        end)
    end
  end

  # Walks the candidate cargo target directories in priority order and
  # returns the first one that exists, or nil. The candidates cover both
  # in-repo builds (Ortex testing itself) and consumer builds (Ortex as
  # a dep), and both modern (`target/release/`) and legacy
  # (`<build_root>/native/<crate>/release/`) Rustler layouts.
  defp find_runtime_dir do
    build_root = Path.absname(:code.priv_dir(:ortex)) |> Path.dirname()
    # build_root is `_build/<env>/lib/ortex`. Going four levels up
    # reaches the project root (Ortex repo) or consumer project root.
    project_root = build_root |> Path.join("../../../..") |> Path.expand()

    [
      # Rustler ≥ 0.30 — Ortex consumed as a dependency.
      Path.join([project_root, "deps", "ortex", "native", "ortex", "target", "release"]),
      # Rustler ≥ 0.30 — Ortex building itself in its own repo.
      Path.join([project_root, "native", "ortex", "target", "release"]),
      # Rustler < 0.30 — legacy staged location under build_root.
      Path.join([build_root, "native", "ortex", "release"])
    ]
    |> Enum.map(&Path.expand/1)
    |> Enum.find(&File.dir?/1)
  end

  defp shared_lib_ext do
    case :os.type() do
      {:win32, _} -> "dll"
      {:unix, :darwin} -> "dylib"
      {:unix, _} -> "so"
    end
  end
end
