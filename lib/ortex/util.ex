defmodule Ortex.Util do
  @moduledoc false
  @doc """
    Copies the libraries downloaded during the ORT build into a path that
    Elixir can use
  """
  def copy_ort_libs() do
    suppress_warning? = suppress_copy_warning?()
    build_root = Path.absname(:code.priv_dir(:ortex)) |> Path.dirname()
    ort_lib_location = System.get_env("ORT_LIB_LOCATION")
    destination_dir = Path.join([:code.priv_dir(:ortex), "native"])
    File.mkdir_p!(destination_dir)

    search_patterns =
      [build_root, find_project_root(build_root), find_project_root(File.cwd!())]
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()
      |> Enum.flat_map(&patterns_for_root/1)
      |> Enum.concat(ort_lib_location_patterns(ort_lib_location))
      |> Enum.uniq()

    onnx_runtime_paths =
      search_patterns
      |> Enum.flat_map(&Path.wildcard/1)
      |> Enum.uniq()

    existing = Path.wildcard(lib_glob(destination_dir))

    cond do
      onnx_runtime_paths == [] and existing != [] ->
        :ok

      onnx_runtime_paths == [] and not is_nil(ort_lib_location) ->
        raise """
        Unable to locate libonnxruntime binaries.
        ORT_LIB_LOCATION: #{ort_lib_location}
        Searched: #{Enum.join(search_patterns, ", ")}
        Destination: #{destination_dir}
        """

      onnx_runtime_paths == [] and (test_env?() or suppress_warning?) ->
        :ok

      onnx_runtime_paths == [] ->
        :ok

      true ->
        Enum.each(onnx_runtime_paths, fn path ->
          File.cp!(path, Path.join([destination_dir, Path.basename(path)]))
        end)
    end
  end

  defp patterns_for_root(root) do
    [
      Path.join([root, "native/ortex/release"]),
      Path.join([root, "native/ortex/debug"]),
      Path.join([root, "native/ortex/target/release"]),
      Path.join([root, "native/ortex/target/debug"]),
      Path.join([root, "native/ortex/target", "**"])
    ]
    |> Enum.map(&lib_glob/1)
  end

  defp ort_lib_location_patterns(nil), do: []

  defp ort_lib_location_patterns(path) do
    expanded = Path.expand(path)

    if File.dir?(expanded) do
      [expanded, Path.join(expanded, "lib"), Path.join(expanded, "lib64")]
      |> Enum.filter(&File.dir?/1)
      |> Enum.map(&lib_glob/1)
    else
      [expanded]
    end
  end

  defp lib_glob(base) do
    suffix =
      case :os.type() do
        {:win32, _} -> "libonnxruntime*.dll*"
        {:unix, :darwin} -> "libonnxruntime*.dylib*"
        {:unix, _} -> "libonnxruntime*.so*"
      end

    Path.join([base, suffix])
  end

  defp find_project_root(path) do
    expanded = Path.expand(path)

    cond do
      File.exists?(Path.join(expanded, "mix.exs")) ->
        expanded

      expanded == Path.dirname(expanded) ->
        nil

      true ->
        find_project_root(Path.dirname(expanded))
    end
  end

  defp test_env?() do
    cond do
      Code.ensure_loaded?(Mix) and function_exported?(Mix, :env, 0) ->
        Mix.env() == :test

      true ->
        System.get_env("MIX_ENV") == "test"
    end
  end

  defp suppress_copy_warning?() do
    truthy_env?("ORTEX_SKIP_COMPILE") or truthy_env?("ORTEX_SKIP_DOWNLOAD") or
      truthy_env?("ORT_PREFER_DYNAMIC_LINK")
  end

  defp truthy_env?(name) do
    case System.get_env(name) do
      nil -> false
      value -> String.downcase(value) in ["1", "true", "yes", "on"]
    end
  end
end
