defmodule Ortex.Native do
  @moduledoc false

  mix_config = Mix.Project.config()
  version = mix_config[:version]
  github_url = mix_config[:package][:links]["GitHub"]

  # `RustlerPrecompiled` handles both paths internally:
  #
  #   * Default path: download the matching prebuilt artifact from the
  #     project's GitHub release and extract it into priv/native/.
  #     No Rust toolchain required.
  #
  #   * `force_build: true` path: fall through to `use Rustler` which
  #     compiles the crate from source. Triggered by the `ORTEX_BUILD`
  #     env var or the standard `:rustler_precompiled, :force_build`
  #     application config. Consumers who want this path must add
  #     `{:rustler, ">= 0.0.0", optional: true}` to their own deps,
  #     since `:rustler` is `optional: true` in this library.
  #
  # The `ort` 2.0.0-rc.x crate links onnxruntime statically into the
  # NIF on every supported platform with stable rustc, so the NIF is
  # self-contained and no `libonnxruntime` sidecar handling is needed
  # on either path.
  use RustlerPrecompiled,
    otp_app: :ortex,
    crate: :ortex,
    version: version,
    base_url: "#{github_url}/releases/download/v#{version}",
    targets: ~w(
      aarch64-apple-darwin
      x86_64-apple-darwin
      x86_64-unknown-linux-gnu
      aarch64-unknown-linux-gnu
      x86_64-pc-windows-msvc
    ),
    force_build: System.get_env("ORTEX_BUILD") in ["1", "true"],
    nif_versions: ["2.16", "2.17"]

  # Stage `libonnxruntime` next to the NIF when source-building. With
  # some toolchains (notably the rustc that ships preinstalled on
  # GitHub-hosted Ubuntu runners) the `ort` crate produces a NIF that
  # dynamically links `libonnxruntime` and expects to find it via the
  # NIF's rpath ($ORIGIN / @loader_path), i.e. in `priv/native/`. The
  # helper finds the sidecar files in the cargo target dir and copies
  # them across.
  #
  # Idempotent + harmless on the other paths:
  #
  #   * Precompiled artifact path: cargo wasn't invoked, so there's
  #     no target dir, the helper finds nothing and is a no-op. The
  #     prebuilt NIF tarball already contains whatever sidecars the
  #     particular target needs.
  #
  #   * Static-link source build (e.g. release.yml's stable rustc):
  #     cargo target dir exists but has no `libonnxruntime` files
  #     because the runtime is linked directly into the NIF. Nothing
  #     to copy. Helper is a no-op.
  #
  #   * Dynamic-link source build (e.g. ci.yml's runner-default
  #     rustc): cargo target dir contains `libonnxruntime.so` (often
  #     as symlinks into the ort cache). Helper resolves and copies
  #     them. Without this step the NIF fails to load at runtime
  #     with `libonnxruntime.so: cannot open shared object file`.
  Ortex.Util.copy_ort_libs()

  # NIF dummies — required so calls compile cleanly before the NIF
  # is loaded. Real implementations are provided by the loaded NIF.
  def init(_model_path, _execution_providers, _optimization_level),
    do: :erlang.nif_error(:nif_not_loaded)

  def run(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def from_binary(_bin, _shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def to_binary(_reference, _bits, _limit), do: :erlang.nif_error(:nif_not_loaded)
  def show_session(_model), do: :erlang.nif_error(:nif_not_loaded)

  def slice(_tensor, _start_indicies, _lengths, _strides),
    do: :erlang.nif_error(:nif_not_loaded)

  def reshape(_tensor, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def concatenate(_tensors_refs, _type, _axis), do: :erlang.nif_error(:nif_not_loaded)
end
