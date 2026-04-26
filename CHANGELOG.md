# Changelog

## v0.1.11

### Added

* Precompiled NIF artifacts for common platforms via [`rustler_precompiled`](https://hex.pm/packages/rustler_precompiled). Consumers on `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu` (glibc >= 2.35), `aarch64-unknown-linux-gnu` (glibc >= 2.35), and `x86_64-pc-windows-msvc` no longer need a Rust toolchain — `mix deps.get` downloads a prebuilt NIF binary instead. `onnxruntime` is statically linked into the artifact so there's no separate runtime install.

* `ORTEX_BUILD=true` environment variable to force source compilation, useful for development, unsupported targets, or when building with a non-default execution-provider feature flag (`cuda`, `tensorrt`, `coreml`, `directml`).

* `RELEASE.md` documents the publish process — version bump, tag push, CI matrix, checksum-file regeneration, smoke-test, hex publish.

### Changed

* `:rustler` is now an `optional` dependency. Consumers using precompiled NIFs don't pull it in.

* `:rustler_precompiled` added as a runtime dependency.

* `lib/ortex/native.ex` branches on `ORTEX_BUILD` between `use RustlerPrecompiled` (default) and `use Rustler` + the legacy `compile_crate/copy_ort_libs` dance (source path). Both paths are tested in CI.

* New `.github/workflows/release.yml` builds a 5-target × 2 NIF-version matrix on tag push and uploads tarballs to a draft GitHub release.

### Notes for consumers

If you've been pinning `ortex` at `~> 0.1.10` and your platform is in the supported precompile list above, upgrading to `0.1.11` will silently switch you from a 30-90s Rust build to a sub-second NIF download on every `mix deps.compile`. Behaviour is otherwise unchanged.

If your platform is **not** in the supported list, `mix` falls back to source build automatically — no action needed, but you'll keep needing a Rust toolchain. Consider opening an issue if your target should be added.

## v0.1.10

* Support Elixir 1.19 — switch to `Inspect.Algebra.to_doc` syntax (#48 by @zentourist).

## v0.1.9 and earlier

See git history.
