# Upgrading Ortex: Elixir ecosystem, `ort` and ONNX Runtime

Snapshot: 2026-10-29, baseline `main` at `55dc3de`. "Verified" means the step was run in a scratch
copy on macOS arm64 (Elixir 1.20.4, OTP 29.1.2, Rust 1.99.1, and Rust 1.100.0-beta.7 where noted).
Facts about `ort` come from its release notes and from its source at each git tag. Task IDs
(COMPAT-01, ORT-03, ...) refer to [`../TODO.md`](../TODO.md).

## 0. Where things stand

| Component | `main` | Target | Why it matters |
| --- | --- | --- | --- |
| Elixir | `~> 1.14` | `~> 1.17` | Nx 1.0's floor; Elixir 1.20 is needed for OTP 29 |
| Nx | `~> 0.6` (lock 0.6.2) | `~> 0.6 or ~> 1.0` for 0.1.x, `~> 1.0` for 0.2 | `~> 0.6` excludes 1.0, so Ortex cannot sit next to Nx/EXLA/Bumblebee 1.0 |
| Rustler (Hex) | `~> 0.27` (lock 0.29.1; new installs get 0.38.0) | `~> 0.38`, `runtime: false` | private-API use in `native.ex`; build layout changed in 0.37 |
| Rustler (crate) | `0.29` | `0.39` | `resource!` and `init!` lists removed; warnings on current Rust |
| ex_doc | `0.29.4` (exact) | `~> 0.40` | |
| `ort` | `2.0.0-rc.8` | `=2.0.0-rc.15` | ONNX Runtime 1.19.2 to 1.31.0, Rust 1.100 compatibility, 30+ UB fixes |
| ONNX Runtime | 1.19.2 (opset 21, IR 10) | 1.31.0 (opset 26) | newer exported models, Blackwell GPUs, security fixes |
| `ndarray` | `0.16` | `0.17` | must match `ort`'s |
| Rust (MSRV) | rc.8 needs 1.70 | 1.92 (rc.14+) | CI and Docker images |

`ort` versions and the ONNX Runtime each downloads (from the `ort-sys` distribution list at each tag):

| `ort` | Released | ONNX Runtime | API features (default) | Binary host |
| --- | --- | --- | --- | --- |
| rc.8 | 2024 | 1.19.2 | (single version) | `parcel.pyke.io` (still serving) |
| rc.9 | 2024-11-21 | 1.20.0 | | `parcel.pyke.io` |
| rc.10 | 2025-06-05 | 1.22.0 | | `cdn.pyke.io` |
| rc.11 | 2026-01-07 | 1.23.2 | | `cdn.pyke.io` |
| rc.12 | 2026-03-05 | 1.24.2 | `api-17`..`api-24` (`api-24`) | `cdn.pyke.io` |
| rc.13 | 2026-07-28 | 1.28.0 | `api-17`..`api-28` (`api-27`) | `cdn.pyke.io` |
| rc.14 | 2026-10-12 | 1.31.0 | `api-17`..`api-31` (`api-31`) | `cdn.pyke.io` |
| rc.15 | 2026-10-19 | 1.31.0 | `api-17`..`api-31` (`api-31`) | `cdn.pyke.io` |

No date is announced for `ort` 2.0.0 ("No definite timeline", pykeio/ort#451). Do not wait for it.

## 1. Recommended order

1. **Patch release `0.1.11` on `ort` rc.8.** Widen Nx, fix the crashes and the GPU library copy, keep CI
   alive on Rust 1.100. Small, low-risk, unblocks most users.
2. **Modernise without touching `ort`** (the "modernisation" half of PR #47): Rustler crate and
   package upgrade, plain `use Rustler`, the Nx 1.0 contract fixes and the serving fixes.
3. **`ort` rc.15** (the "ort" half of PR #47, retargeted from rc.11).
4. **Session options**: one keyword API (consolidating PRs #50, #51, #52).
5. **Precompiled NIFs** (PR #49) against the final NIF and ONNX Runtime.

Each step builds and passes the full test suite on its own; dependent steps can be stacked PRs.

## 2. Elixir ecosystem (all verified)

### 2.1 `mix.exs`

For `0.1.11` keep the floor low and widen Nx:

```elixir
{:nx, "~> 0.6 or ~> 1.0"},
```

For the modernisation release:

```elixir
def project do
  [elixir: "~> 1.17", ...]
end

defp deps do
  [
    {:rustler, "~> 0.38", runtime: false},
    {:nx, "~> 1.0"},
    {:ex_doc, "~> 0.40", only: :dev, runtime: false}
    # exla/torchx are unused; tokenizers only serves examples: move examples to Mix.install
  ]
end
```

Then `mix deps.unlock --unused` to drop stale lock entries (axon, axon_onnx, ...). With these changes
and no Elixir code changes, `mix compile --warnings-as-errors`, `mix format --check-formatted`,
`mix docs` and all 117 tests pass on Elixir 1.20.4 / OTP 29.1.2 (verified).

### 2.2 Rust crate: Rustler 0.29 to 0.39

`cargo check` with `rustler = "0.39"` fails in three layers; fix each to reveal the next (verified):

1. `rustler::init!` no longer accepts a function list (NIFs are discovered automatically; the list was
   ignored from 0.34 and removed in 0.38), and `rustler::resource` is now private:

   ```rust
   use rustler::ResourceArc;              // was rustler::resource::ResourceArc
   rustler::init!("Elixir.Ortex.Native"); // no list, no load callback needed for resources
   ```

2. `rustler::resource!` is gone (deprecated 0.34, removed 0.38) and `ResourceArc<T>` now requires
   `T: Resource`:

   ```rust
   #[rustler::resource_impl]
   impl rustler::Resource for OrtexModel {}

   #[rustler::resource_impl]
   impl rustler::Resource for OrtexTensor {}
   ```

3. NIF return types must be `RefUnwindSafe` (since 0.35) and `ort::Session` is not:

   ```rust
   impl std::panic::RefUnwindSafe for OrtexModel {}
   ```

After that only lints remain. Other facts: MSRV 1.91; the default NIF version is still 2.15 (loads on
OTP 22 to 29; `nif_version_2_18` exists since 0.38); `Error::Term` still returns `{:error, term}`;
panics are still caught and raised as `:nif_panicked`; 0.39 requires `panic = "unwind"`.

### 2.3 `lib/ortex/native.ex`

Replace the version switch around the private `Rustler.Compiler.compile_crate` and the
`skip_compilation?: true` with:

```elixir
defmodule Ortex.Native do
  @moduledoc false
  use Rustler, otp_app: :ortex, crate: :ortex

  # Still needed only while some builds ship side libraries (e.g. CUDA provider .so files).
  Ortex.Util.copy_ort_libs()

  # NIF stubs ...
end
```

Verified: tests pass, and editing Rust sources now triggers a rebuild (Rustler registers the crate
files as external resources only when it compiles the crate itself).

Rustler 0.37+ builds in `native/ortex/target/<profile>/` and writes `priv/native/ortex.so` into the
source tree, so add `/priv/native/` to `.gitignore`, and make `copy_ort_libs/0` read the real target
directory (`cargo metadata --format-version=1 --no-deps` gives `target_directory`; add `<triple>/` when
`:target` is set and the profile from `:mode`).

### 2.4 `.cargo/config.toml`

With crate 0.36+, macOS needs only the rpath flag:

```toml
[target.'cfg(target_os = "macos")']
rustflags = ["-C", "link-args=-Wl,-rpath,@loader_path"]
```

Verified with crate 0.39 and the static ONNX Runtime 1.19.2 (all tests pass). Keep the Linux `$ORIGIN`
rpath entries. Dropping `-undefined dynamic_lookup` has a second benefit after the `ort` upgrade:
`ort`'s "unsatisfiable execution providers" error is a deliberate unresolved symbol, and the flag
hides it (section 3.3).

### 2.5 Nx contract work exposed by newer Nx

| Item | What to do | TODO |
| --- | --- | --- |
| `Nx.stack/2` raises (Nx >= 0.8 made it a callback) | implement `stack/3`: reshape each input with a new axis, then the concatenate NIF | BUG-14 |
| `backend_copy/3`, `backend_deallocate/1` raise | delegate copy to the transfer logic; return `:ok` | BUG-05 |
| `to_binary/2` ignores `limit` | return exactly `limit` elements (`binary_part/3` on the resource binary is zero-copy) | BUG-04 |
| tensor start indices in `slice/5` | `Nx.to_number/1` each, clamp to `[0, dim - len]` | BUG-06 |
| mixed backends or types in `concatenate/3` | transfer non-Ortex inputs and cast instead of `MatchError`/`RuntimeError` | BUG-15 |
| new types (`{:s, 4}`, `{:u, 2}`, `{:f, 8}`, `{:f8_e4m3fn, 8}`, complex) | raise `ArgumentError` in Elixir before the NIF; never `unimplemented!()` | SAFE-02 |
| `squeeze/3` calls private `Nx.Shape.squeeze/3` | use `out.shape` and `out.names` | BUG-12 |

### 2.6 `Ortex.Serving`

* Accept any list of partition options in `init/3` (BUG-08).
* Do the batch materialisation and `Ortex.run/2` inside the function returned from `handle_batch/3`,
  so inference does not block the serving process and a failing batch does not kill it (BUG-16).
* Prefer `Nx.Defn.Evaluator` (or `Nx.LazyContainer.traverse/3`) for materialising the batch so the
  application's default compiler (often EXLA) is not involved (PERF-06).

### 2.7 CI

Test the floor and the newest versions: Elixir 1.17 / OTP 25 to 27 and Elixir 1.20.4 / OTP 29.1, plus
Nx 0.x and 1.0 while both are allowed. Use current `actions/checkout`, `erlef/setup-beam` on every
OS, and make Linux CI CPU-only (the CUDA build will not compile on Rust 1.100 with rc.8).

## 3. `ort` rc.8 to rc.15 (ONNX Runtime 1.19.2 to 1.31.0)

### 3.1 What the upgrade buys and costs

Buys:

* ONNX Runtime 1.31.0: opset 26 instead of 21, newer contrib ops, Blackwell GPU support (issue #45),
  aarch64 CUDA builds for Jetson Orin/Thor and Grace (issue #41), MIGraphX builds for AMD on x64 Linux.
* Compatibility with Rust 1.100's build-dir layout (rc.14), more than 30 undefined-behaviour fixes
  and several memory-leak fixes (rc.15).
* Every pyke package is a single **static** `libonnxruntime.a`, so a CPU NIF is self-contained
  (verified: a 27.4 MB release NIF on macOS arm64 that depends only on system libraries).
* APIs for strings, sequences and maps, complex numbers (`num-complex` feature), output selection,
  run cancellation and loading from memory (section 4).

Costs (platform floor of pyke's prebuilt binaries):

| Platform | Requirement |
| --- | --- |
| Linux (x86_64, aarch64) | **glibc >= 2.39 and libstdc++ >= 13.2** (Ubuntu 24.04+, Debian 13+), per `ort`'s docs and confirmed by symbol inspection (`__isoc23_strto*`, `std::ios_base_library_init`). rc.8's libraries needed far less (around glibc 2.32), so Ubuntu 22.04, Debian 12 and RHEL 9 lose prebuilt support. |
| x86_64 | AVX2 (x86-64-v3 baseline since rc.11) |
| macOS | arm64 only, macOS >= 13.4 (no Intel macOS binaries since rc.11) |
| musl, armv7/arm, i686, riscv, FreeBSD | no prebuilt binaries |
| CUDA | CUDA >= 13.2 and cuDNN >= 9.23 (no CUDA 12 since rc.13); Turing (sm_75) or newer, so Pascal and Volta are out |
| Build host | Rust >= 1.92; OpenSSL headers on Linux (downloads use `tls-native`; the `tls-rustls` feature avoids OpenSSL) |

For older Linux hosts, link Microsoft's official ONNX Runtime 1.31.1 (glibc >= 2.28) through
`ORT_LIB_PATH` with `ORT_PREFER_DYNAMIC_LINK=1` and ship it next to the NIF, or use `load-dynamic`.

### 3.2 Breaking changes by release candidate

| RC | Change | Ortex code affected |
| --- | --- | --- |
| rc.9 | Crate-root re-exports removed ("undo the flattening"); only `ort::{Error, ErrorCode, Result, init, init_from}` stay at the root. `Value::dtype()` returns `&ValueType`; `ValueType::Tensor` gains `dimension_symbols`; `SessionOutputs` no longer derefs to a `BTreeMap` (`.get(name)` still works). | every `use ort::...`; the `ValueType` pattern in `tensor.rs` |
| rc.10 | `Session::run(&mut self)`. `try_extract_tensor` renamed to `try_extract_array` (returns a view; the old name now returns `(&Shape, &[T])`). All `TryFrom<Array>` impls for values removed: use `Tensor::from_array(owned)` or zero-copy `TensorRef::from_array_view` (standard layout). `ValueType::Tensor { ty, shape, dimension_symbols }`; `tensor_dimensions()` became `tensor_shape()`. **`half` is no longer a default feature.** New element types (Complex, Float8, Int4/Uint4, Undefined). Binaries move to `cdn.pyke.io`, become static, and download with native TLS. MSRV 1.81. | model sharing; 13 extraction calls and 13 input conversions in `tensor.rs`; `model::show`; `Cargo.toml` |
| rc.11 | `session.inputs`/`outputs` fields became methods returning `&[Outlet]` (`name()`, `dtype()`). `execution_providers` renamed `ep`, `XXXExecutionProvider` renamed `XXX` (old names removed in rc.13/rc.14). `ndarray 0.17`. **`GraphOptimizationLevel::Level3` now means `ORT_ENABLE_LAYOUT`; the new `All` means `ORT_ENABLE_ALL`.** Binaries: no Intel macOS, AVX2 baseline, Clang builds, and a client-package build that **disables thread spinning by default** (the notes advise re-enabling inter/intra-op spinning for servers). MSRV 1.88. | `model.rs` introspection; `utils.rs` providers and optimisation level |
| rc.12 | Multiversioning with `api-*` features. **`ORT_LIB_LOCATION` renamed `ORT_LIB_PATH`** (rc.15 still reads both; the old name will not reach 2.0.0). `ort::tensor` merged into `ort::value` (breaks PR #47's `ort::tensor::TensorElementType`). Builder methods return `BuilderResult` (`?` still works). **New default `AutoDevicePolicy::MaxEfficiency`** (see 3.3). | docs, `util.ex`, imports |
| rc.13 | Provider structs exist only with their Cargo feature (`#[cfg]` each arm). Unsatisfiable feature sets are reported through a deliberate link error (`lax-feature-matching` relaxes it). `TensorElementType` is `#[non_exhaustive]`. No CUDA 12 builds. Sessions with map inputs/outputs no longer panic on commit. | `utils.rs::map_eps`, `constants.rs`, `Cargo.toml` features |
| rc.14 | **Manual `Environment`:** `let env = ort::init().build()?; Session::builder(&env)?`. **Only one environment per OS process, ever** (verified: a second `build()` fails with "only one environment is allowed per process" even after dropping the first). `ort::execution_providers` removed; `EP::NAME`/`is_available()`. Rust 1.100 build-dir support; Blackwell x64, aarch64 CUDA and MIGraphX binaries; MSRV 1.92. | `model::init`; a new environment owner |
| rc.15 | 30+ UB fixes and memory-leak fixes. `run_async` takes owned inputs; `ExecutionProviderLibrary::unregister` removed; `with_external_initializer_file_in_memory` takes `Arc<[u8]>`. | none directly |

### 3.3 Design decisions to make first

**Environment (ORT-04).** There can be exactly one ONNX Runtime environment per OS process, and its
options (name, telemetry, logger, global thread pool, default providers) are fixed when it is first
built. Every `Session` holds a clone of it (it is an `Arc`), so it lives as long as any model.
Therefore:

* Build it once, inside the NIF: in the Rustler `load` callback (options passed from application
  config through `use Rustler, load_data: ...`) or lazily on the first `Ortex.load`. Keep it in a
  `static OnceLock<Environment>` (lazy fallible initialisation needs a `Mutex<Option<_>>` or
  `once_cell`). A verified run with a static environment, sessions leaked as BEAM resources are at
  `halt`, and a normal exit was clean three times out of three on macOS arm64.
* `ort`'s request that libraries "allow the user to pass their own `Environment`" becomes: let the
  application configure the one environment (e.g. `config :ortex, :environment, global_thread_pool:
  [...], telemetry: false, log_level: :warning`).
* Set `with_telemetry(false)` (it defaults to true; pyke's binaries contain no telemetry client, but
  Microsoft's 1.29+ binaries used with `load-dynamic` do; also set `ORT_DISABLE_TELEMETRY=1` there).
* A global thread pool (`with_global_thread_pool`) replaces one intra-op pool per model. Sessions then
  use it unless built with `with_independent_thread_pool()`, so per-session thread options must raise
  or force an independent pool in that case.
* Logging: `with_logger(LoggerFunction)` receives every ONNX Runtime log line. To reach Elixir's
  `Logger`, push lines into a channel drained by one dedicated OS thread that sends them to a
  registered process with `OwnedEnv::send_and_clear` (the callback runs on ONNX Runtime threads).
* Do not purge or reload the NIF module while ONNX Runtime threads may be running.

**Concurrency (ORT-03, PERF-04, SAFE-06).** `run`, `run_with_options`, `run_binding` and `run_async`
all take `&mut self`. `ort`'s documentation says this is because some ONNX Runtime internals (certain
provider allocators, statistics) are not thread safe and "previous versions of `ort` that allowed
concurrent inference often saw crashes and memory corruption"; its maintainer has declined to revert
it (pykeio/ort#415, #597) and to add a session pool (#469). Upstream, CUDA graphs and DirectML forbid
concurrent `Run()` on one session. Options:

| Option | Behaviour | Cost |
| --- | --- | --- |
| `Mutex<Session>` (PRs #46, #47) | correct; one inference at a time per model | throughput regression for concurrent callers; blocked callers occupy dirty IO threads (10 by default), so bound concurrency on the Elixir side (batching, a queue) |
| Pool of N sessions per model | parallel up to N | memory per session; CPU sessions can share weights with `SessionBuilder::with_prepacked_weights` |
| One session per `Nx.Serving` partition | matches Nx's partition model | only helps serving users |

Start with the mutex (simple and safe), cache metadata so `show_session` never locks, add a
`:pool_size` option later, and benchmark concurrent serving before and after. `SessionOutputs`
borrows the session, so collect output names before `run` and copy outputs out before unlocking.
`RunOptions::terminate()` can cancel a running inference from another thread.

**Execution providers (ORT-05).**

* One Rust function owns the atom-to-provider table: `ort::ep::{CPU, CUDA, TensorRT, DirectML,
  CoreML, MIGraphX, OpenVINO, WebGPU, ...}::default().build()`, each arm behind `#[cfg(feature = ...)]`.
  A requested provider that is not compiled in returns an error naming the missing Cargo feature.
* **Auto device policy (pykeio/ort#662, open).** Since rc.12 sessions default to
  `AutoDevicePolicy::MaxEfficiency`, and registering `ep::CPU` does not override it, so with ONNX
  Runtime 1.31 a `[:cpu]` session on macOS can run on CoreML. When only `:cpu` is requested, call
  `.with_auto_device(AutoDevicePolicy::PreferCPU)`.
* **The "unsatisfiable providers" link error does not protect Ortex today.** `ort` reports it as an
  unresolved symbol; with `-undefined dynamic_lookup` on macOS the NIF links anyway (verified: a
  740 KB cdylib without ONNX Runtime) and only fails when the BEAM loads it ("symbol not found
  '_OrtGetApiBase'"). Linux shared objects allow undefined symbols too. Drop the macOS flag (2.4),
  validate feature/target combinations in `native.ex`, and load the NIF in CI on every target.
* Valid prebuilt combinations in rc.15 (`ort-sys` `dist.tsv`):

  | Target | Packages |
  | --- | --- |
  | `aarch64-apple-darwin` | `coreml`; `coreml,webgpu` |
  | `x86_64-unknown-linux-gnu` | CPU; `openvino`; `webgpu`; `nvrtx`; `migraphx`; `cuda13,tensorrt,nvrtx` |
  | `aarch64-unknown-linux-gnu` | CPU; `cuda13,tensorrt,nvrtx` |
  | `x86_64-pc-windows-msvc` | `directml`; `webgpu`; `openvino`; `nvrtx,directml`; `cuda13,tensorrt,nvrtx,directml` |
  | `aarch64-pc-windows-msvc` | `directml` |

  No prebuilt binaries exist for ROCm, QNN, XNNPACK, ACL, oneDNN, CANN, Vitis and others; those need
  `ORT_LIB_PATH`. ROCm was removed from ONNX Runtime in 1.23 (use MIGraphX); ACL is deprecated (1.31).
* CUDA packages are a static core plus provider libraries loaded at run time
  (`libonnxruntime_providers_shared.so`, `..._cuda.so` (121 MB), `..._tensorrt.so`,
  `..._nv_tensorrt_rtx.so`). They must sit next to the NIF (`priv/native`); this is the only remaining
  job for `copy_ort_libs/0`. `ort::ep::cuda::preload_dylibs` (feature `preload-dylibs`) can preload the
  CUDA 13 runtime libraries.
* Thread spinning is off by default in pyke's builds since rc.11. For servers, expose
  `with_intra_op_spinning`/`with_inter_op_spinning` (or the global pool's `with_spin_control`).

**Optimisation level (ORT-06).** Map Ortex's default `3` to `GraphOptimizationLevel::All`
(`ORT_ENABLE_ALL`), not `Level3` (`ORT_ENABLE_LAYOUT` since rc.11), or expose the five levels as atoms.

**Metadata (ORT-07, BUG-01).** Read inputs and outputs once in `init` (`session.inputs()` /
`session.outputs()` return `&[Outlet]`; `ValueType::tensor_shape()` gives the shape), convert to plain
Rust data (name, element type, dimensions with `None` for dynamic axes, symbolic names) and store it
next to the session. `show_session` then never locks. `ValueType`'s `Display` (e.g.
`Tensor<f32>(dyn, 3, 224, 224)`) is a better human-readable form than `Debug`.

### 3.4 Measured effort (compile probe, Rust 1.99.1)

Starting from `main` with only `ort = "=2.0.0-rc.15"` and `ndarray = "0.17"` changed, `cargo check`
reports errors in four waves (each fix reveals the next):

| Wave | Errors | Causes |
| --- | --- | --- |
| 1 | 41 | name resolution: crate-root imports (`model.rs` 6, `tensor.rs` 18, `utils.rs` 17) |
| 2 | 43 | 13x `Value: TryFrom<Array<T>>` missing; 13x `.into_owned()` on the new `try_extract_tensor` tuple; 7x provider structs gated by features; 4x private `inputs`/`outputs`; 2x `half` types (feature off); `Session::builder()` needs `&Environment`; `&ValueType`; the `dimensions` pattern |
| 3 | 2 | `session.run` needs `&mut`; non-exhaustive `TensorElementType` |
| 4 | 1 | `SessionOutputs` borrow (collect output names first) |

About 172 changed lines reach a clean build, also clean with `--features coreml` and
`--features cuda,tensorrt`. (The probe built a new environment per load, which would fail on the second
load at run time; see 3.3.) From PR #47's rc.11 code the remaining delta is small: about four import
fixes (`ort::ep::*`, `ort::value::TensorElementType`), `#[cfg]` gating for providers,
`Session::builder(&env)` with the environment lifecycle, plus the semantic fixes it lacks (level `3` to
`All`, `PreferCPU`, `ORT_LIB_PATH`).

Both `main` and PR #47 use caret requirements (`"2.0.0-rc.8"`, `"2.0.0-rc.11"`), which allow later
release candidates; a plain `cargo update` would jump to rc.15 and break the build. Pin with `=`.
Published Hex packages ship `Cargo.lock`, so released versions are not affected.

### 3.5 Code migration checklist

1. `native/ortex/Cargo.toml`: `ort = { version = "=2.0.0-rc.15", features = ["half"] }` (add
   `"num-complex"` for complex support, `"tls-rustls"` to avoid OpenSSL), `ndarray = "0.17"`; forward
   features `cuda`, `tensorrt`, `coreml`, `directml` (optionally `migraphx`, `openvino`, `webgpu`,
   `nvrtx`); drop `rocm`; remove `rustls`, `tracing-subscriber`, `num-traits`; `rust-version = "1.92"`.
2. New `runtime.rs`: owns the single `Environment` and its configuration; hands out session builders.
3. `model.rs`: `Session::builder(env)?`, optimisation level `All`, providers, `PreferCPU` for CPU-only,
   `commit_from_file`; metadata snapshot; session behind a mutex (or pool); no `unsafe impl Sync`.
4. `tensor.rs`: inputs through `TensorRef::from_array_view` (zero copy; arrays are standard layout) or
   `Tensor::from_array(...)`; outputs through `try_extract_array::<T>()`; a wildcard arm returning an
   error; bool stays `u8`.
5. `utils.rs` (or a new `ep.rs`): safe `from_binary` (length check and copy), the provider table with
   `cfg` gates and clear errors.
6. `lib.rs`: Rustler 0.39 style (section 2.2) if not done already.
7. Elixir: `config/config.exs` feature sets only for valid combinations; `util.ex` copies only
   `libonnxruntime_providers_*` (and WebGPU's Dawn library) and reads `ORT_LIB_PATH` then
   `ORT_LIB_LOCATION`; update `inspect` format and docs (platform floor, CUDA 13, opset 26,
   serialised runs per model).
8. CI: Rust >= 1.92, cache the `ort` download cache (`ORT_CACHE_DIR`, default `~/.cache/ort.pyke.io` on
   Linux and `~/Library/Caches/ort.pyke.io` on macOS), load the NIF on every target.
9. Tests: TODO TEST-01 to TEST-04, plus concurrent `run` from many processes, bool inputs, `f16`/`bf16`
   outputs, optimisation levels, provider errors, `[:cpu]` staying on CPU on macOS.

### 3.6 Using an ONNX Runtime newer than `ort` bundles

rc.15 supports ONNX Runtime API 31, so it can run against 1.31.1:

* `ORT_LIB_PATH=<custom build>` (static preferred, or `ORT_PREFER_DYNAMIC_LINK=1`), or
* the `load-dynamic` feature with `ort::init_from(path)` or `ORT_DYLIB_PATH`. Relative paths resolve
  against the executable, which for a NIF is `beam.smp`, so pass absolute paths. `load-dynamic`
  refuses libraries older than the compiled API level and only logs for newer ones.

Other build-time variables in rc.15: `ORT_LIB_PROFILE`, `ORT_SKIP_DOWNLOAD`, `ORT_OFFLINE`,
`ORT_CXX_STDLIB`, `ORT_CUDA_VERSION`, `ORT_CACHE_DIR`; run time: `ORT_DYLIB_PATH`, `ORT_LOG`.

## 4. After the upgrade: things that become possible

| Capability | `ort` rc.15 API | TODO |
| --- | --- | --- |
| Load from memory (issue #29; also in rc.8) | `SessionBuilder::commit_from_memory(&[u8])` (copies) or `commit_from_memory_directly` (borrows) | ONNX-08 |
| Cancel a slow load | `SessionBuilder::canceler()` | SAFE-05 |
| String tensors | `try_extract_strings`, `try_extract_string_array`, `Tensor::from_string_array` | ONNX-02 |
| Sequences and maps (scikit-learn ZipMap) | `try_extract_sequence::<DynMapValueType>()`, then `try_extract_map::<K, V>()` or `try_extract_key_values` | ONNX-03 |
| Complex tensors (`{:c, 64}`, `{:c, 128}`) | `num-complex` feature: `Complex32`, `Complex64` are primitive element types | ONNX-05 |
| Float8, Int4/Uint4, Int2/Uint2, Float4 | no Rust types; build with `DynTensor::new` and copy raw bytes | ONNX-04 |
| Output selection, run tags, cancellation | `Session::run_with_options` with `RunOptions::with_outputs`, `with_tag`, `terminate` | ONNX-06, ONNX-14 |
| External weights from memory | `with_external_initializer_file_in_memory(name, Arc<[u8]>)` | ONNX-09 |
| Device-resident outputs | `Session::create_binding()` + `run_binding` (note open bug #667: `bind_input` can deadlock in rc.15 for borrowed tensors already on the session device) | ONNX-13 |
| Plugin providers and device selection | `Environment::register_ep_library(name, path)`, `env.devices()`, `SessionBuilder::with_devices` | ONNX-12 |
| Profiling, dimension overrides, optimised model output | `with_profiling`, `with_dimension_override`, `with_optimized_model_path` | ONNX-07, ONNX-15 |
