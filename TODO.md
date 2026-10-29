# Ortex TODO: review findings and roadmap

Snapshot: 2026-10-29. Baseline: `main` at `55dc3de` (Hex `0.1.10` + PR #48). Architecture background
for every item is in [`docs/`](docs/README.md); contributor rules are in [`AGENTS.md`](AGENTS.md); the
step-by-step upgrade guide is [`docs/upgrading.md`](docs/upgrading.md).

**How these findings were produced.** A full read of every Elixir, Rust, test, config, CI and example file.
`main` was built and tested on macOS arm64 with Elixir 1.20.4 / OTP 29.1.2 / Rust 1.99.1. A fresh
consumer install was simulated (a new Mix app depending on Ortex, with fresh dependency resolution).
Dependency-bump probes covered Nx 1.0.0, Hex Rustler 0.38 and crate Rustler 0.39. Edge cases and
memory behaviour were probed with scripts against the baseline build. All open pull requests were
reviewed statically; code from forks was not executed. All 25 GitHub issues were read with their
comments. Upstream release notes and sources were checked for `ort`, ONNX Runtime, ONNX, Nx,
Rustler and rustler_precompiled. Line numbers refer to the baseline commit.

**Legend.** Priority: **P0** hurts users today or can crash or corrupt; **P1** needed for the upgrade, or a
major gap; **P2** important improvement; **P3** cleanup. IDs are stable. Tick the box when an item is
done and leave it in place with a note. "Verified" means the behaviour was reproduced by running code.

---

## 1. Answers in brief

### 1.1 What is left to get onto the latest Elixir

The code itself already runs on the newest Elixir. On Elixir 1.20.4 / OTP 29.1.2, `main` compiles with
`--warnings-as-errors` and passes all 117 tests. A fresh application can install it and run a model,
resolving Nx 0.13.1 and Rustler 0.38.0. Bumping to `nx ~> 1.0`, `rustler ~> 0.38` and `ex_doc ~> 0.40`
also passes all tests with no Elixir code changes (verified). What is left:

1. **Release.** Hex `0.1.10` predates PR #48, so `inspect(model)` is broken for Hex users on Elixir 1.19+
   (BUG-13). It also requires `{:nx, "~> 0.6"}`, which cannot resolve next to Nx/EXLA/Bumblebee 1.0
   (COMPAT-01).
2. **Nx contract gaps that newer Nx exposes.** `Nx.stack/2` raises on Ortex tensors since Nx 0.8
   (BUG-14). Mixed-backend `Nx.concatenate/2` raises `MatchError` (BUG-15). Partitioned serving crashes
   (BUG-08). There are also the older gaps: `backend_copy`, `backend_deallocate`, `to_binary` limits,
   and dynamic slices.
3. **Rustler.**
   - The crate is at 0.29. Moving to 0.39 means three layers of mechanical compile errors (COMPAT-02).
   - Hex Rustler 0.37+ moved the cargo build directory, which silently broke `copy_ort_libs` for
     dynamically linked (GPU) builds (BUILD-01).
   - `lib/ortex/native.ex` should use plain `use Rustler` instead of the private
     `Rustler.Compiler.compile_crate` (COMPAT-07).
4. **Floors, pins and CI.** Raise the Elixir floor to 1.17 (Nx 1.0's floor). Unpin `ex_doc 0.29.4`. CI
   currently covers only Elixir 1.17 / OTP 27 (COMPAT-03, COMPAT-04, CI-02).

### 1.2 What it takes to reach the latest ONNX Runtime

The move is from `ort 2.0.0-rc.8` (ONNX Runtime 1.19.2, max opset 21) to `ort 2.0.0-rc.15` (ONNX
Runtime 1.31.0, opset 26). ONNX Runtime 1.31.1 (2026-10-26) is not bundled by any `ort` release yet.
`ort` has not reached 2.0.0; waiting for it has stalled Ortex since 2025 (issue #45), so pin an exact RC.

Seven release candidates of breaking changes touch every Rust file:

- module paths;
- `Session::run(&mut self)`, which forces a concurrency decision;
- a mandatory `Environment` that `ort` asks libraries to let users supply;
- execution providers gated at compile time, with strict matching against the downloadable binaries;
- `GraphOptimizationLevel::Level3` no longer meaning "all";
- new tensor element types, `ndarray 0.17`, CUDA 13 only, no Intel macOS binaries;
- a minimum Rust of 1.92.

PR #47 already does most of the work up to rc.11.

Measured effort: from `main`, a compile probe needed about 172 changed lines to reach a clean
`cargo check` against rc.15. From PR #47 the remaining delta is about four import fixes, provider
`#[cfg]` gating, and the environment lifecycle (ONNX Runtime allows exactly one environment per OS
process). The upgrade also raises the platform floor of the prebuilt binaries:

- glibc 2.39 / libstdc++ 13.2 on Linux;
- AVX2 on x86_64;
- arm64-only macOS 13.4+;
- CUDA 13.2+ with cuDNN 9.23+.

Older Linux hosts would need Microsoft's ONNX Runtime binaries (glibc 2.28+) linked through
`ORT_LIB_PATH` or loaded with `load-dynamic`.

One item is time-sensitive. Rust 1.100, the next stable (about 2026-11-12), changes cargo's build-dir
layout. `ort` warns that versions before rc.14 "will fail to build when `copy-dylibs` is enabled".
Reproduced on Rust 1.100 beta: dynamically linked builds of `main` fail, and that includes this repo's
own Linux CI configuration (BUILD-10). The full checklist is in section 3.2 and `docs/upgrading.md`.

### 1.3 What is missing for complete ONNX support

- Tensor types beyond the 12 numeric ones: string, complex, and the 8-, 4- and 2-bit families. Today
  they panic.
- Non-tensor values (sequences, maps, optionals), which also panic.
- Named inputs and output selection.
- Session options (three competing PRs: #50, #51, #52).
- Loading from memory (issue #29). The API already exists in `ort`.
- Model metadata.
- Custom-op libraries and onnxruntime-extensions. Issue #32's contrib ops are already present in ONNX
  Runtime 1.19.2.
- Execution-provider options and availability reporting.
- Device-resident outputs (IoBinding), run cancellation, profiling, logging and telemetry.
- Models above opset 21, which needs the upgrade.

Section 3.5 has the full list.

---

## 2. Roadmap

Ordered so each step ships on its own. Use stacked PRs where a step depends on an unmerged one.

| Phase | Goal | Items | Existing PRs |
| --- | --- | --- | --- |
| 0 | **Patch release 0.1.11 on the current `ort`**: unblock Nx 1.0, Elixir 1.19+ inspect, crashes, GPU loading | BUG-13, COMPAT-01, BUG-14, BUILD-01, SAFE-01, SAFE-02, SAFE-05, BUILD-10, BUG-02, BUG-04, BUG-05, BUG-07, BUG-08, TEST-01, CI-01, CI-02, DOC-01 | pieces of #47 |
| 1 | **Modernise without changing `ort`** (the first half of #47, which the maintainer asked to split) | COMPAT-02 to COMPAT-07, BUG-01, BUG-03, BUG-06, BUG-15, BUG-16, BUILD-05 to BUILD-09, BUILD-11, REPO-* | #47 (part A) |
| 2 | **`ort` rc.15 / ONNX Runtime 1.31** | ORT-01 to ORT-17, SAFE-06, PERF-01, PERF-04 | #47 (part B, retargeted to rc.15); close #46 |
| 3 | **Session options API** | ARCH-07, PERF-05 | #52 as base, absorbing #50 and #51 |
| 4 | **Precompiled NIFs and platforms** | BUILD-02 to BUILD-04, CI-05 | #49, rebased last (it can ship earlier on rc.8 if phase 2 stalls) |
| 5 | **Complete ONNX support** | ONNX-01 to ONNX-16, ARCH-08 | |
| 6 | **Architecture and performance follow-ups** | ARCH-01 to ARCH-11, PERF-02, PERF-03, PERF-06 to PERF-08 | |

---

## 3. Findings

### 3.1 Elixir ecosystem compatibility

- [ ] **COMPAT-01 (P0) Allow Nx 1.0.**
  - `mix.exs:35` has `{:nx, "~> 0.6"}`, i.e. `< 1.0.0`. Nx, EXLA and Torchx 1.0.0 shipped 2026-09-10.
  - Verified: the suite passes on Nx 1.0.0 unchanged, because `Ortex.Backend` generates stubs from
    `behaviour_info`. That covers callbacks added in 1.0 (`stack/3`, `to_pointer/2`, `from_pointer/5`,
    `block/4`), callbacks removed (21), and arity changes (`gather/4`, `indexed_add/5`, `indexed_put/5`).
  - For 0.1.11 use `"~> 0.6 or ~> 1.0"` and test both ends. Narrow it later.
  - PR #47's `"~> 0.10"` still excludes 1.0.
- [ ] **COMPAT-02 (P1) Upgrade the Rustler crate 0.29 to 0.39.** These are the probe's three error layers, in order:
  1. Rustler 0.38 removed the NIF list in `rustler::init!` (`lib.rs:107-118`), and the `rustler::resource` module is private, so import `rustler::ResourceArc` from the root.
  2. Rustler 0.38 removed `rustler::resource!` (`lib.rs:119-123`). Use `#[rustler::resource_impl] impl rustler::Resource for OrtexModel {}` and the same for `OrtexTensor`.
  3. `ort::Session` is not `RefUnwindSafe`; the bound has existed since Rustler 0.35. Add `impl std::panic::RefUnwindSafe for OrtexModel {}`.

  Crate MSRV is 1.91. NIF version defaults to 2.15, which loads on OTP 22 to 29.
- [ ] **COMPAT-03 (P1) Raise the Elixir floor.** `mix.exs:8`: `elixir: "~> 1.17"` (Nx 1.0's floor). OTP 29 needs Elixir 1.20.
- [ ] **COMPAT-04 (P2) Unpin ex_doc.** `mix.exs:37` exact `0.29.4` to `~> 0.40` (`mix docs` verified clean with 0.40.4).
- [ ] **COMPAT-05 (P2) Refresh the lock.** Nx 0.6.2 emits about 890 deprecation warnings on Elixir 1.20,
  including unpinned bitstring `size(...)`, which is a future hard error. Remove stale `mix.lock` entries.
- [ ] **COMPAT-06 (P3) Use built-in `JSON`** (Elixir 1.18+) in `examples/distilbert` instead of the undeclared `Jason`.
- [ ] **COMPAT-07 (P1) Stop calling Rustler's private API.**
  - `lib/ortex/native.ex:4-24` calls `@doc false` `Rustler.Compiler.compile_crate` behind a version switch and then sets `use Rustler, skip_compilation?: true`. That also means Mix never sees the Rust sources (BUILD-05).
  - Replace it with `use Rustler, otp_app: :ortex, crate: :ortex` followed by the library copy step.
  - Verified: tests pass and Rust edits then trigger a rebuild.
  - Declare `{:rustler, "~> 0.38", runtime: false}`.

### 3.2 ONNX Runtime upgrade (`ort` rc.8 to rc.15)

Code-level detail is in `docs/upgrading.md`. Do these as one PR stack after phase 1, starting from PR #47's
`ort` commit (`9a9b8f2b`) and crediting #46.

- [ ] **ORT-01 (P1) Pin `ort = { version = "=2.0.0-rc.15", features = ["half"] }`.**
  - `half` has not been a default feature since rc.10.
  - Bump `ndarray = "0.17"` in the same change.
  - Today's caret requirement (`"2.0.0-rc.8"`) allows later RCs, so a plain `cargo update` would jump to rc.15 and break the build.
- [ ] **ORT-02 (P1) Module paths (rc.9, rc.11, rc.12).** Only `ort::{Error, ErrorCode, Result, init, init_from}` remain at the crate root. Imports now come from:
  - `ort::session::{Session, SessionInputValue, SessionOutputs, RunOptions}`
  - `ort::session::builder::{GraphOptimizationLevel, AutoDevicePolicy}`
  - `ort::value::{Value, DynValue, ValueType, TensorElementType, Tensor, TensorRef, Outlet}` (`ort::tensor` merged into `ort::value` in rc.12, which breaks PR #47's import)
  - `ort::ep::{ExecutionProviderDispatch, CPU, CUDA, ...}`, renamed from `execution_providers::XXXExecutionProvider`.
- [ ] **ORT-03 (P1) `Session::run` takes `&mut self` (rc.10+).**
  - Pick a concurrency model.
  - PR #47's `Mutex<Session>` serialises every inference on a model, a regression from today's parallel runs (verified parallel on rc.8).
  - Options: a small session pool per model, or one session per `Nx.Serving` partition.
  - Remove `unsafe impl Sync` (`model.rs:30`).
- [ ] **ORT-04 (P1) Mandatory `Environment` (rc.14), one per OS process.**
  - Sessions are built with `Session::builder(&env)`.
  - A second `ort::init().build()` fails even after the first environment is dropped (verified), so per-model environments are impossible.
  - Build it once in the NIF (in the `load` callback with options from app config, or lazily) and keep it in a `static OnceLock`. A static environment with leaked sessions exited cleanly in 3 of 3 runs on macOS.
  - Expose its options through application config: global thread pool, logger bridge to `Logger`, `with_telemetry(false)`.
  - Design: `docs/upgrading.md` section 3.3.
- [ ] **ORT-05 (P1) Execution providers are gated at compile time (rc.13).**
  - Wrap each `ort::ep::*` arm in `#[cfg(feature = ...)]`, and return a clear error for providers that were not compiled in.
  - `ort`'s "unsatisfiable features" check is a deliberate unresolved symbol, which `-undefined dynamic_lookup` hides on macOS. A bad combination builds a NIF without ONNX Runtime that only fails at load (verified). So: drop the flag (BUILD-07), validate feature/target combinations in `native.ex`, and load the NIF in CI per target.
  - Since rc.12 the default `AutoDevicePolicy::MaxEfficiency` can put a `[:cpu]` session on CoreML with ONNX Runtime 1.31 (pykeio/ort#662). Use `PreferCPU` when only `:cpu` is requested.
  - Drop `:rocm` (ONNX Runtime 1.23 removed it; use MIGraphX, prebuilt on x64 Linux since rc.14). ACL is deprecated in 1.31.
- [ ] **ORT-06 (P1) The optimisation level was renumbered.**
  - In rc.8, `Level3` is `ORT_ENABLE_ALL`. In rc.11 and later, `Level3` is `ORT_ENABLE_LAYOUT`, and a new `All` is `ORT_ENABLE_ALL`. Both verified in source.
  - Map Ortex's default `3` to `All`, or the default silently loses optimisations. PR #47 maps it to `Level3`.
- [ ] **ORT-07 (P1) Introspection API.**
  - Inputs and outputs are accessed through methods, with `name()` and `dtype()`.
  - `ValueType::Tensor` carries `shape` and `dimension_symbols`.
  - Snapshot this metadata at load time so `show_session` never touches the session (ORT-12).
- [ ] **ORT-08 (P1) Extraction API.** `try_extract_tensor` became `try_extract_array`. Match new element types with a fallback arm instead of exhaustively.
- [ ] **ORT-09 (P2) Zero-copy inputs.** `TensorRef::from_array_view` (rc.10+) replaces cloning every input. It needs standard layout, which `OrtexTensor` already guarantees.
- [ ] **ORT-10 (P1) Platform floor of the prebuilt binaries.**
  - Linux: glibc >= 2.39 and libstdc++ >= 13.2, i.e. Ubuntu 24.04+ and Debian 13+ (per `ort` docs and symbol inspection; rc.8 needed about glibc 2.32).
  - x86_64 needs AVX2.
  - macOS: arm64 only, 13.4+. No Intel macOS since rc.11, which ends issue #34's platform.
  - No musl, armv7 or riscv.
  - CUDA 13.2+ and cuDNN 9.23+ (no CUDA 12 since rc.13): Turing or newer, so Pascal and Volta are out.
  - rc.14 adds Blackwell x64 and aarch64 Jetson/Grace CUDA builds (helps #45 and #41).
  - For older Linux, offer Microsoft's ONNX Runtime 1.31.1 (glibc 2.28+) via `ORT_LIB_PATH` + dynamic link or `load-dynamic`. Re-check the target list before promising platforms (BUILD-02).
- [ ] **ORT-11 (P1) Build requirements.**
  - rc.14+ needs Rust 1.92.
  - rc.10+ downloads through native TLS, so Linux builds need OpenSSL headers. The `tls-rustls` feature avoids that.
  - `ORT_LIB_LOCATION` was renamed `ORT_LIB_PATH` in rc.12. rc.15 still reads the old name, but it will not survive into 2.0.0.
  - Update the Docker guidance (issue #43).
- [ ] **ORT-12 (P2) `show_session` must not lock.** It is a normal-scheduler NIF; behind a mutex it would block a scheduler during inference. PR #47 has this bug.
- [ ] **ORT-13 (P2) `api-*` features (rc.12).** Defaults use the newest API level. With `default-features = false`, enable the matching `api-NN`.
- [ ] **ORT-14 (P2) Remove the `rustls` pin** (`Cargo.toml:20`). Nothing uses it, and it forces `ring` to build for the target, which breaks Nerves (issue #41). Also remove `tracing-subscriber` and `num-traits`.
- [ ] **ORT-15 (P2) Re-validate packaging.**
  - Every pyke package is now a static `libonnxruntime.a`. Verified: the CPU NIF is self-contained, 27.4 MB on macOS arm64.
  - CUDA packages add provider libraries loaded at run time: `libonnxruntime_providers_{shared,cuda,tensorrt,nv_tensorrt_rtx}.so`, with the CUDA one at 121 MB. They must sit next to the NIF, and copying them is the only job left for `copy_ort_libs/0`.
  - Re-check the precompiled matrix (PR #49).
- [ ] **ORT-16 (P3) Bring-your-own ONNX Runtime.** Offer `load-dynamic` (`ort::init_from(path)` or an absolute `ORT_DYLIB_PATH`) so users can run a newer ONNX Runtime (e.g. 1.31.1), or Microsoft's builds for older glibc, without waiting for `ort`.
- [ ] **ORT-17 (P2) Thread spinning is off by default in pyke builds since rc.11** (client-package build). rc.8 spins by default. For servers, expose `with_intra_op_spinning`/`with_inter_op_spinning` or the global pool's `with_spin_control`, and benchmark before and after the upgrade.

### 3.3 Correctness bugs

All "verified" items were reproduced on the baseline build (`docs/testing-and-ci.md` section 4).

- [ ] **BUG-01 (P1) Model inspect output does not match the docs (verified).**
  - `model.rs:54-79` returns `format!("{:#?}", value_type)`, a multi-line `Tensor { ty: Float32, dimensions: [...] }`, and `-1` for dynamic axes.
  - The README and `Ortex.Model` docs promise `{"x", "Int32", [nil, 100]}`.
  - Return structured data and format it in Elixir (ARCH-05).
- [ ] **BUG-02 (P1) Input count is not validated (verified).**
  - `model.rs:92` zips inputs with the model inputs, so extra inputs are silently dropped.
  - Missing inputs fail inside a kernel with an unrelated message.
  - Validate the count and name the expected inputs. PR #47 adds the count check.
- [ ] **BUG-03 (P1) Execution providers fall back silently (verified; issue #17).**
  - Unknown atoms map to CPU (`utils.rs:105`).
  - Registration failures are only `tracing::warn!`ed by `ort`, and Ortex installs no subscriber, so `[:cuda]` without CUDA quietly runs on CPU.
  - Strings raise a bare `ArgumentError`.
  - Fix: reject unknown atoms (PR #47), surface registration failures (`ExecutionProviderDispatch::error_on_failure()` for providers the user asked for, or a Logger warning), and add `Ortex.available_execution_providers/0`.
- [ ] **BUG-04 (P1) `Nx.to_binary/2` ignores `limit` (verified).**
  - `backend.ex:44-49`, `utils.rs:84-91`. Nx expects exactly the first `limit` elements.
  - Fix in Rust from the tensor's own element size, or zero-copy in Elixir with `binary_part/3` on the resource binary.
- [ ] **BUG-05 (P1) `backend_copy/3` and `backend_deallocate/1` raise (verified).**
  - Nx.Serving's distributed docs recommend `Nx.backend_copy(&1, Nx.BinaryBackend)`, which fails on Ortex outputs.
  - Make `backend_copy` delegate to the transfer logic, and have `backend_deallocate` return `:ok`.
- [ ] **BUG-06 (P2) `Nx.slice/4` with tensor start indices raises `ArgumentError` (verified; issue #8).**
  - Nx turns every index into a scalar tensor when any is dynamic and expects the backend to clamp to `[0, dim - len]` (`backend.ex:81-85`).
- [ ] **BUG-07 (P1) NIF errors are ignored in shape ops.**
  - `slice`, `reshape`, `squeeze` and `concatenate` (`backend.ex:81-125`) store a `{:error, msg}` tuple as the tensor ref.
  - Match and raise (PR #47).
- [ ] **BUG-08 (P1) `Ortex.Serving.init/3` accepts exactly one partition (verified).**
  - `serving.ex:62` matches `[_defn_options]`. `partitions: true` gives one option list per partition (`max_concurrency` with the Evaluator, one per device with EXLA), so it raises `FunctionClauseError`.
  - Accept any list (PR #47).
- [ ] **BUG-09 (P3) Optional Nx callbacks are stubbed to raise** (`backend.ex:141`). This blocks Nx fallbacks on Nx < 1.0. Nx 1.0 has no optional callbacks, so the item disappears once the floor is 1.0.
- [ ] **BUG-10 (P2) Shape NIFs copy whole arrays on normal schedulers** (`lib.rs:73-105`: `slice`, `reshape`, `concatenate`; issues #8, #11). Mark them `DirtyCpu`, or make reshape and squeeze metadata-only.
- [ ] **BUG-11 (P2) Bool inputs panic on common tensors.**
  - A model `bool` input given `Nx.tensor([1, 0])` (`s32` on Nx 0.9+) or a `u8` value other than 0 or 1 panics in `tensor.rs:187-203`.
  - Accept any integer type (non-zero means true), or return an error that names the input.
- [ ] **BUG-12 (P3) Small dead or fragile code.**
  - `backend_transfer(tensor, Nx.Tensor, _)` is a dead clause (`backend.ex:51-54`).
  - `@derive {Nx.Container, containers: [:ref]}` is unused, and wrong for a reference (`backend.ex:16`).
  - `inspect/2` relies on `min(:infinity, n)` term ordering (`backend.ex:73-76`).
  - The `:dnnl` atom is declared but unmapped (`constants.rs:20`).
  - Levels outside 1..3 silently disable optimisation (`utils.rs:111-118`).
  - `squeeze/3` recomputes a shape Nx already put in `out` via the private `Nx.Shape.squeeze/3`.
- [ ] **BUG-13 (P0) Hex 0.1.10 has a broken `inspect(model)` on Elixir 1.19+.** Elixir 1.19 changed `Inspect.List.inspect/2`'s return value. PR #48 fixed it on `main`, but there has been no release since. Release 0.1.11.
- [ ] **BUG-14 (P1) `Nx.stack/2` raises on Ortex tensors (verified with Nx 1.0; affects Nx 0.8+).**
  - Since Nx 0.8, `stack` is a backend callback instead of `new_axis` plus `concatenate`.
  - Fresh installs already resolve Nx 0.13.1, so this affects users today.
  - Implement `stack/3` with reshape plus the concatenate NIF.
- [ ] **BUG-15 (P2) Mixed-backend `Nx.concatenate/2` raises `MatchError` (verified with Nx 1.0).**
  - Nx 1.0 hands mixed Binary/Ortex lists to the non-Binary backend (`backend.ex:116-120`); mixed types raise `RuntimeError`.
  - Transfer and cast inputs instead.
- [ ] **BUG-16 (P1) `Ortex.Serving` runs inference inside `handle_batch/3`** (`serving.ex:68-72`). Nx.Serving calls it in its GenServer and runs the returned function in a supervised task. Ortex's blocking call stalls the serving process, and a raise kills the whole serving instead of one batch. Move materialisation and `Ortex.run/2` into the `:execute` function.

### 3.4 Memory safety and crashes

- [ ] **SAFE-01 (P0) Unchecked raw-pointer view over BEAM binaries.**
  - `utils.rs:16-20, 41-78` builds `ArrayViewMut::from_shape_ptr(shape, bin.as_ptr() as *mut T)`.
  - There is no length check (a short binary reads out of bounds), no alignment guarantee for sub-binaries, and it creates a mutable view of immutable memory.
  - Validate `len == product(shape) * size_of::<T>()` and copy into a `Vec<T>` (PR #47's `array_from_binary`).
- [ ] **SAFE-02 (P0) Panics reachable from Elixir (verified for `c64`, `s4`, `u2`, `f8`, `f8_e4m3fn`).** Each becomes `ErlangError :nif_panicked`, with the message only on stderr. Return errors (PR #47 covers most) and raise `ArgumentError` in Elixir for unsupported types. The sites:
  - `utils.rs:78` `unimplemented!()` for unsupported Nx types;
  - `tensor.rs:239` `panic!` for sequence and map outputs;
  - `tensor.rs:279-281` `todo!()` for string outputs;
  - `tensor.rs:187-203` bool conversion.
- [ ] **SAFE-03 (P2) Remove `unsafe impl Sync for OrtexModel`** (`model.rs:30`). It is redundant on rc.8 (`ort` already makes `Session` `Send + Sync`) and unsound once `run` needs `&mut self`.
- [ ] **SAFE-04 (P3) Clean up `get_bytes`** (`tensor.rs:219-228`). It uses `std::mem::zeroed()` to learn an element size and assumes standard layout. Use `size_of::<T>()` with `as_slice_memory_order()` or `bytemuck`. Add `// SAFETY:` comments to every `unsafe` block.
- [ ] **SAFE-05 (P0) Loading a large model can crash the VM (issue #42).**
  - ONNX Runtime graph optimisation at level 3 overflowed the C stack while loading a 300 MB model: a silent BEAM exit on Windows, a segfault on WSL.
  - `init` runs on a dirty IO scheduler, whose threads default to 40 kilowords of stack (320 KiB on 64-bit; `+sssdio`).
  - Build the session on a dedicated Rust thread with a large, configurable stack (`std::thread::Builder::stack_size`) and join it.
  - Document `+sssdio` and the optimisation-level workaround meanwhile.
- [ ] **SAFE-06 (P1) Concurrent `run` on one session is unsafe for some providers.**
  - Today `Ortex.run/2` calls `Session::run(&self)` on one shared session from many dirty schedulers. That is legal in rc.8, and verified to work on the CPU provider.
  - `ort` changed `run` to `&mut self` because "previous versions of `ort` that allowed concurrent inference often saw crashes and memory corruption".
  - Upstream, ONNX Runtime forbids concurrent `Run()` on one session with DirectML and CUDA graphs.
  - Until the upgrade (ORT-03), serialise runs per model when a non-CPU provider is active, or document the restriction.

### 3.5 Complete ONNX support

| ID | Pri | Gap | Notes |
| --- | --- | --- | --- |
| ONNX-01 | P1 | Models above opset 21 / IR 10 cannot load | Needs the ONNX Runtime upgrade (opset 26 on 1.28 to 1.31). Issues #16 and #36 were early versions of this. |
| ONNX-02 | P2 | String tensors (in and out) | Common in NLP and scikit-learn models. Represent as lists of binaries. Today output hits `todo!()`. rc.15: `try_extract_strings`, `try_extract_string_array`, `Tensor::from_string_array`. |
| ONNX-03 | P2 | Non-tensor values: sequences, maps, optionals | scikit-learn classifiers (ZipMap) return `seq(map(int64, float))`, which panics today. Return Elixir lists and maps. rc.15: `try_extract_sequence`, `try_extract_map`/`try_extract_key_values`; rc.13 fixed a panic when committing sessions with map inputs/outputs. |
| ONNX-04 | P2 | `FLOAT8E5M2` and `FLOAT8E4M3FN` (Nx `{:f, 8}`, `{:f8_e4m3fn, 8}`); `INT4`/`UINT4`, `INT2`/`UINT2` (Nx `{:s, 4}`, `{:u, 4}`, `{:s, 2}`, `{:u, 2}`) | Nx 1.0 has all of these. Mind sub-byte packing order (ONNX packs two 4-bit values per byte, low nibble first) and that Nx accepts sub-byte bitstrings since 1.0. `ort` has no Rust element types for these: build `DynTensor`s from raw bytes. |
| ONNX-05 | P3 | `COMPLEX64`/`COMPLEX128`, `FLOAT8E4M3FNUZ`, `FLOAT8E5M2FNUZ`, `FLOAT4E2M1`, `FLOAT8E8M0` | Complex maps to Nx `{:c, 64}`/`{:c, 128}` through `ort`'s `num-complex` feature (ONNX Runtime's complex kernel coverage is small). The others have no Nx type: return a clear "unsupported" error. |
| ONNX-06 | P2 | Named inputs and output selection | Accept `%{"input_ids" => t}`. Add an option to request specific outputs (ONNX Runtime then skips work), and optionally return a map. |
| ONNX-07 | P1 | Session options | Threads, execution mode, spinning, memory pattern, arena, deterministic compute, config entries, optimised-model path, free-dimension overrides. See ARCH-07. |
| ONNX-08 | P2 | Load from memory (issue #29) | `SessionBuilder::commit_from_memory(&[u8])` exists in rc.8 and rc.15 (rc.15 also has a borrowing `commit_from_memory_directly` and a `canceler()` for aborting slow loads); about 20 lines plus a dirty NIF. The maintainer's "wait for ort 2.0" reason no longer applies. |
| ONNX-09 | P2 | External data files, ORT-format models | Weights in `*.onnx_data` load from a path only when side by side. From memory they need `with_external_initializer_file`/`..._in_memory`. |
| ONNX-10 | P2 | Model metadata | Producer, version, description, custom metadata map, opset imports, symbolic dimension names. |
| ONNX-11 | P2 | Custom-op libraries, onnxruntime-extensions (issue #32) | `com.microsoft` contrib ops such as `MatMulNBits` and `GroupQueryAttention` are already in the ONNX Runtime 1.19.2 binary (checked on the macOS artifact). Add a CI model using one, then close #32. Custom-op library registration is still missing. |
| ONNX-12 | P2 | Execution-provider options and availability | Device id, memory limits, TensorRT engine cache, CoreML flags. Prior art: the unmerged branch `tm-execution-providers`. |
| ONNX-13 | P3 | IoBinding / device-resident outputs | Keep data on the GPU between calls (`create_binding` + `run_binding`; open bug pykeio/ort#667: `bind_input` can deadlock in rc.15 for borrowed tensors already on the device). |
| ONNX-14 | P3 | Run options | Cancel long inferences (`RunOptions::terminate`), run tags, per-run log level, output selection (`with_outputs`) through `run_with_options`. |
| ONNX-15 | P3 | Profiling and logging | Route ONNX Runtime logs to Logger (`model.rs:39` TODO); expose profiling output. |
| ONNX-16 | P3 | Telemetry | `:telemetry` spans for load and run, like the rest of the Nx ecosystem. |

### 3.6 Build, packaging, distribution

- [ ] **BUILD-01 (P0) `copy_ort_libs` finds nothing with current Rustler.**
  - Hex Rustler 0.37.0 stopped setting the cargo target directory. Builds now land in `native/ortex/target/<profile>/`; for a dependency, `deps/ortex/native/ortex/target/release/` (verified with 0.38).
  - `util.ex:8-17` only searches `_build/<env>/lib/ortex/native/ortex/{release,debug}`, so dynamically linked ONNX Runtime builds (CUDA, TensorRT, a shared `ORT_LIB_LOCATION`) load without their libraries.
  - It also misses Windows' `onnxruntime*.dll` (issue #26) and cross-compiled `<triple>/` directories (issue #41).
  - Short term: ask `cargo metadata` for `target_directory`, then add `<triple>/` and the profile.
  - Long term: make the native build self-contained (ARCH-02).
- [ ] **BUILD-02 (P1) Precompiled NIFs (issue #27, PR #49).** Section 5 has the PR review. Decide:
  - provider coverage per artifact, since CoreML and DirectML need their features;
  - GPU variants (`rustler_precompiled` `variants`);
  - a glibc baseline (PR #49 builds on ubuntu-22.04, i.e. glibc 2.35+, which excludes Debian 11, Ubuntu 20.04, RHEL 8 and Amazon Linux 2);
  - the target list after the upgrade, with no Intel macOS;
  - NIF version 2.15 only, which covers OTP 22 to 29;
  - the checksum step in the release flow.

  Require `rustler_precompiled >= 0.10.1`: earlier versions fail on GitHub's large response headers.
- [ ] **BUILD-03 (P1) Builds depend on pyke's CDN.** A transient 404 from `parcel.pyke.io` failed PR #50's Linux CI.
  - Cache ONNX Runtime downloads in CI.
  - Document `ORT_LIB_LOCATION` for offline builds.
  - Precompiled NIFs remove the dependency for most users.
- [ ] **BUILD-04 (P1) Cross-compiling and Nerves (issue #41).**
  - A working community recipe (`abelino/ortex_fw`) sets `config :ortex, Ortex.Native, target: "aarch64-unknown-linux-gnu", env: [{"CC", ""}, {"CFLAGS", ""}, {"CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER", "aarch64-nerves-linux-gnu-gcc"}]`.
  - What Ortex should do:
    - derive the triple from Nerves' environment and pass `target:` and `env:` to Rustler, letting user overrides win;
    - drop `rustls` (ORT-14);
    - fix the copy step for triple directories and the target OS;
    - document that 32-bit ARM, musl and riscv64 need a self-built ONNX Runtime via `ORT_LIB_LOCATION`.
  - On the device, the static ONNX Runtime still needs `libstdc++`.
  - aarch64 Nerves may simply use PR #49's `aarch64-unknown-linux-gnu` artifact; test on hardware.
- [ ] **BUILD-05 (P2) Rust edits do not trigger rebuilds (verified).** `skip_compilation?: true` means no `@external_resource`. COMPAT-07 fixes it.
- [ ] **BUILD-06 (P2) Dev config differs from what users get (issue #17).**
  - `config/config.exs:13-18` enables `cuda`/`tensorrt` on Linux, so CI tests the dynamic CUDA build while users get the static CPU build.
  - Mix never loads a dependency's config (verified), so shipping `config/config.exs` (`mix.exs:46`) does nothing, and PR #47's `ORTEX_FEATURES` (read in that file) has the same flaw.
  - Default to CPU (PR #47 does).
  - If an env var is wanted, read it in `lib/ortex/native.ex`.
  - Remove the no-op `:dbg_callback` line (`config/config.exs:2-4`).
  - Stop shipping the file.
- [ ] **BUILD-07 (P2) Simplify `.cargo/config.toml`.** With the Rustler crate 0.36 or later, macOS needs neither `-undefined dynamic_lookup` nor `-fapple-link-rtlib`; only the rpath flag remains. This was verified with crate 0.39 on macOS arm64. Re-check after the ORT upgrade.
- [ ] **BUILD-08 (P2) Cargo hygiene.**
  - `edition = "2018"` to 2021 or 2024.
  - Remove the unused `tracing-subscriber`, `num-traits` and `rustls`.
  - Set `rust-version`.
  - Pin `ort` exactly.
- [ ] **BUILD-09 (P3) Heavy unused dev deps.** `exla` and `torchx` (`mix.exs:38-39`) are unused and make `:dev` builds and `mix docs` download large artifacts. `tokenizers` only serves the examples. Move the examples to `Mix.install` scripts.
- [ ] **BUILD-10 (P0, deadline about 2026-11-12) Rust 1.100 breaks dynamically linked builds (verified).**
  - Rust 1.100 ships cargo's new build-dir layout: `OUT_DIR` becomes `target/<profile>/build/<pkg>/<hash>/out`, one level deeper.
  - `ort-sys` rc.8's `copy-dylibs` locates the target directory as `OUT_DIR.ancestors().nth(3)`, now `target/<profile>/build`, and `unwrap()`s a symlink into `.../build/examples`, which does not exist.
  - Reproduced with Rust 1.100.0-beta.7 and a dynamic ONNX Runtime: `failed to run custom build command for ort-sys v2.0.0-rc.8 ... panicked at build.rs:111:66 ... No such file or directory`. The same build passes on 1.99.1.
  - Static CPU builds (what dependents get by default) still build on the beta.
  - Affected:
    - every `cuda`/`tensorrt` build on x86_64 Linux/Windows, including **this repo's own Linux CI**, because `config/config.exs` enables those features;
    - any build with a dynamic `ORT_LIB_LOCATION`.
  - Real fix: `ort` rc.14+ (section 3.2).
  - Stopgap for 0.1.11: make Linux CI CPU-only, and pin Rust 1.99 for GPU builds (document it, or add `native/ortex/rust-toolchain.toml` for GPU users).
- [ ] **BUILD-11 (P2) Ignore `/priv/native/`.** Rustler 0.37+ writes `priv/native/ortex.so` into the source tree (`_build/.../priv` becomes a symlink), so add `/priv/native/` to `.gitignore`.

### 3.7 Performance

- [ ] **PERF-01 (P2) Inputs are cloned into `ort` values on every run** (three copies for bool inputs). Use zero-copy `TensorRef` views after the upgrade (ORT-09).
- [ ] **PERF-02 (P3) Outputs are copied out of ONNX Runtime memory** (`into_owned`). Keeping the `ort` value inside the resource would make the output path zero-copy.
- [ ] **PERF-03 (P3) `from_binary` copies BEAM binaries into Rust.** Revisit only if profiling shows it matters, e.g. by passing borrowed binaries straight into `run`.
- [ ] **PERF-04 (P1, with ORT-03) Do not serialise inference per model after the upgrade.** Use a session pool or per-partition sessions, and benchmark concurrent `Nx.Serving` before and after.
- [ ] **PERF-05 (P2) Thread oversubscription.** Every session has its own intra-op pool with spinning on. Expose thread and spinning options and a process-wide pool (ARCH-07, ORT-04), and document settings for BEAM nodes. Note that `session.intra_op_num_threads` style config keys and `ORT_INTRA_OP_NUM_THREADS` only exist in ONNX Runtime 1.28 and 1.29+.
- [ ] **PERF-06 (P3) Batch stacking uses the default `defn` compiler.** `Ortex.Serving` stacks with whatever compiler is the default (EXLA in many apps, which adds device copies). Use `Nx.Defn.Evaluator` explicitly (PR #47), or `Nx.LazyContainer.traverse/3`.
- [ ] **PERF-07 (P3) `concatenate` copies twice** (`tensor.rs:359-363`). Concatenate into a standard-layout array directly.
- [ ] **PERF-08 (P2) Native tensor memory is invisible to the BEAM GC (issue #44).**
  - `ResourceArc::new` accounts about 100 bytes per tensor, while the data is a separate allocation. Dead tensors are freed only when their owner process collects them, and old-heap garbage waits for a full sweep.
  - Probe results (macOS, rc.8, 600 runs):
    - Fixed shapes: RSS rose about 45 MB, then stayed flat; GC returned it.
    - Random shapes: it peaked about 180 MB above baseline, then drifted down.
    - No unbounded growth was reproduced.
  - Options:
    - keep the resource binary in `%Ortex.Backend{}` so its size counts;
    - implement `backend_deallocate` with interior mutability;
    - expose arena options;
    - document `:erlang.garbage_collect/1` and short-lived inference processes.

### 3.8 Architecture and API (Responsibility-Driven Design)

Background: `docs/architecture.md` sections 3 and 9.

- [ ] **ARCH-01 (P2)** `Ortex.Model.run/2` builds and destructures `%Ortex.Backend{}` inside `%Nx.Tensor{}` (`model.ex:44-64`). Give `Ortex.Backend` that job (`wrap(ref, shape, type)`, `ref!(tensor)`).
- [ ] **ARCH-02 (P2)** `Ortex.Util.copy_ort_libs/0` mirrors Rustler and `ort` build internals. The native build should place everything the NIF needs, or link statically.
- [ ] **ARCH-03 (P2)** Execution-provider knowledge lives in five places: config, Cargo features, `constants.rs`, `map_eps` and docs. The native layer should own the catalogue and report it.
- [ ] **ARCH-04 (P3)** Elixir passes the output dtype to `concatenate` and the bit width to `to_binary`, though the tensors know both. Let Rust derive and validate them.
- [ ] **ARCH-05 (P1)** Model metadata leaks `ort`'s `Debug` format (BUG-01). Return structured data.
- [ ] **ARCH-06 (P2)** Give the NIF error contract one owner: a single boundary module converts native results to `{:ok, _}`/`{:error, %Ortex.Error{}}`, and public functions decide whether to raise.
- [ ] **ARCH-07 (P1) Options API.**
  - `Ortex.load(path, opts)` takes keyword options.
  - An `Ortex.SessionOptions` module validates them; it owns names, defaults, docs and the legacy positional mapping.
  - A Rust `SessionOptions` (`NifMap`) applies them through a single `init/2` NIF.
  - Keep `load(path, eps)` and `load(path, eps, opt)` working.
  - This unifies PRs #50, #51 and #52 (section 5).
- [ ] **ARCH-08 (P2)** Named inputs and outputs (ONNX-06).
- [ ] **ARCH-09 (P3)** Telemetry events (ONNX-16).
- [ ] **ARCH-10 (P3)** A logging bridge from ONNX Runtime to Logger (ONNX-15).
- [ ] **ARCH-11 (P2)** Decide the backend policy. Issue #23 ("Implement argmax") was closed without code. Either document "storage only: transfer before compute" and point users to `Nx.backend_transfer/2`, or implement selected ops.

### 3.9 Tests

- [ ] **TEST-01 (P1)** `test/shape/concat_test.exs:6-19` maps every dtype except `s8` to `{:s, 16}`, so 10 of 12 dtypes are never exercised. Fix the table (PR #47).
- [ ] **TEST-02 (P1)** Turn every probe result in `docs/testing-and-ci.md` section 4 into a regression test.
- [ ] **TEST-03 (P1) Add tiny generated fixture models** (`onnx.helper`, no PyTorch):
  - bool in and out;
  - int64 and other integer outputs (these would have caught PR #46's integer corruption);
  - outputs with different shapes, so declared output order is tested (issues #26 and #37);
  - string, sequence and map outputs;
  - several dynamic axes;
  - external data;
  - a `com.microsoft` contrib op (issue #32).
- [ ] **TEST-04 (P2)** A concurrency test (many tasks, one model) and a bounded-memory test, tagged slow.
- [ ] **TEST-05 (P2)** Rust unit tests (`cargo test`) for conversions and error paths.
- [ ] **TEST-06 (P3)** `reshape_test.exs` has a TODO saying it does not validate the Ortex path; assert on the backend and the data.
- [ ] **TEST-07 (P3)** The resnet50 tests never run in CI. Use a small committed model instead, or cache the export.
- [ ] **TEST-08 (P3)** `test/test_helper.exs:5` uses `IO.warn`, so `mix test --warnings-as-errors` fails after a passing run whenever resnet50 is missing. Use `IO.puts(:stderr, ...)`.

### 3.10 CI

- [ ] **CI-01 (P1)** `actions/checkout@v2` (`ci.yml:20, 37`) is deprecated, and macOS installs floating Homebrew Elixir/Erlang. Use current actions and `erlef/setup-beam` everywhere.
- [ ] **CI-02 (P1)** Build a matrix of floor and newest: Elixir 1.17 / OTP 25 to 27 and Elixir 1.20.4 / OTP 29.1, plus Nx 0.x and 1.0 while both are supported.
- [ ] **CI-03 (P2)** Cache `deps`, `_build`, `~/.cargo` and the ONNX Runtime download (BUILD-03).
- [ ] **CI-04 (P2)** Add `cargo fmt --check`, `cargo clippy -- -D warnings`, `mix docs`, `mix hex.build`, and a Rust beta build (BUILD-10).
- [ ] **CI-05 (P2)** Add Windows and Linux aarch64 jobs.
- [ ] **CI-06 (P2)** Fork PRs need maintainer approval to run: PR #52 is `action_required`. PR #51's runs failed with "workflow file issue" and zero jobs. Review the workflow settings.
- [ ] **CI-07 (P3)** Add Dependabot or Renovate for cargo, mix and GitHub Actions.

### 3.11 Documentation

- [ ] **DOC-01 (P1) README.** Fix:
  - the `inspect` output;
  - argmax now prints `s32`, not `s64` (Nx 0.9+ default integer type);
  - installation requirements (Rust, network at build time, supported targets);
  - GPU setup for applications (features in the app's own config, CUDA libraries);
  - the silent provider fallback.
- [ ] **DOC-02 (P2) Module docs.** Fix `Ortex.Model`'s example output. In `Ortex.load`, cover the provider list, optimisation levels and `ORT_LIB_LOCATION`.
- [ ] **DOC-03 (P2)** Add `CHANGELOG.md` (PR #49 adds one).
- [ ] **DOC-04 (P3)** Replace `native/ortex/README.md` (the Rustler template, mentioning an `add/2` NIF) with a pointer to `docs/native-layer.md`.
- [ ] **DOC-05 (P3)** Ship the user-relevant parts of `docs/` (execution providers, building, Docker, Nerves) as ExDoc extras.
- [ ] **DOC-06 (P3)** The Python scripts write into the current directory instead of `models/` (PR #47 fixes `export_resnet.py`). `export_resnet.py` uses the deprecated `pretrained=True`.
- [ ] **DOC-07 (P3)** Document the supported opset and IR versions per release, and the external-data requirement (issues #16, #36).

### 3.12 Repository hygiene

- [ ] **REPO-01 (P3)** Delete the merged 2023 branches `develop`, `jv-readme` and `tm-backend-ops`. Keep `tm-execution-providers` (2024-03, provider option structs) or convert it into an issue.
- [ ] **REPO-02 (P3)** `mix.lock` has stale entries (axon, axon_onnx, protox, erlex, dll_loader_helper, useful, ...). Run `mix deps.unlock --unused`.
- [ ] **REPO-03 (P3)** Add a toolchain file (`.tool-versions` or `mise.toml`).
- [ ] **REPO-04 (P3)** Tags exist only for 0.1.8 to 0.1.10, and there are no GitHub releases.
- [ ] **REPO-05 (P2) Maintainer bandwidth.**
  - There is one maintainer: `@mortont` (Travis Morton), the only Hex owner (`travismorton`).
  - Last merge 2026-02-10; last maintainer comment 2026-02-24.
  - No responses on PRs #49 to #52 or issue #27 since.
  - About 2.7k Hex downloads a week.
  - Consider adding maintainers or a second Hex owner.

---

## 4. GitHub issues (25 total, reviewed 2026-10-29)

| # | State | Title | Status on `main` today | Action |
| --- | --- | --- | --- | --- |
| 45 | open | Upgrade to ort 2.0.0-rc.10 | Blackwell GPUs fail on the ONNX Runtime 1.19 CUDA build. PR #46 (reject) and PR #47 (rc.11) are stalled waiting for `ort` 2.0. | Section 3.2; go to rc.15 now |
| 44 | open | Memory leak | No leak found in Ortex or `ort` rc.8 code paths; no unbounded growth reproduced. Likely GC-invisible native memory or arena growth. Reporter never sent a repro. | PERF-08; ask for a repro (Windows, YOLO, varying shapes) |
| 41 | open | Cross-compiling for Nerves | A community recipe works for aarch64. Blockers are listed in BUILD-04. | BUILD-04, ORT-14, BUILD-01 |
| 32 | open | Missing contrib operators | `MatMulNBits` and other `com.microsoft` ops are in the 1.19.2 binary. Probably fixed in 0.1.10; the reporter never re-tested. | TEST-03, then close |
| 29 | open | Session from memory | API available since rc.8. | ONNX-08 |
| 27 | open | Rustler precompiled | PR #49 open since 2026-04-26; no maintainer reply. | BUILD-02 |
| 43 | closed | Docker Debian compile | `cargo` was not on PATH. Rust is still required; rc.10+ will also need OpenSSL and rc.14+ Rust 1.92. | BUILD-02, ORT-11, DOC-05 |
| 42 | closed | BEAM crash loading a model | Workaround only (lower optimisation level). The stack overflow on a dirty IO thread is still live. | SAFE-05 |
| 37 | closed | silero VAD output order | Fixed in 0.1.10; no regression test. | TEST-03 |
| 36 | closed | Upgrade to 1.16 | ai.onnx.ml opset 4 works on 1.19; opset drift continues. | ONNX-01 |
| 34 | closed | Missing onnxruntime on Mac | Fixed in 0.1.10. Intel Mac support ends with the next upgrade. | ORT-10 |
| 33 | closed | More flexible rustler dependency | Done via a private-API version switch. | COMPAT-07 |
| 30 | closed | BERT example | Done (`examples/distilbert`). | none |
| 26 | closed | Unable to use ortex from main | Fixed in 0.1.10. The Windows DLL glob is still wrong. | BUILD-01, TEST-03 |
| 24 | closed | Update ort to 1.16.3 | Superseded. | Section 3.2 |
| 23 | closed | Implement argmax | Closed without implementation. | ARCH-11 |
| 20 | closed | Update to Nx ~> 0.6 | Done in 0.1.8. The same problem is back with Nx 1.0. | COMPAT-01 |
| 19 | closed | Bump Nx version | Duplicate of #20. | none |
| 17 | closed | Ort execution providers | The "fix" (shipping `config/config.exs`) has no effect for dependents; providers fall back silently. | BUG-03, BUILD-06, DOC-01 |
| 16 | closed | Error loading onnx model | Opset and export issue on the user side. | DOC-07 |
| 14 | closed | Rustler 0.29 for OTP 26 | Done; the crate is now 10 minor versions behind. | COMPAT-02 |
| 11 | closed | reshape, squeeze, concatenate | Done; full copies on normal schedulers. | BUG-10 |
| 8 | closed | slice | Done; tensor start indices unsupported. | BUG-06, BUG-10 |
| 6 | closed | Docs tidy up | Done. | none |
| 1 | closed | Building on macOS M2 | Fixed (per-OS features, rpath). | none |

Maintainer statements worth knowing:

- Precompiled NIFs and Ortex-managed ONNX Runtime binaries are "on the roadmap" (#27, #34, #17).
- From-memory loading was waiting on `ort` 2.0 (#29).
- They "would love" Nerves on rpi4 and Jetson (#41).
- They are waiting for `ort` 2.0 stable and have considered binding libonnxruntime directly (#45).
- They asked for PR #47 to be split (Feb 2026).

---

## 5. Pull requests (reviewed statically 2026-10-29)

| PR | Author | State and CI | What it does | Verdict |
| --- | --- | --- | --- | --- |
| #52 | omondiedward | open; CI never ran (fork approval pending) | Typed options: `:intra_op_num_threads`, `:inter_op_num_threads`, `:execution_mode`, spinning flags. Validated by a new `Ortex.SessionOptions`, applied through a second NIF `init_with_options/4`. | **Base for the consolidated options PR**, with changes: one NIF, keyword `load(path, opts)`, absorb #50, fix formatting (one line over 98 columns), drop `unless`, trim README, rebase on #47, credit #50 and #51. |
| #51 | pshoukry | open; CI never ran ("workflow file issue") | `intra_threads` option validated with `Keyword.validate!`; NIF `init/4`. | **Close in favour of the consolidated PR.** Keep its validation, tests and benchmark rationale. Its docs claim ONNX Runtime honours `ORT_INTRA_OP_NUM_THREADS`, which is only true from 1.29. |
| #50 | mattnenterprise | open; Linux CI failed (CDN 404, unrelated), macOS green | Raw ONNX Runtime config entries as a 4th positional argument; NIF `init/4`. | **Port as `:config_entries`**, applied before the typed options. Reject `nil` and malformed entries. Link the header for the bundled ONNX Runtime version (many keys only exist in 1.28+). |
| #49 | kipcole9 | open; CI green incl. 10-cell release matrix | `rustler_precompiled` for 5 targets x NIF 2.16/2.17; `ORTEX_BUILD` forces a source build; release workflow; `RELEASE.md`; `CHANGELOG.md`. | **Merge after changes, ideally after the `ort` upgrade.** Changes: generate the checksum file in the release flow (it ships as `%{}`); decide provider features per artifact (no CoreML/DirectML today); add GPU `variants` or document source builds; lower the glibc baseline; drop `x86_64-apple-darwin` for new ONNX Runtime; NIF 2.15 is enough; require `rustler_precompiled >= 0.10.1`; fix the OTP-range comment; rebase on #47 (5 conflicting files). |
| #47 | serpent213 | open; mergeable; CI green | `ort` rc.11 (ONNX Runtime 1.23), Rustler 0.37 with `resource_impl`, safe `from_binary`, panics turned into errors, input count check, `to_binary` limit, unknown providers rejected, multi-partition serving, concat test fix, CPU default on Linux, env switches, version 0.2.0-rc.1. | **Split it as the maintainer asked; fix before merging:** map level `3` to `All` (ORT-06); make `show_session` lock-free (ORT-12); avoid serialising inference (ORT-03); move `ORTEX_FEATURES` out of `config/config.exs` (BUILD-06); simplify `copy_ort_libs` (it globs the consumer project and `target/**`, has dead `cond` branches, and still misses Windows and triple directories); widen Nx to include 1.0; use plain `use Rustler`; then go to rc.15 (about four import fixes including `ort::tensor` to `ort::value`, provider `#[cfg]` gating, the one-per-process environment, `PreferCPU` for CPU-only sessions, `ORT_LIB_PATH`). |
| #46 | m0rt3nlund | open; conflicts; no CI | `ort` rc.10, Rustler 0.37, Nx 0.10, plus app-specific YOLO and image code. | **Close as superseded by #47.** It also corrupts integer outputs: unsigned outputs are clamped to `u8` and signed ones remapped (`tensor.rs:295-367`), and dtypes are reported as `{:u, 8}`. Also: CRLF rewrites of whole files, hard-enabled CUDA, `tracing_subscriber::fmt::init()` in `on_load`. Keep only the idea of an opt-in strict provider flag. |
| #28 | michaelbearne | closed 2024-11-19 | `ort` rc.2. | Superseded; nothing to port. |

**Textual conflicts** (`git merge-tree`):

- Each open PR merges cleanly on `main` except #46.
- #47 conflicts with #49 in 5 files.
- #50, #51 and #52 conflict with each other in the `load` path.
- #47 conflicts with #50 or #51 in one hunk of `lib.rs`.

**Suggested merge order:**

1. Close #46.
2. Phase 0 and phase 1 changes, including #47 part A.
3. #47 part B, retargeted to rc.15.
4. The consolidated options PR, then close #50, #51 and #52.
5. #49, rebased, with a version bump (the NIF arity changes).

---

## 6. Verification log

| What | Result |
| --- | --- |
| `main` on Elixir 1.20.4 / OTP 29.1.2 / Rust 1.99.1 (macOS arm64) | compile OK (dependency warnings only), format OK, 117 tests pass, 2 excluded |
| Fresh consumer app depending on `main` | resolves Nx 0.13.1 + Rustler 0.38.0; compiles; runs `tinymodel.onnx` |
| Dependency bump probe (Nx 1.0.0, Hex Rustler 0.38, ex_doc 0.40.4, crate unchanged) | compile, format, docs OK; 117 tests pass |
| Crate Rustler 0.39 probe | three layers of compile errors, fixed mechanically; full stack passes 117 tests on OTP 29 |
| Behaviour probe (`docs/testing-and-ci.md` section 4) | BUG-01 to BUG-06, BUG-14, BUG-15, SAFE-02 reproduced |
| Memory probe | bounded growth; no leak reproduced (PERF-08) |
| Rust 1.100.0-beta.7 (isolated toolchain) | static CPU build of `main` OK; dynamic ONNX Runtime build fails in `ort-sys` rc.8 `copy-dylibs` (BUILD-10); same build OK on 1.99.1 |
| `ort` source checks | rc.8 `Session` is `Send + Sync` with `run(&self)`; rc.15 `run(&mut self)`; `Level3` remapped; rc.8 `copy-dylibs` path logic |
| `ort` rc.15 compile probe (clone of `main`, only versions changed) | 41, then 43, 2 and 1 errors in successive waves; about 172 changed lines to a clean `cargo check` (also with `coreml` and `cuda,tensorrt` features) |
| `ort` rc.15 runtime checks | second `ort::init().build()` in one process fails; static environment + leaked sessions exit cleanly (3/3); unsatisfiable provider features still build a NIF on macOS that fails at load |
| PR builds | not executed (fork code); static review only |
