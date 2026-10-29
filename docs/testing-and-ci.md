# Testing and CI

## 1. Running the tests

```sh
mix deps.get
MIX_ENV=test mix compile --warnings-as-errors   # also builds the Rust crate and downloads ONNX Runtime
mix format --check-formatted
mix test                                        # resnet50 tests are excluded unless models/resnet50.onnx exists
mix test --only resnet50                        # after generating the model, see below
```

* The first build needs network access: `ort`'s build script downloads ONNX Runtime.
* **Editing Rust code does not trigger a rebuild.** `lib/ortex/native.ex` builds the crate by hand
  and registers no external resources, so `mix compile` skips it (verified by touching
  `native/ortex/src/utils.rs`). Use `mix compile --force` or `touch lib/ortex/native.ex`.
* Use `MIX_ENV=test` when you only need the library and its tests: the `:dev`-only deps (`exla`,
  `torchx`, `tokenizers`) download large native artifacts.
* Rust-only checks: `cd native/ortex && cargo fmt --check && cargo clippy`. Running cargo directly
  builds into `native/ortex/target` (ignored by git) and downloads ONNX Runtime again.

## 2. Test suite layout

| File | Covers | Notes |
| --- | --- | --- |
| `test/test_helper.exs` | Excludes `:resnet50` tests unless `models/resnet50.onnx` exists | |
| `test/ortex_test.exs` | `doctest Ortex`; resnet50 inference and `Nx.Serving` (tagged); `Nx.Serving` with `tinymodel` (batch of 3, two inputs of different dtypes, three outputs) | The `Ortex.load/3` doctests only check that loading does not raise. |
| `test/dtype/dtype_test.exs` | Binary round trip BinaryBackend → Ortex → BinaryBackend for all 12 supported dtypes plus a scalar | |
| `test/shape/concat_test.exs` | `Nx.concatenate` on Ortex tensors (1-D, 3-D, axes 1 and 2, three inputs, mixed types raise) | **Bug:** the type table maps every name except `s8` to `{:s, 16}`, so 10 of the 12 dtype variants are never exercised. PR #47 fixes the table. |
| `test/shape/reshape_test.exs` | `Nx.reshape` including `:auto` and a tensor-as-shape | Has a `TODO` saying it does not truly validate the Ortex side. |
| `test/shape/slice_test.exs` | `Nx.slice` 1-D and 2-D, with and without strides, `f32` and `u8` | Integer start indices only. |
| `test/shape/squeeze_test.exs` | `Nx.squeeze` with and without axes, with names | |

### Test models

| Model | Where | How to (re)create |
| --- | --- | --- |
| `models/tinymodel.onnx` | committed (8.7 KB) | `python python/multi_input.py` (PyTorch, opset 19). The script writes `tinymodel.onnx` into the **current directory**; move it to `models/`. Inputs: `x` int32 `[batch, 100]`, `y` float32 `[batch, 100]`; three float32 `[batch, 10]` outputs. |
| `models/resnet50.onnx` | not committed (~100 MB) | `python python/export_resnet.py` (torchvision, opset 19). Also writes into the current directory. The script uses the deprecated `pretrained=True` torchvision argument; PR #47 modernises it. |

## 3. Verified results (2026-10-29)

`main` (`55dc3de`) was cloned into a scratch directory and built on macOS arm64 with
Elixir 1.20.4, Erlang/OTP 29.1.2 (NIF 2.18) and Rust 1.99.1, `MIX_ENV=test`:

| Step | Result |
| --- | --- |
| `mix deps.get` | OK (locked: nx 0.6.2, rustler 0.29.1, complex 0.5.0, telemetry 1.2.1, jason 1.4.1, toml 0.7.0) |
| `mix compile --warnings-as-errors` | OK in about 16 s. Hundreds of warnings, **all from dependencies** (nx 0.6.2: pin-operator and charlist deprecations; jason, toml, rustler 0.29.1: charlists, `~~~`, typing). None from Ortex's own Elixir code. |
| Rust crate | OK, 15 warnings (see `native-layer.md` section 8). ONNX Runtime 1.19.2 is linked statically into a 21 MB `libortex.so` (with CoreML). No `libonnxruntime` side library is produced on macOS. |
| `mix format --check-formatted` | OK |
| `mix test` | **117 passed** (2 doctests, 115 tests), 2 excluded (`:resnet50`) |

Conclusion: the current code already runs on the newest Elixir/OTP; what blocks real-world use is
dependency constraints and drift (for example `{:nx, "~> 0.6"}` cannot resolve together with Nx 1.0).
See [`upgrading.md`](upgrading.md).

Further runs on the same machine:

| Run | Result |
| --- | --- |
| Fresh consumer app depending on `main` (dependency resolution without the lock) | Resolves Nx 0.13.1 and Hex Rustler 0.38.0; compiles; `Ortex.run` on `tinymodel.onnx` works. Rustler 0.38 builds in `native/ortex/target/release` and installs `priv/native/ortex.so`. |
| `mix.exs` bumped to `nx ~> 1.0`, `rustler ~> 0.38`, `ex_doc ~> 0.40`, `elixir ~> 1.17` (crate unchanged) | compile with `--warnings-as-errors`, `format`, `mix docs` OK; 117 tests pass |
| Rust crate bumped to `rustler 0.39` | Three layers of compile errors (see `upgrading.md`); after mechanical fixes, 117 tests pass on OTP 29 |
| Rust **1.100.0-beta.7**, static CPU build | OK (new cargo lint: `rustls` and `tracing-subscriber` are unused dependencies) |
| Rust **1.100.0-beta.7**, dynamically linked ONNX Runtime (`ORT_LIB_LOCATION`, `copy-dylibs` path) | **Fails**: `ort-sys` rc.8 build script panics (`build.rs:111`, "No such file or directory") because the new build-dir layout moves `OUT_DIR`. Same build passes on 1.99.1. Affects CUDA/TensorRT builds, including this repo's Linux CI. |
| `mix test --warnings-as-errors` | Fails after a fully passing run because `test/test_helper.exs` uses `IO.warn` when `models/resnet50.onnx` is missing. |

## 4. Behaviour probe results

A throwaway script exercised edge cases against the same build. These are the facts the docs
and `TODO.md` rely on; each should become a regression test.

| Case | Observed |
| --- | --- |
| `inspect(model)` | Type strings are multi-line `ort` `Debug` output; dynamic dims are `-1` (docs say `nil`). |
| `Ortex.run` with too few inputs | `RuntimeError` from a Gemm kernel ("Non-zero status code returned while running Gemm node ...") |
| `Ortex.run` with an extra input | Succeeds; the extra input is silently ignored. |
| Wrong dtype / wrong shape | `RuntimeError` with clear ONNX Runtime messages. |
| `Ortex.load` missing file | `RuntimeError, "File at `./models/nope.onnx` does not exist"` |
| `Ortex.load(path, [:bogus])` / `[:cuda]` on macOS | Loads on CPU, no warning. |
| `Ortex.load(path, ["cpu"])` | `ArgumentError, "argument error"` |
| `Nx.backend_copy(t, Nx.BinaryBackend)` | `RuntimeError, "operation backend_copy is not yet supported on Ortex.Backend."` |
| `Nx.backend_deallocate(t)` | `RuntimeError, "operation backend_deallocate is not yet supported ..."` |
| `Nx.to_binary(t, limit: 2)` on 6 x f32 | 24 bytes (limit ignored) |
| `Nx.tensor([1.0], type: :c64) \|> Nx.backend_transfer(Ortex.Backend)` | `ErlangError :nif_panicked` (`not implemented` at `utils.rs:78`) |
| `Nx.slice(t, [Nx.tensor(1), 0], [1, 3])` | `ArgumentError, "argument error"` |
| 32 concurrent tasks x 20 `Ortex.run` on one model | OK (runs in parallel on rc.8) |
| Process-based `Nx.Serving.batched_run` | OK |
| `Nx.Serving` with `partitions: true` and 2 partitions (Nx 1.0) | `FunctionClauseError` in `Ortex.Serving.init/3` |
| `Nx.stack/2` on Ortex tensors (Nx 1.0; any Nx >= 0.8) | `RuntimeError, "operation stack is not yet supported on Ortex.Backend."` |
| `Nx.concatenate([ortex_tensor, binary_tensor])` (Nx 1.0) | `MatchError` in `Ortex.Backend.concatenate/3` |
| `{:s, 4}`, `{:u, 2}`, `{:f, 8}`, `{:f8_e4m3fn, 8}` tensors to Ortex (Nx 1.0) | `ErlangError :nif_panicked` |

Memory probe (issue #44): a GenServer kept the latest outputs of `tinymodel.onnx` in its state while
running 600 inferences with 20,000-row batches (about 8 MB of input and 2.4 MB of output per run).

| Variant | RSS |
| --- | --- |
| Fixed batch size | 139 MB at start, about 188 MB after 100 runs, flat until 600 runs, 143 MB after `:erlang.garbage_collect/1` on the holder |
| Random batch size (1 to 20,000) | 109 MB at start, peak 287 MB after 100 runs, drifting down to 201 MB at 600 runs, 183 MB after GC |

The BEAM's own memory stayed near 75 MB throughout. No unbounded growth was reproduced on macOS with
rc.8: native tensor memory waits for garbage collection, and ONNX Runtime's arena keeps some memory.

## 5. Gaps in the test suite

* No tests for error paths (bad path, wrong input count, wrong dtype/shape, unknown provider).
* No test model with a `bool` input or output (bool support from PR #39 is untested), none with
  string, sequence or map outputs, none with multiple partitions in `Nx.Serving`.
* No concurrency, memory-growth (issue #44) or large-tensor tests.
* No tests for `inspect(model)` format, `backend_copy`, `backend_deallocate`, `to_binary` limits.
* No Rust unit tests (`cargo test`), no `cargo clippy`/`cargo fmt` in CI.
* resnet50 tests never run in CI (no model), so real-model inference is untested in CI.

## 6. CI (`.github/workflows/ci.yml`)

Two jobs on push to `main` and on pull requests, both `MIX_ENV=test`, steps
`deps.get` → `compile --warnings-as-errors` → `format --check-formatted` → `test`:

| Job | Runner | Toolchain |
| --- | --- | --- |
| `linux` | `ubuntu-latest` | Elixir 1.17.3 / OTP 27.1.2 via `erlef/setup-beam@v1` |
| `macos` | `macos-latest` | Homebrew `erlang@27` and whatever `elixir` Homebrew ships |

Gaps and problems:

* `actions/checkout@v2` is long deprecated (runs on an old Node runtime); use the current major.
* Single version pair; nothing tests the declared minimum (`elixir: "~> 1.14"`) or the newest
  Elixir/OTP. The macOS job's Elixir version floats with Homebrew.
* No caching of `deps`, `_build`, `~/.cargo` or the ONNX Runtime download.
* On Linux the repo's `config/config.exs` enables the `cuda` and `tensorrt` features, so CI
  downloads the CUDA build of ONNX Runtime and runs the tests on its CPU provider. That is a
  different binary from what a dependent gets by default (CPU-only, statically linked). It also
  makes Linux CI depend on pyke's CDN for a 111 MB download instead of the 18 MB CPU package (a
  transient 404 failed PR #50) and
  will break when the runners move to Rust 1.100 (see the table in section 3).
* No Windows job, no Rust lint job, no docs build (`mix docs`), no Hex package build check.
* No release workflow; releases are cut by hand. PR #49 adds a precompiled-NIF release workflow.
