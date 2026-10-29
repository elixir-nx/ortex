# Ortex maintainer and agent docs

These documents explain how Ortex is built and how it works inside, for people and AI agents who
change it. User-facing documentation stays in the top-level `README.md` and the module docs.

Snapshot: 2026-10-29, `main` at `55dc3de` (Hex `0.1.10` + PR #48). Each document says which facts were
verified by running code and which come from reading it.

## Reading order

1. [`../AGENTS.md`](../AGENTS.md): setup, commands, gotchas, design rules, recipes.
2. [`architecture.md`](architecture.md): layers, responsibilities, the load/run/transfer/serving
   flows, memory, concurrency, error model, invariants, design debts.
3. [`elixir-layer.md`](elixir-layer.md): public API behaviour, `Nx.Backend` coverage, dtype matrix,
   configuration keys.
4. [`native-layer.md`](native-layer.md): the Rust crate: NIF table, types, conversions, unsafe and
   panic inventories, the `ort` API surface in use.
5. [`build-and-packaging.md`](build-and-packaging.md): how `mix compile` builds the NIF, how ONNX
   Runtime is downloaded and linked, platforms, environment variables, Hex package, precompiled
   NIFs, cross-compiling.
6. [`execution-providers.md`](execution-providers.md): CPU/CUDA/TensorRT/CoreML/DirectML and friends,
   how selection and fallback work, what changes upstream.
7. [`testing-and-ci.md`](testing-and-ci.md): running tests, test models, verified results,
   behaviour probe results, CI.
8. [`upgrading.md`](upgrading.md): moving to the latest Elixir/Nx/Rustler and to the latest `ort` and
   ONNX Runtime, with checklists and the order to do it in.
9. [`../TODO.md`](../TODO.md): every finding from the 2026-10 review, GitHub issue and PR triage,
   and a roadmap.

## Version snapshot

| Component | Ortex `main` | Latest upstream (2026-10-29) |
| --- | --- | --- |
| Elixir / Erlang/OTP | declares `~> 1.14`; CI 1.17.3 / 27.1.2 | 1.20.4 / 29.1.2 (verified working) |
| Nx | `~> 0.6` (lock 0.6.2) | 1.0.0 |
| Rustler (Hex / crate) | `~> 0.27` (lock 0.29.1) / `0.29` | 0.38.0 / 0.39.0 |
| rustler_precompiled | not used | 0.10.1 |
| `ort` crate | `2.0.0-rc.8` | `2.0.0-rc.15` (no 2.0.0 yet) |
| ONNX Runtime | 1.19.2 (via `ort`) | 1.31.1 (`ort` rc.15 bundles 1.31.0) |
| Prebuilt ONNX Runtime platform floor | about glibc 2.32; Intel and Apple Silicon macOS | glibc 2.39 / libstdc++ 13.2, AVX2, Apple Silicon only (rc.15) |
| ONNX opset supported | 21 (ONNX 1.16.1) | 26 in ONNX Runtime 1.28 to 1.31 (ONNX 1.22); ONNX itself is at 1.23.2 / opset 28 |
| `ndarray` / `half` | 0.16.1 / 2.2.1 | 0.17.2 / 2.7.1 |
| ex_doc | pinned `0.29.4` | 0.40.4 |

## Glossary

| Term | Meaning |
| --- | --- |
| ONNX | Open Neural Network Exchange: a file format (protobuf) for ML models. Versioned by *IR version* (file format) and *opset* (operator set) versions. |
| ONNX Runtime (ORT) | Microsoft's C++ inference engine for ONNX models. |
| `ort` | The Rust crate (by pyke, `pykeio/ort`) that binds ONNX Runtime. Its 2.0 line has been in release candidates since 2024. |
| pyke CDN | Where `ort` downloads prebuilt ONNX Runtime builds during `cargo build` (`parcel.pyke.io` for rc.8, `cdn.pyke.io` later). |
| Execution provider (EP) | An ONNX Runtime backend for some hardware (CPU, CUDA, TensorRT, CoreML, DirectML, ...). |
| Session | A loaded, optimised model in ONNX Runtime, ready to run. Ortex wraps one per `%Ortex.Model{}`. |
| NIF | Native Implemented Function: a C-ABI function loaded into the BEAM. Ortex's NIFs are written in Rust with Rustler. |
| Dirty scheduler | BEAM thread pool for NIFs that run longer than about 1 ms (dirty CPU, dirty IO). |
| Resource (`ResourceArc`) | A reference-counted native object owned by the BEAM GC. `OrtexModel` and `OrtexTensor` are resources. |
| Resource binary | A BEAM binary that points into a resource's memory without copying (`ResourceArc::make_binary`). |
| `Nx.Backend` | The behaviour that stores and computes tensors for Nx. `Ortex.Backend` only stores. |
| `Nx.Serving` | Nx's batching and serving abstraction; `Ortex.Serving` implements its callbacks. |
| rustler_precompiled | Library that downloads prebuilt NIFs from GitHub releases instead of compiling Rust. |
| NIF version | ABI version of the NIF API (OTP 24-25 = 2.16, OTP 26-28 = 2.17, OTP 29 = 2.18). A library built for an older minor version loads on newer runtimes. |
