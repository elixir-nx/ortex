# Execution providers

An execution provider (EP) is the ONNX Runtime backend that runs (part of) a graph: CPU, CUDA,
TensorRT, CoreML, DirectML and so on. A session gets an ordered list of providers; ONNX Runtime
assigns each graph node to the first provider that supports it and falls back to CPU.

Snapshot: `main` at `55dc3de` (`ort 2.0.0-rc.8`, ONNX Runtime 1.19.2). Changes in newer `ort` and
ONNX Runtime releases are summarised in section 5 and detailed in [`upgrading.md`](upgrading.md).

## 1. How Ortex selects providers today

`Ortex.load(path, eps)` passes a list of atoms. `utils::map_eps` turns each into
`ort::<Name>ExecutionProvider::default().build()` and the session builder registers them in order.

| Atom | `ort` type (rc.8) | Cargo feature | Prebuilt ONNX Runtime (rc.8 download) | Notes |
| --- | --- | --- | --- | --- |
| `:cpu` | `CPUExecutionProvider` | none | all six targets | Always available. |
| `:cuda` | `CUDAExecutionProvider` | `cuda` | `cu12` variant on x86_64 Linux and Windows | Needs CUDA 12 + cuDNN at run time. Dynamic library (see build doc). |
| `:tensorrt` | `TensorRTExecutionProvider` | `tensorrt` | same `cu12` variant | Also needs TensorRT libraries installed. |
| `:coreml` | `CoreMLExecutionProvider` | `coreml` | macOS builds include it | The feature is required for registration even though the binary is the same. |
| `:directml` | `DirectMLExecutionProvider` | `directml` | Windows | Feature required for registration. |
| `:acl` | `ACLExecutionProvider` | (none forwarded) | no | Arm Compute Library; needs a custom ONNX Runtime. |
| `:onednn` | `OneDNNExecutionProvider` | (none forwarded) | no | Needs a custom ONNX Runtime. (`:dnnl` exists as an atom but is not mapped.) |
| `:rocm` | `ROCmExecutionProvider` | (none forwarded) | no | Removed from ONNX Runtime in 1.23. |
| anything else | `CPUExecutionProvider` | | | **Silent fallback.** |

`ort` only registers a provider when its Cargo feature is on; otherwise registration returns an error.
`ort` treats registration errors as non-fatal by default and only logs them with `tracing::warn!`.
Ortex installs no `tracing` subscriber, so the warning is lost: **asking for an unavailable provider
silently runs on CPU** (verified: `[:cuda]` and `[:bogus]` on macOS). `ort` offers
`ExecutionProviderDispatch::error_on_failure()` to make registration failures fatal; Ortex does not use it.

## 2. Enabling a provider in an application

1. Choose features in the application's config (the Ortex repo's own `config/config.exs` is not used
   for dependents):

   ```elixir
   config :ortex, Ortex.Native, features: ["cuda", "tensorrt"]
   ```

2. Recompile Ortex (`mix deps.compile ortex --force`). With `cuda`/`tensorrt` on x86_64 Linux or
   Windows, `ort` downloads the dynamic CUDA 12 build of ONNX Runtime.
3. Make the ONNX Runtime side libraries loadable next to the NIF (`priv/native`). On `main` this
   relies on `Ortex.Util.copy_ort_libs/0`, which does not find them with Rustler >= 0.37 (the
   version new installs resolve). GPU builds also fail to compile on Rust 1.100 until Ortex moves
   to `ort` rc.14+ (see `build-and-packaging.md`).
4. Install the provider's system libraries (CUDA 12 runtime, cuDNN, TensorRT) on the machine that runs
   the code.
5. Load with the provider first and CPU last: `Ortex.load(path, [:cuda, :cpu])`.

There is no API to ask which providers are compiled in or available, and no way to pass provider
options (device id, memory limit, TensorRT engine cache, CoreML flags...). An unmerged maintainer
branch, `tm-execution-providers` (March 2024), sketches option structs such as
`%Ortex.CUDAExecutionProvider{device_id: ..., gpu_mem_limit: ...}` and
`%Ortex.TensorRTExecutionProvider{...}`; it is useful prior art for the API.

## 3. Interaction with BEAM scheduling

Inference runs on a dirty IO scheduler while ONNX Runtime uses its own intra-op thread pool (by
default one per session, sized to the machine, with busy-wait spinning enabled). Several loaded
models, or a model next to busy BEAM schedulers, oversubscribe the CPU. Useful settings once Ortex
exposes session options (PRs #50, #51, #52): `intra_op_num_threads`, `inter_op_num_threads`,
`session.intra_op.allow_spinning = "0"`, and a shared global thread pool.

## 4. GPU memory and lifetimes

Tensors cross the boundary through host memory: inputs are copied into ONNX Runtime-owned tensors
and outputs are copied out into `ndarray`s. Nothing keeps data on the GPU between calls (no
IoBinding). A model's GPU memory is released when the `%Ortex.Model{}` is garbage collected.

## 5. What changes with newer `ort` and ONNX Runtime

Upstream ONNX Runtime:

* **1.23** removed the ROCm provider (AMD now points to MIGraphX or Vitis AI) and introduced the
  plugin execution-provider API.
* **1.24** added CUDA 13 support and stopped publishing x86_64 macOS binaries (minimum macOS 14).
  CUDA 12 packages are deprecated (still published through 1.31).
* **1.28 to 1.31** also release CUDA and WebGPU as separate *plugin* providers that must match the core
  version (1.31.0 pairs with CUDA plugin 0.3.0). The in-tree CUDA provider is still the default in the
  core GPU packages. 1.31 brings CoreML into device-based provider selection and deprecates ACL; DirectML
  is in "sustained engineering". Microsoft's official binaries from 1.29 can send telemetry
  (`ORT_DISABLE_TELEMETRY=1` turns it off).

`ort` (details and sources in [`upgrading.md`](upgrading.md) section 3.3):

* **rc.11** renamed `XXXExecutionProvider` to `ort::ep::XXX` and builds pyke binaries with thread
  spinning off by default (re-enable it for servers).
* **rc.12** made `AutoDevicePolicy::MaxEfficiency` the session default; registering `ep::CPU` does not
  override it, so with ONNX Runtime 1.31 a `[:cpu]` session on macOS can run on CoreML (pykeio/ort#662).
  Use `with_auto_device(AutoDevicePolicy::PreferCPU)` for CPU-only sessions.
* **rc.13** compiles provider structs only with their Cargo feature, reports unsatisfiable feature sets
  as an unresolved-symbol link error (`lax-feature-matching` relaxes it), and ships CUDA 13 only
  (CUDA >= 13.2 and cuDNN >= 9.23; Turing or newer GPUs).
* **rc.14** adds CUDA builds for Blackwell (x64) and aarch64 (Jetson Orin/Thor, Grace) and MIGraphX on
  x64 Linux.

Prebuilt provider packages in rc.15:

| Target | Packages |
| --- | --- |
| `aarch64-apple-darwin` | `coreml`; `coreml,webgpu` |
| `x86_64-unknown-linux-gnu` | CPU; `openvino`; `webgpu`; `nvrtx`; `migraphx`; `cuda13,tensorrt,nvrtx` |
| `aarch64-unknown-linux-gnu` | CPU; `cuda13,tensorrt,nvrtx` |
| `x86_64-pc-windows-msvc` | `directml`; `webgpu`; `openvino`; `nvrtx,directml`; `cuda13,tensorrt,nvrtx,directml` |
| `aarch64-pc-windows-msvc` | `directml` |

Providers without prebuilt packages (ROCm, QNN, XNNPACK, ACL, oneDNN, CANN, Vitis, ...) need a custom
ONNX Runtime through `ORT_LIB_PATH`. In CUDA packages the core is static but the provider libraries
(`libonnxruntime_providers_{shared,cuda,tensorrt,nv_tensorrt_rtx}.so`) are loaded at run time and must
sit next to the NIF.

Consequences for Ortex:

* Drop `:rocm` (or map it to MIGraphX) and consider `:migraphx`, `:openvino`, `:webgpu`, `:nvrtx`.
* Gate each provider with `#[cfg(feature = ...)]` and return a clear error when an unavailable one is
  requested.
* `-undefined dynamic_lookup` on macOS hides `ort`'s link-time check: a bad feature set builds a NIF
  without ONNX Runtime that only fails at load. Validate feature/target combinations in Elixir and load
  the NIF in CI.
* Make "available providers" queryable (rc.14 has `EP::is_available()`), and decide whether GPU support
  ships as a precompiled `rustler_precompiled` variant or stays source-build only.
