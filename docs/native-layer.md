# Native layer (Rust NIF crate)

Snapshot: `main` at `55dc3de`. Crate: `native/ortex` (`crate-type = ["cdylib"]`, `edition = "2018"`).
How the crate is compiled and linked is in [`build-and-packaging.md`](build-and-packaging.md).

## 1. Dependencies (as pinned on `main`)

| Crate | Pin (Cargo.toml) | Locked | Used for | Notes |
| --- | --- | --- | --- | --- |
| `rustler` | `0.29.0` | `0.29.1` | NIF macros, resources, term codecs | `rustler::resource!` triggers 2 `non_local_definitions` warnings on Rust 1.99 |
| `ort` | `2.0.0-rc.8` | `2.0.0-rc.8` | ONNX Runtime bindings | default features: `ndarray`, `half`, `download-binaries`, `copy-dylibs`; bundles ONNX Runtime 1.19.2 |
| `ndarray` | `0.16.1` | `0.16.1` | owned n-d arrays inside `OrtexTensor` | must be the same major version `ort` uses, or `ort`'s ndarray conversions will not accept the arrays |
| `half` | `2.2.1` | `2.3.1` | `f16`, `bf16` element types | |
| `tracing-subscriber` | `0.3` | | nothing (only a commented-out call in `model.rs`) | unused, remove |
| `num-traits` | `0.2.15` | | nothing | unused, remove |
| `rustls` | `0.22.4` | | nothing directly | pinned to force a patched transitive version (dependabot alert on `ort-sys`'s downloader); remove when `ort` is upgraded |

Cargo features forwarded to `ort`: `directml`, `coreml`, `cuda`, `tensorrt`. Which ones are enabled
comes from `config :ortex, Ortex.Native, features: [...]` (see build doc).

## 2. NIF table

All NIFs are registered in `rustler::init!("Elixir.Ortex.Native", [...], load = ...)` in `lib.rs` and
stubbed in `lib/ortex/native.ex`.

| Elixir stub | Rust fn | Arguments (decoded as) | Returns | Scheduler | Data movement |
| --- | --- | --- | --- | --- | --- |
| `init(path, eps, opt)` | `lib.rs::init` | `String`, `Vec<Atom>`, `i32` | `ResourceArc<OrtexModel>` or `{:error, String}` | DirtyIo | reads model file |
| `run(model, inputs)` | `lib.rs::run` | `ResourceArc<OrtexModel>`, `Vec<ResourceArc<OrtexTensor>>` | `[{tensor_ref, [dims], dtype_atom, bits}]` or `{:error, String}` | DirtyIo | clones every input, copies every output |
| `from_binary(bin, shape, type)` | `lib.rs::from_binary` | `Binary`, shape tuple, `{atom, bits}` | `ResourceArc<OrtexTensor>` | DirtyCpu | copies the binary (unsafe, see section 5) |
| `to_binary(ref, bits, limit)` | `lib.rs::to_binary` | `ResourceArc<OrtexTensor>`, `usize`, `usize` | resource binary aliasing the tensor | DirtyCpu | zero-copy; `bits` and `limit` are **ignored** |
| `show_session(model)` | `lib.rs::show_session` | `ResourceArc<OrtexModel>` | `{[{name, debug_string, dims}], [...]}` | normal | metadata only |
| `slice(ref, starts, lengths, strides)` | `lib.rs::slice` | `Vec<isize>` x3 | `ResourceArc<OrtexTensor>` | normal (should be dirty) | copies the slice |
| `reshape(ref, shape)` | `lib.rs::reshape` | `Vec<usize>` | `ResourceArc<OrtexTensor>` or `{:error, String}` | normal (should be dirty) | clones the whole array |
| `concatenate(refs, type, axis)` | `lib.rs::concatenate` | `Vec<ResourceArc>`, `{atom, bits}`, `i32` | `ResourceArc<OrtexTensor>` | normal (should be dirty) | copies twice (concatenate, then re-layout) |

Error convention: `rustler::Error::Term(Box::new(string))` encodes as `{:error, string}` (a return
value, not an exception). Decoding failures return `badarg`, which Elixir sees as `ArgumentError`.

## 3. Types

### `OrtexModel` (`model.rs`)

```rust
pub struct OrtexModel { pub session: ort::Session }
unsafe impl Sync for OrtexModel {}
```

Registered as a resource in `lib.rs`'s `load` callback (`rustler::resource!(OrtexModel, env)`).
Shared by reference across BEAM processes; `run` uses `&Session` concurrently (valid for
`ort 2.0.0-rc.8` where `Session::run(&self, ...)`).

### `OrtexTensor` (`tensor.rs`)

An enum with one variant per element type, each holding an owned `ndarray::Array<T, IxDyn>`:

| Variant | Rust element | Nx type | ONNX element type |
| --- | --- | --- | --- |
| `s8`, `s16`, `s32`, `s64` | `i8`..`i64` | `{:s, 8..64}` | `INT8`..`INT64` |
| `u8`, `u16`, `u32`, `u64` | `u8`..`u64` | `{:u, 8..64}` | `UINT8`..`UINT64` |
| `f16` | `half::f16` | `{:f, 16}` | `FLOAT16` |
| `bf16` | `half::bf16` | `{:bf, 16}` | `BFLOAT16` |
| `f32`, `f64` | `f32`, `f64` | `{:f, 32}`, `{:f, 64}` | `FLOAT`, `DOUBLE` |
| `bool` | `bool` | (none; internal only) | `BOOL` |

Methods: `shape`, `reshape`, `dtype` (returns `(atom, bits)`), `to_bytes`, `slice`, `to_bool`; free
function `concatenate` (macro-expanded per dtype). `Clone` is implemented by hand.

Conversions:

* `TryFrom<&ort::Value> for OrtexTensor`: matches `ValueType::Tensor { ty, .. }`, extracts with
  `try_extract_tensor::<T>()?.into_owned()` (a copy), maps `Bool` to `u8`, `todo!()` on `String`,
  `panic!` on non-tensor values.
* `TryFrom<&OrtexTensor> for ort::SessionInputValue`: `arr.clone().try_into::<DynValue>()` (a copy).

## 4. Other helpers (`utils.rs`, `constants.rs`)

* `from_binary`: matches the `(type_string, bits)` pair and builds the variant from an unsafe view
  over the BEAM binary followed by `.to_owned()`.
* `to_binary`: `reference.make_binary(env, |t| t.to_bytes())`.
* `map_eps`: atom string to `ort::*ExecutionProvider::default().build()` for `cpu`, `cuda`,
  `tensorrt`, `acl`, `onednn`, `coreml`, `directml`, `rocm`; **anything else silently maps to CPU**.
  The `dnnl` atom exists in `constants.rs` but is not mapped.
* `map_opt_level`: `1|2|3` map to `Level1|2|3`; any other integer disables graph optimisation.
* `is_bool_input`: true when the model input's element type (or a map/sequence/optional's inner
  type) is `Bool`.

## 5. Unsafe code inventory

| Location | What | Risk | Fix |
| --- | --- | --- | --- |
| `model.rs` `unsafe impl Sync for OrtexModel` | Declares the session shareable across threads | Redundant with rc.8 (the session is already shareable); becomes **unsound** with `ort` versions where `run` needs `&mut self` | Remove; on newer `ort`, wrap state in a lock or a session pool (see `upgrading.md`) |
| `utils.rs` `initialize_from_raw_ptr` | `ArrayViewMut::from_shape_ptr(shape, bin.as_ptr() as *mut T)` over BEAM binary memory | (1) no check that `bin.len() == product(shape) * size_of::<T>()`: a short binary reads out of bounds; (2) BEAM binaries (especially sub-binaries) are not guaranteed to be aligned for `T`; (3) creates a *mutable* view of shared immutable memory | Validate the length, copy bytes into a `Vec<T>` (`copy_nonoverlapping` or `bytemuck::pod_collect_to_vec`), build with `Array::from_shape_vec`. PR #47 does this. |
| `tensor.rs` `get_bytes` | `std::mem::zeroed()` of a generic element just to learn its size; `slice::from_raw_parts(array.as_ptr(), len * size)` | Correct only for standard-layout arrays (invariant), needless `zeroed` | Use `size_of::<A>()` and `array.as_slice_memory_order()` (returns `None` if not contiguous) or `bytemuck::cast_slice` |

Every `unsafe` block should carry a `// SAFETY:` comment stating the invariant it relies on.

## 6. Panic inventory

Rustler turns a panic into an `ErlangError` with reason `:nif_panicked`; nothing crashes the VM,
but the caller gets no useful message.

| Location | Trigger | Reachable from Elixir? |
| --- | --- | --- |
| `utils.rs::from_binary` `unimplemented!()` | Nx type without a variant: `{:c, 64}`, `{:c, 128}`, and the newer Nx sub-byte and 8-bit float types | **Yes** (verified with `{:c, 64}`) |
| `tensor.rs::to_bool` `panic!` | A model bool input receives a non-`u8` tensor (e.g. `Nx.tensor([1, 0])`, which is `s64`) or a `u8` value other than 0/1 | **Yes** |
| `tensor.rs` `TryFrom<&Value>` `panic!` | Model output is a sequence, map or optional (e.g. scikit-learn classifiers with ZipMap) | **Yes** |
| `tensor.rs` `TryFrom<&Value>` `todo!` | Model output is a string tensor | **Yes** |
| `tensor.rs` `shape/reshape/dtype/to_bytes/slice` `panic!` | Called on the internal `bool` variant | No (bool tensors never reach Elixir) |
| `tensor.rs` `concatenate!` `.unwrap()` | Shapes that cannot be concatenated | Not via Nx (Nx validates shapes first) |
| `tensor.rs::concatenate` `unimplemented!()` | Unsupported dtype | Not via Nx |
| `model.rs::run` `.expect(..)` | Output name missing from `SessionOutputs` | Not normally |
| `tensor.rs::slice_array` index / `slice_each_axis` bounds | Start/length vectors that do not match the rank or exceed a dimension | Not via Nx with integer indices |

## 7. `ort` API surface used (rc.8)

This is the list to re-check on every `ort` upgrade. The rc.15 replacements (checked in the rc.15
source and by a compile probe) are below; [`upgrading.md`](upgrading.md) explains the changes by
release candidate.

| rc.8 (today) | rc.15 |
| --- | --- |
| `use ort::{Session, SessionInputValue}` | `ort::session::{Session, SessionInputValue, SessionOutputs, RunOptions}` |
| `ort::GraphOptimizationLevel::Level3` | `ort::session::builder::GraphOptimizationLevel::All` (`Level3` now means "layout") |
| `ort::{Value, DynValue, ValueType, TensorElementType}` | `ort::value::{Value, DynValue, ValueType, TensorElementType, Tensor, TensorRef, Outlet}` |
| `ort::CUDAExecutionProvider::default().build()` | `ort::ep::CUDA::default().build()`, only with the `cuda` feature (same pattern for every provider) |
| `ort::ExecutionProviderDispatch` | `ort::ep::ExecutionProviderDispatch` |
| `Session::builder()?` | `Session::builder(&env)?` with the process-wide `Environment` from `ort::init().build()?` |
| `session.inputs` / `session.outputs` fields; `.input_type` | `session.inputs()` / `session.outputs()` returning `&[Outlet]`; `outlet.name()`, `outlet.dtype()` |
| `ValueType::tensor_dimensions() -> Option<&Vec<i64>>` | `ValueType::tensor_shape() -> Option<&Shape>` (`Shape` derefs to `[i64]`) |
| `ValueType::Tensor { ty, dimensions }` | `ValueType::Tensor { ty, shape, dimension_symbols }` |
| `session.run(&inputs)` with `&self` | `session.run(&inputs)` with `&mut self` (lock or pool) |
| `value.try_extract_tensor::<T>()?.into_owned()` | `value.try_extract_array::<T>()?.to_owned()` |
| `array.clone().try_into::<DynValue>()` | `Tensor::from_array(owned)?` or zero-copy `TensorRef::from_array_view(view)?` |
| exhaustive `match` on `TensorElementType` | wildcard arm required (`#[non_exhaustive]`; new Complex, Float8, Int4/Uint4, Int2/Uint2, Float4, Float8E8M0, Undefined variants) |
| `f16`/`bf16` extraction works by default | requires `features = ["half"]` |

* Crate-root imports: `ort::{Session, Error, ExecutionProviderDispatch, GraphOptimizationLevel, Value, DynValue, SessionInputValue, TensorElementType, ValueType}`.
* Session building: `Session::builder()?`, `.with_optimization_level(..)?`, `.with_execution_providers(Vec<ExecutionProviderDispatch>)?`, `.commit_from_file(path)?`. No environment is created explicitly.
* Execution providers: `ort::{CPU,CUDA,TensorRT,ACL,OneDNN,CoreML,DirectML,ROCm}ExecutionProvider::default().build()`.
* Introspection: `session.inputs` / `session.outputs` (public fields), `.name`, `.input_type` / `.output_type`, `ValueType::tensor_dimensions()`, `Debug` formatting of `ValueType`, matching `ValueType::{Tensor, Map, Sequence, Optional}`.
* Inference: `session.run(&[SessionInputValue])`, `SessionOutputs::get(name)`, `Value::dtype()`, `Value::try_extract_tensor::<T>()`, `Array<T, IxDyn>: TryInto<DynValue>`.
* Element types matched exhaustively: `TensorElementType::{Bfloat16, Float16, Float32, Float64, Uint8, Uint16, Uint32, Uint64, Int8, Int16, Int32, Int64, String, Bool}`.

## 8. Rust warnings on the current toolchain

Building `main` with Rust 1.99.1 produces 15 warnings and no errors:

* 12x `mismatched_lifetime_syntaxes` from the `concatenate!` macro's `filter` helper (`tensor.rs`).
* 1x the same lint on `initialize_from_raw_ptr` (`utils.rs`).
* 2x `non_local_definitions` from `rustler::resource!` in `lib.rs` (fixed by Rustler 0.34+ `#[rustler::resource_impl]`).

There is no `cargo fmt` or `cargo clippy` step in CI.
