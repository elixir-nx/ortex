# Elixir layer: public API and Nx integration

Snapshot: `main` at `55dc3de` with the locked dependencies (`nx 0.6.2`, `rustler 0.29.1`). Behaviour
marked "verified" was observed by running the code on Elixir 1.20.4 / OTP 29.1.2 (see
[`testing-and-ci.md`](testing-and-ci.md#4-behaviour-probe-results)).

## 1. Public API

### `Ortex.load(path, eps \\ [:cpu], opt \\ 3)`

Delegates to `Ortex.Model.load/3`, which calls `Ortex.Native.init/3` on a dirty IO scheduler.

| Argument | Accepted | Behaviour |
| --- | --- | --- |
| `path` | binary string | Charlists and other terms raise `ArgumentError`. A missing file raises `RuntimeError, "File at `...` does not exist"` (verified). |
| `eps` | list of atoms, highest priority first: `:cpu`, `:cuda`, `:tensorrt`, `:acl`, `:onednn`, `:coreml`, `:directml`, `:rocm` | Unknown atoms silently become CPU (verified with `:bogus`). Providers that were not compiled in or are unavailable at run time silently fall back to CPU (verified with `:cuda` on macOS). Strings raise `ArgumentError` (verified). |
| `opt` | integer | `1`, `2`, `3` select ONNX Runtime graph optimisation levels basic, extended, all; any other integer disables graph optimisation. |

Returns `%Ortex.Model{reference: ref}`. There is no `{:ok, _}`/`{:error, _}` variant and no keyword
options (thread counts, memory settings, config entries, loading from memory: see `TODO.md`).

### `Ortex.run(model, inputs)`

Delegates to `Ortex.Model.run/2`.

* `inputs` is a tuple of tensors in model input order, or a single tensor (wrapped into a 1-tuple).
* Each input is moved to `Ortex.Backend` with `Nx.backend_transfer/2` (a copy if it lives elsewhere).
* Returns a tuple of `Nx.Tensor`s on `Ortex.Backend`, one per model output, in model output order.
  A single-output model returns `{tensor}`.
* ONNX `bool` outputs come back as `{:u, 8}`. A `bool` model input needs a `{:u, 8}` tensor holding
  only 0 and 1; anything else panics inside the NIF (`ErlangError :nif_panicked`).
* Errors: wrong dtype or shape raise `RuntimeError` with ONNX Runtime's message (verified, the
  messages are good). Too few inputs raise an unhelpful kernel error from deep inside the graph
  (verified). **Extra inputs are silently ignored** (verified).

### `Ortex.Model`

`%Ortex.Model{reference: ref}` where `ref` is the NIF resource. The `Inspect` implementation calls
`Ortex.Native.show_session/1`. Actual output (verified):

```text
#Ortex.Model<
  inputs: [{"x", "Tensor {\n    ty: Int32,\n    dimensions: [\n        -1,\n        100,\n    ],\n}", [-1, 100]}, ...]
  outputs: [...]>
```

The README and the module doc still show the older format (`{"x", "Int32", [nil, 100]}`): the type
string is now `ort`'s multi-line `Debug` output and dynamic dimensions are `-1`, not `nil`.

### `Ortex.Serving`

Implements `Nx.Serving`. `Nx.Serving.new(Ortex.Serving, model)` works inline (`Nx.Serving.run/2`) and
as a process (`{Nx.Serving, serving: ..., name: ..., batch_size: ..., batch_timeout: ...}` plus
`Nx.Serving.batched_run/2`); both verified with `models/tinymodel.onnx`.

* `init(_type, model, [_defn_options])` only accepts **exactly one** partition. Nx passes one keyword
  list per partition (each with `:batch_keys`); with `partitions: true` that is `max_concurrency`
  lists for `Nx.Defn.Evaluator` and one per device for EXLA, so `init/3` raises
  `FunctionClauseError` (verified with Nx 1.0).
* `handle_batch/3` materialises the batch with `Nx.Defn.jit_apply(&Function.identity/1, [batch])`
  (the code calls it "a hack"; it is in fact what Nx's own `Nx.Batch` docs use), so stacking runs on
  the default `defn` compiler. It then calls `Ortex.run/2` synchronously before returning
  `{:execute, fn -> {out, :server_info} end, state}`.
* That ordering is a problem. Nx.Serving calls `handle_batch/3` inside its GenServer and runs the
  returned function in a supervised task, so a failing batch only fails that batch. Ortex runs the
  inference before returning the function, so inference blocks the serving process and an
  exception kills the whole serving. Materialisation and `Ortex.run/2` belong inside the `:execute`
  function. `Nx.LazyContainer.traverse/3` can materialise a batch without `defn`.
* `jit` and `defn` pre/post-processing inside the serving are not supported (module doc).

## 2. `Ortex.Backend`

A storage-only `Nx.Backend`: struct `%Ortex.Backend{ref: resource}`. The module also derives
`Nx.Container` for its struct (`containers: [:ref]`), which nothing uses; a reference is not a
container, so traversing it would fail. Remove it unless a test proves it is needed.

At compile time the module computes `Nx.Backend.behaviour_info(:callbacks) -- Module.definitions_in(__MODULE__, :def)`
and defines every missing callback to raise `"operation <name> is not yet supported on Ortex.Backend."`.
`Ortex.Backend.__unimplemented__/0` returns that list.

### Callback coverage (Nx 0.6.2: 132 callbacks)

| Status | Callbacks |
| --- | --- |
| Implemented (9) | `init/1`, `from_binary/3`, `to_binary/2`, `backend_transfer/3`, `inspect/2`, `slice/5`, `reshape/2`, `squeeze/3`, `concatenate/3` |
| Raise (123) | everything else, including the data-movement ones a storage backend should have: **`backend_copy/3`**, **`backend_deallocate/1`**, `as_type/2`, `to_batched/3`, `bitcast/2`, `constant/3`, `iota/3`, `eye/2`, `transpose/3`, `broadcast/4`, `pad/4`, `put_slice/4`, `reverse/3`, `select/4` |
| Optional callbacks also stubbed | `optional/3`, `top_k/3`, `svd/3`, `solve/3`, `phase/2`, `logical_not/2`, `determinant/2`, `cumulative_*/3`, `all_close/4`. Defining them stops Nx from using its built-in fallbacks; the generator should skip `behaviour_info(:optional_callbacks)`. |

### What changes with Nx 1.0.0 (115 callbacks, none optional)

Because the stubs are generated from `behaviour_info`, Ortex compiles and its tests pass on Nx 1.0.0
unchanged (verified). The callback set did change:

| Change | Callbacks | Effect on Ortex |
| --- | --- | --- |
| Added | `stack/3` (Nx 0.8), `to_pointer/2` and `from_pointer/5` (0.8), `block/4` (0.12, replaces `optional/3`) | **`Nx.stack/2` on Ortex tensors now raises.** Before Nx 0.8 the front end built `stack` from `new_axis` + `concatenate`, which Ortex supports. Implement `stack/3` with reshape + the concatenate NIF. Pointers are not needed. |
| Removed (21) | `all_close/4`, `cholesky/2`, `cumulative_*/3`, `determinant/2`, `eigh/3`, `logical_not/2`, `lu/3`, `map/4`, `optional/3`, `phase/2`, `qr/3`, `random_normal/4`, `random_uniform/4`, `solve/3`, `svd/3`, `take/4`, `take_along_axis/4`, `top_k/3` | None (stubs disappear). The "optional callbacks are stubbed" problem goes away. |
| Arity changed | `gather/4`, `indexed_add/5`, `indexed_put/5` | None (stubs regenerate). |

Other Nx 1.0 facts that matter here:

* The default integer type is `{:s, 32}` since Nx 0.9 (`Nx.tensor([1])`), so examples that print
  `s64` are out of date.
* `%Nx.Tensor{}` gained `donatable?: false`; `Ortex.Model.run/2` builds structs by hand and gets the
  default, which is fine.
* With mixed backends, Nx 1.0 hands the operation to the non-`BinaryBackend` backend, so
  `Nx.concatenate([ortex_tensor, binary_tensor])` reaches `Ortex.Backend.concatenate/3` and fails
  with `MatchError` (verified). It should transfer and cast inputs instead.
* `backend_deallocate/1` must return `:ok | :already_deallocated`. `Nx.BinaryBackend` implements
  `backend_copy/3` by delegating to `backend_transfer/3`. Nx.Serving's distributed docs recommend
  `Nx.backend_copy(&1, Nx.BinaryBackend)`, which raises on Ortex outputs today.
* `to_binary(tensor, limit)` must return exactly the first `limit` elements (Nx passes
  `min(size, limit)`); for sub-byte types the result may be a bitstring.
* Slice start indices: all-integer starts arrive pre-clamped; if any start is a tensor, Nx turns
  every start into a scalar tensor and the backend must clamp to `[0, dim - len]` itself.
* New numeric types exist: `{:s, 2}`, `{:s, 4}`, `{:u, 2}`, `{:u, 4}`, `{:f, 8}` (E5M2, Nx 0.9) and
  `{:f8_e4m3fn, 8}` (Nx 0.11). All of them panic the NIF today (verified).
* `Nx.Shape.squeeze/3`, which `Ortex.Backend.squeeze/3` calls, is a private API (`@moduledoc false`);
  Nx already passes the squeezed shape and names in `out`.

Implementation notes and known deviations from the Nx contract:

| Callback | Deviation |
| --- | --- |
| `to_binary/2` | Ignores `limit`; always returns the whole tensor (verified: `Nx.to_binary(t, limit: 2)` on a 6-element `f32` tensor returned 24 bytes, expected 8). |
| `backend_transfer/3` | Has a dead clause for `Nx.Tensor` as a target backend. The generic clause returns a zero-copy resource binary to the target backend (fine). Does not deallocate the source (fine for a GC-managed resource). |
| `backend_copy/3` | Not implemented, so `Nx.backend_copy(ortex_tensor, Nx.BinaryBackend)` raises (verified). |
| `backend_deallocate/1` | Not implemented, so `Nx.backend_deallocate/1` raises (verified). Should return `:ok`. |
| `slice/5` | Only integer start indices. Tensor start indices (allowed by Nx, which then clamps them) raise `ArgumentError` (verified). Calls a non-dirty NIF that copies data. |
| `reshape/2`, `squeeze/3`, `concatenate/3`, `slice/5` | Do not check for `{:error, msg}` from the NIF. An error would be stored as the tensor's `ref`. |
| `concatenate/3` | Raises unless all inputs have the same type (Nx would normally have cast them already). |
| `inspect/2` | Uses `min(:infinity, size)`, which works only because atoms sort after numbers. Adds an `Ortex.Backend` line when `config :ortex, add_backend_on_inspect: true` (the default; this repo turns it off in `:test`). |

## 3. Data types

| Nx type | Ortex (`OrtexTensor`) | ONNX type | Status |
| --- | --- | --- | --- |
| `{:s, 8}` `{:s, 16}` `{:s, 32}` `{:s, 64}` | `s8`..`s64` | `INT8`..`INT64` | supported in and out |
| `{:u, 8}` `{:u, 16}` `{:u, 32}` `{:u, 64}` | `u8`..`u64` | `UINT8`..`UINT64` | supported in and out |
| `{:f, 16}` `{:bf, 16}` `{:f, 32}` `{:f, 64}` | `f16`, `bf16`, `f32`, `f64` | `FLOAT16`, `BFLOAT16`, `FLOAT`, `DOUBLE` | supported in and out |
| `{:u, 8}` holding 0/1 | internal `bool` | `BOOL` | inputs converted when the model wants bool; outputs returned as `{:u, 8}` |
| `{:c, 64}` `{:c, 128}` | none | `COMPLEX64`, `COMPLEX128` | **panics** on transfer (verified); `ort` rc.15's `num-complex` feature could back them |
| (no Nx type) | none | `STRING` | output hits `todo!()` (panic) |
| `{:s, 4}` `{:u, 4}` `{:s, 2}` `{:u, 2}` (Nx 0.9+) | none | `INT4`, `UINT4`, `INT2`, `UINT2` | unsupported; panics on transfer (verified) |
| `{:f, 8}` (E5M2), `{:f8_e4m3fn, 8}` (Nx 0.9+, 0.11+) | none | `FLOAT8E5M2`, `FLOAT8E4M3FN` | unsupported; panics on transfer (verified) |
| (no Nx type) | none | `FLOAT8E4M3FNUZ`, `FLOAT8E5M2FNUZ` | unsupported |
| (none) | none | `FLOAT4E2M1`, `FLOAT8E8M0`, `FLOAT6*` | unsupported |
| (containers) | none | sequence, map, optional values | output **panics** |

## 4. Configuration keys

| Key | When read | Effect |
| --- | --- | --- |
| `config :ortex, add_backend_on_inspect: boolean` | compile time (`Application.compile_env`) | Adds the `Ortex.Backend` line to tensor `inspect` output. |
| `config :ortex, Ortex.Native, features: ["cuda", ...]` | compile time | Cargo features passed to the crate (selects which ONNX Runtime build `ort` downloads and which providers can register). Must be set in the **consuming application's** config; this repo's `config/config.exs` is ignored for dependents. |
| Other `config :ortex, Ortex.Native, ...` keys | compile time | Passed through to Rustler (e.g. `mode`, `target`, `env`), with the caveats of whichever Rustler version resolves. |

`config/config.exs` also sets `config :elixir, :dbg_callback, {Macro, :dbg, []}` with a comment that
"something is setting this to IEx.Pry". That is leftover debugging configuration; remove it.
