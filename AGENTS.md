# AGENTS.md

Instructions for AI coding agents (and humans) working on Ortex. Read this file first, then
[`docs/README.md`](docs/README.md). Known problems and planned work are in [`TODO.md`](TODO.md).

## What Ortex is

Ortex is an Elixir library that runs ONNX models through ONNX Runtime. Elixir code
(`lib/`) talks to a Rust NIF (`native/ortex`, built with Rustler) that uses the `ort` crate, which
links ONNX Runtime (a C++ library downloaded at build time). Ortex provides `Ortex.load/3`,
`Ortex.run/2`, a storage-only `Nx.Backend` (`Ortex.Backend`) and an `Nx.Serving` module
(`Ortex.Serving`).

## Status snapshot (2026-10-29)

* `main` is `55dc3de`: Hex release `0.1.10` (2024-11-19) plus the Elixir 1.19 `Inspect` fix (PR #48).
* Pins: `ort 2.0.0-rc.8` (ONNX Runtime 1.19.2, max ONNX opset 21), Rustler crate `0.29`, Hex
  `{:rustler, "~> 0.27"}`, `{:nx, "~> 0.6"}` (excludes Nx 1.0), `ndarray 0.16`, `edition = "2018"`.
* Verified: `main` compiles and all 117 tests pass on Elixir 1.20.4 / OTP 29.1.2 / Rust 1.99.1, and
  also with Nx 1.0.0 + Hex Rustler 0.38 (no code changes). Hex `0.1.10` itself predates PR #48, so
  `inspect(model)` is broken for Hex users on Elixir 1.19+ until the next release.
* Time bomb: Rust 1.100 (about 2026-11-12) breaks every dynamically linked build (CUDA/TensorRT,
  including this repo's Linux CI) because `ort-sys` rc.8's `copy-dylibs` cannot handle cargo's new
  build-dir layout. Verified on 1.100.0-beta.7. See `TODO.md` BUILD-10.
* Upstream today: `ort 2.0.0-rc.15` (still a release candidate), ONNX Runtime 1.31.1, Nx 1.0.0,
  Rustler 0.38 (Hex) / 0.39 (crate), rustler_precompiled 0.10.1.
* Open PRs touch most files: #47 (modernise + `ort` rc.11), #49 (precompiled NIFs), #46 (`ort` rc.10),
  #50, #51, #52 (session options). Read `TODO.md` section "Pull requests" before starting work
  that overlaps them. The maintainer asked for #47 to be split into "modernisation" and "ort upgrade".

## Repository map

```text
lib/ortex.ex               Public facade: load/3, run/2 (delegates to Ortex.Model)
lib/ortex/model.ex         %Ortex.Model{reference}: load, run, Inspect implementation
lib/ortex/backend.ex       Ortex.Backend: storage-only Nx.Backend (9 callbacks, rest raise)
lib/ortex/serving.ex       Ortex.Serving: Nx.Serving behaviour
lib/ortex/native.ex        NIF stubs; builds the Rust crate at compile time; loads the NIF
lib/ortex/util.ex          copy_ort_libs/0: copies libonnxruntime* next to the NIF
native/ortex/              Rust crate (cdylib "ortex")
  src/lib.rs               NIF table (rustler::init!), term decoding, scheduler choice
  src/model.rs             OrtexModel { session }: init, show, run
  src/tensor.rs            OrtexTensor enum (one ndarray variant per dtype), conversions
  src/utils.rs             binary <-> tensor, execution provider and optimisation mapping
  src/constants.rs         atoms and execution provider names
  .cargo/config.toml       rpath and macOS link flags (shipped in the Hex package)
config/config.exs          Dev-only: per-OS cargo features. Ignored when Ortex is a dependency.
test/                      ExUnit tests (dtype, shape ops, serving); resnet50 tests are opt-in
models/tinymodel.onnx      Small 2-input/3-output test model (regenerate: python/multi_input.py)
python/                    Model export scripts (PyTorch); write into the current directory
examples/                  distilbert and stablelm scripts (need dev deps and exported models)
docs/                      Architecture and maintenance docs for agents
TODO.md                    Review findings, GitHub issue/PR triage, roadmap
```

## Setup

* `mix.exs` declares `elixir: "~> 1.14"` (no OTP constraint). CI runs Elixir 1.17.3 / OTP 27.1.2;
  Elixir 1.20.4 / OTP 29.1.2 is verified to work.
* A stable Rust toolchain (verified with 1.99.1). There is no precompiled NIF yet.
* Network access on the first build: `ort`'s build script downloads ONNX Runtime from
  `parcel.pyke.io`. Offline or unusual targets need `ORT_LIB_LOCATION` (see
  [`docs/build-and-packaging.md`](docs/build-and-packaging.md)).
* Python with `torch`/`torchvision`/`onnx` only if you regenerate test models.

## Commands

```sh
mix deps.get
MIX_ENV=test mix compile --warnings-as-errors   # builds the NIF too
mix test                                        # resnet50 tests auto-excluded without models/resnet50.onnx
mix format --check-formatted
(cd native/ortex && cargo fmt --check && cargo clippy --all-targets)
```

Use `MIX_ENV=test` for library work. The `:dev` environment pulls `exla`, `torchx` and
`tokenizers`, which download large native artifacts; `exla` and `torchx` are not used anywhere.
That also makes `mix docs` slow until the dev deps are trimmed (see `TODO.md`).
`mix test --warnings-as-errors` currently fails after a passing run because `test/test_helper.exs`
uses `IO.warn` when `models/resnet50.onnx` is missing (TODO TEST-08).

## Gotchas (read before changing code)

1. **Rust changes are not picked up by `mix compile`.** `lib/ortex/native.ex` builds the crate
   itself through Rustler's private `compile_crate` and then sets `skip_compilation?: true`, so Mix
   does not track the Rust sources. Run `mix compile --force` (or touch `lib/ortex/native.ex`) after
   editing anything under `native/`. With Rustler 0.37+ (what new installs resolve) the NIF is
   written to `priv/native/ortex.so` in the source tree and cargo builds in `native/ortex/target/`;
   do not commit either.
2. **A NIF is declared in three places** that must agree: the `#[rustler::nif]` function in
   `native/ortex/src/lib.rs`, the list inside `rustler::init!` (Rustler 0.29 needs it), and the
   stub in `lib/ortex/native.ex` (same name and arity, body `:erlang.nif_error(:nif_not_loaded)`).
3. **NIF errors are values.** Rust returns `rustler::Error::Term(Box::new(msg))`, which arrives in
   Elixir as `{:error, msg}`. Every Elixir caller must match it. Decoding failures arrive as
   `ArgumentError`; Rust panics arrive as `ErlangError :nif_panicked`.
4. **No panics on user-reachable paths.** No `unwrap`, `expect`, `panic!`, `todo!` or
   `unimplemented!` on data that came from Elixir or from a model. Return an error instead.
5. **Tensors are immutable and contiguous.** `OrtexTensor` arrays must be standard (row-major)
   layout and must never be mutated: `to_binary` hands out zero-copy BEAM binaries that alias them.
6. **Schedulers.** Anything that can take more than about 1 ms (copies of large arrays, file IO,
   inference) must be a dirty NIF (`schedule = "DirtyCpu"` or `"DirtyIo"`).
7. **Bool.** Nx has no boolean type. ONNX bool outputs are returned as `{:u, 8}`; `{:u, 8}` inputs are
   converted to bool only when the model input is bool.
8. **Dependency config is not loaded.** `config/config.exs` only affects development of this repo.
   Applications that depend on Ortex set `config :ortex, Ortex.Native, features: [...]` themselves.
9. **`ort` and `ndarray` move together.** `ort`'s ndarray conversions only accept the `ndarray`
   major version `ort` itself depends on. Bump both at once.
10. **`ort` is pre-2.0 with breaking changes between release candidates.** Pin an exact version
    (`=2.0.0-rc.N`; a caret requirement lets `cargo update` jump RCs) and read the notes of every
    intermediate RC. Rules that bite during the rc.15 upgrade (details in `docs/upgrading.md` 3.3):
    only one ONNX Runtime `Environment` per OS process, ever; `Session::run` needs `&mut self`;
    `GraphOptimizationLevel::Level3` no longer means "all" (use `All`); CPU-only sessions need
    `AutoDevicePolicy::PreferCPU`; `half` must be enabled as a feature.
11. **Execution providers fail silently.** An unknown atom, or a provider that is not compiled in
    or not installed, quietly runs on CPU. Do not rely on `Ortex.load(path, [:cuda])` raising.

## Design rules

All code follows Responsibility-Driven Design: every module owns its data and behaviour, and
collaborators ask it to act instead of reading or duplicating its state. Before writing code, state
each new or changed module's responsibilities (what it knows, what it does) and its
collaborators, and check that nothing is split or misplaced. The current ownership map and the
known violations are in [`docs/architecture.md`](docs/architecture.md) sections 3 and 9. In practice:

* `Ortex.Backend` owns the shape of its tensor data. Other modules must ask it to wrap or unwrap
  NIF references instead of pattern-matching `%Ortex.Backend{}` inside `%Nx.Tensor{}`.
* The native layer owns everything about how ONNX Runtime is built, linked and which execution
  providers exist. Elixir asks the NIF; it does not mirror Cargo features or build directories.
* A tensor knows its own dtype and shape. Do not pass them from Elixir when Rust can read them.
* The NIF boundary owns error translation: one place converts native results into Elixir errors.

## Recipes

**Add a NIF.** Write the Rust function in `lib.rs` (choose a scheduler), add it to `rustler::init!`
if the Rustler version needs the list, add the Elixir stub to `lib/ortex/native.ex`, call it from
the owning Elixir module, handle `{:error, msg}`, add a test, run `mix compile --force`.

**Add a dtype.** Add a variant to `OrtexTensor` and to every `match` in `tensor.rs` (`shape`, `reshape`,
`dtype`, `to_bytes`, `slice`, `Clone`, both `TryFrom` impls, `concatenate`), extend
`utils::from_binary`, add the atom to `constants.rs` if needed, extend `test/dtype/dtype_test.exs`
and the concat test table. Check `docs/elixir-layer.md` section 3 for the Nx/ONNX mapping.

**Add or change an execution provider.** Cargo feature in `native/ortex/Cargo.toml`, name in
`constants.rs`, mapping in `utils::map_eps`, docs in `docs/execution-providers.md`. Remember that
`ort` rc.13+ only compiles a provider's struct when its Cargo feature is enabled.

**Add a session option.** Prefer a keyword option on `Ortex.load/2` (see the unified design in
`TODO.md`) mapped to one `SessionBuilder` call in `model.rs`. Do not add more positional arguments.

**Upgrade `ort` / ONNX Runtime or Elixir dependencies.** Follow [`docs/upgrading.md`](docs/upgrading.md).

## Tests and verification

* Every change: `mix format`, `MIX_ENV=test mix compile --warnings-as-errors`, `mix test`.
  Rust changes also: `cargo fmt` and `cargo clippy` in `native/ortex`.
* Bug fixes come with a regression test. Many current bugs are listed with reproduction steps in
  [`docs/testing-and-ci.md`](docs/testing-and-ci.md#4-behaviour-probe-results).
* Keep committed test models tiny and commit the script that generates them.

## Pull requests

* Keep PRs small and focused; split dependency bumps, refactors and features. When work depends on
  an unmerged layer, use stacked PRs where each layer builds and passes tests on its own.
* Update `docs/` when behaviour or architecture changes, and tick or edit items in `TODO.md`.
* Never commit `_build/`, `deps/`, `native/ortex/target/` or large `.onnx` files.
