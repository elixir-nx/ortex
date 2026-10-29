# Build, linking and packaging

Snapshot: `main` at `55dc3de` (`ort 2.0.0-rc.8`, ONNX Runtime 1.19.2). For what changes with newer `ort`
releases see [`upgrading.md`](upgrading.md).

## 1. The short version

* Ortex has **no precompiled NIF**. Every install compiles the Rust crate, so every user needs a
  Rust toolchain (issue #27; PR #49 proposes `rustler_precompiled`).
* During that cargo build, `ort-sys`'s build script **downloads a prebuilt ONNX Runtime** from pyke's
  CDN (`parcel.pyke.io`), verifies its SHA-256, and links it. CPU builds are linked **statically**
  (no side library); CUDA builds are **dynamic** and need `libonnxruntime*` next to the NIF.
* Only six targets have prebuilt ONNX Runtime binaries. Anything else (Nerves armv7/armv6, musl,
  RISC-V, FreeBSD) fails the build unless you supply ONNX Runtime yourself via `ORT_LIB_LOCATION`.

## 2. What happens during `mix compile`

The body of `lib/ortex/native.ex` runs at compile time:

1. Reads the Rustler version (`Application.spec(:rustler, :vsn)`) and calls the matching API:
   `Rustler.Compiler.compile_crate(:ortex, Application.compile_env(:ortex, Ortex.Native, []), otp_app: :ortex, crate: :ortex)`
   for Rustler >= 0.30, or `compile_crate(__MODULE__, otp_app: :ortex, crate: :ortex)` for older ones.
2. Rustler runs cargo in release mode with the configured Cargo features (observed: "Compiling crate
   ortex in release mode" even with `MIX_ENV=test`).
3. `ort-sys`'s build script picks a binary *variant* from the enabled features, downloads and links it
   (section 3).
4. Rustler copies the built NIF into `priv/native/`. File name and build directory depend on Rustler:

   | Rustler (Hex) | Cargo target dir | NIF location |
   | --- | --- | --- |
   | 0.29.x to 0.36.x (this repo's `mix.lock` has 0.29.1) | `_build/<env>/lib/ortex/native/ortex/<profile>/` (Rustler sets `CARGO_TARGET_DIR`) | `_build/<env>/lib/ortex/priv/native/libortex.so` |
   | 0.37.0+ (`~> 0.27` resolves to 0.38.0 for new users) | cargo's default: `<ortex source>/native/ortex/target/<profile>/` (for a dependency: `deps/ortex/native/ortex/target/release/`) | `<ortex source>/priv/native/ortex.so`; `_build/<env>/lib/ortex/priv` becomes a symlink to the source `priv/` |

   The 0.37.0 change is not in Rustler's changelog ("Refactor build logic to remove TOML dependency");
   it was confirmed by comparing the 0.36.2 and 0.37.0 sources and by building a consumer app.
   Because the NIF now lands in the source tree, `/priv/native/` should be in `.gitignore`.

5. `Ortex.Util.copy_ort_libs/0` globs `libonnxruntime*.{so,dylib,dll}*` in
   `_build/<env>/lib/ortex/native/ortex/{release|debug}` and copies matches into `priv/native/`.
   **With Rustler >= 0.37 that directory does not exist**, so nothing is copied. Static builds do not
   care; dynamic (CUDA) builds then fail to load with "cannot open shared object file". On Windows
   the glob also misses `onnxruntime.dll` (no `lib` prefix), and cross-compiled builds put their
   output under `target/<triple>/<profile>/`, which is not searched either.
6. `use Rustler, otp_app: :ortex, crate: :ortex, skip_compilation?: true` installs `@on_load`, which
   loads the NIF from `priv/native/` when the module loads.

Mix does not track the Rust sources (no `@external_resource`), so editing them does not trigger a
rebuild: use `mix compile --force`. The cause is `skip_compilation?: true`: Rustler only records the
crate's files as external resources when it compiles the crate itself. Replacing the private
`Rustler.Compiler.compile_crate` call with a plain `use Rustler, otp_app: :ortex, crate: :ortex`
followed by the library copy works with Rustler 0.38 and restores rebuild tracking (verified).

### Rust 1.100 and `copy-dylibs` (deadline about 2026-11-12)

Rust 1.100 ships cargo's new build-dir layout (`OUT_DIR` moves one level deeper, to
`target/<profile>/build/<pkg>/<hash>/out`). `ort-sys` rc.8 finds the target directory with
`OUT_DIR.ancestors().nth(3)` and `unwrap()`s symlink creation there. On Rust 1.100.0-beta.7:

* static CPU builds (no dylibs to copy) build fine;
* any build whose ONNX Runtime is a dynamic library (the `cu12` CUDA/TensorRT download, or a
  dynamic `ORT_LIB_LOCATION`) fails: `failed to run custom build command for ort-sys v2.0.0-rc.8 ...
  panicked at build.rs:111:66 ... No such file or directory`.

`ort` rc.14 and later support the new layout. Until Ortex upgrades, GPU users must stay on Rust 1.99,
and this repository's Linux CI (which enables `cuda`/`tensorrt`) will fail once runners update.

## 3. How `ort` 2.0.0-rc.8 obtains ONNX Runtime

Default `ort` features: `ndarray`, `half`, `download-binaries`, `copy-dylibs`.

**Variant selection** (`ort-sys` build script): `train` if the `training` feature is on; `cu12` if
`cuda` or `tensorrt` is on (else `rocm` if `rocm`); otherwise `none`. If the variant does not exist for
the target, it silently falls back to `none`; if `none` does not exist either, the build panics with
"downloaded binaries not available for target ... you may have to compile ONNX Runtime from source".

| Target | `none` (CPU, static) | `cu12` (CUDA 12, dynamic) | `train` |
| --- | --- | --- | --- |
| `aarch64-apple-darwin` | yes (includes CoreML) | no | yes |
| `x86_64-apple-darwin` | yes | no | yes |
| `x86_64-unknown-linux-gnu` | yes | **yes** | yes |
| `aarch64-unknown-linux-gnu` | yes | no (falls back to CPU) | yes |
| `x86_64-pc-windows-msvc` | yes | **yes** | yes |
| `aarch64-pc-windows-msvc` | yes | no | yes |
| anything else (armv7, musl, riscv64, ...) | **no: build fails** | no | no |

Binaries come from `https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.19.2/...`
(still reachable on 2026-10-29) and are checked against SHA-256 values compiled into `ort-sys`.

**Environment variables** read by `ort-sys` rc.8:

| Variable | Effect |
| --- | --- |
| `ORT_LIB_LOCATION` | Use an ONNX Runtime you built or downloaded instead of downloading. Points at the directory with the libraries (static or dynamic). |
| `ORT_LIB_PROFILE` | Build profile subdirectory to use inside an ONNX Runtime build tree. |
| `ORT_PREFER_DYNAMIC_LINK` | Link dynamically even if a static library is available. |

`copy-dylibs` copies dynamic libraries into cargo's output directory so something can place them
next to the NIF (that something is `Ortex.Util.copy_ort_libs/0`). The `load-dynamic` feature (load
ONNX Runtime at run time from `ORT_DYLIB_PATH`) is not used.

## 4. Linking and loading the NIF

`native/ortex/.cargo/config.toml` (also shipped in the Hex package):

| Setting | Why |
| --- | --- |
| `rpath = true` for dev and release profiles | Let the NIF find side libraries relative to itself. |
| macOS: `-undefined dynamic_lookup` | NIF symbols (`enif_*`) are resolved from the VM at load time. |
| macOS: `-fapple-link-rtlib` | The static ONNX Runtime needs compiler-rt builtins (e.g. `___isPlatformVersionAtLeast`); added by the "fix for MacOS linking failure" commit. |
| macOS: `-Wl,-rpath,@loader_path`; Linux x86_64/aarch64: `-Wl,-rpath,$ORIGIN` | Search the NIF's own directory (`priv/native`) for `libonnxruntime*`. |

Windows has no rpath. A dynamically linked ONNX Runtime must be found through the DLL search order
(directory of the executable, system directories, `PATH`), which `priv/native` is not part of.

With the Rustler crate 0.36 or later, `enif_*` symbols are resolved at load time by Rustler itself,
so macOS no longer needs `-undefined dynamic_lookup`. A probe with crate 0.39 linked and passed all
tests on macOS arm64 with only `-Wl,-rpath,@loader_path`, without `-fapple-link-rtlib` either. With
crate 0.29 (today), removing `-undefined dynamic_lookup` fails with "Undefined symbols _enif_*".

Verified on macOS arm64: the NIF is a single 21 MB library linking ONNX Runtime statically, plus
system frameworks (Foundation, CoreFoundation, CoreML) and `libc++`.

## 5. Choosing Cargo features (execution providers)

* **This repository** (`config/config.exs`, dev and test only): Windows `["directml"]`, macOS
  `["coreml"]`, other Unix `["cuda", "tensorrt"]`. On Linux this makes CI download the CUDA 12
  dynamic build.
* **Applications using Ortex**: Mix never loads a dependency's `config/config.exs`, so the default is
  **no features**: the static CPU build on every target. To get GPU support an application sets:

  ```elixir
  # config/config.exs of the application
  config :ortex, Ortex.Native, features: ["cuda"]
  ```

  and then needs the CUDA 12 and cuDNN libraries at run time, plus a working `copy_ort_libs`
  (broken with Rustler >= 0.37, see section 2). The `coreml` and `directml` features only matter
  for registering those providers; on rc.8 their binaries are the same static `none` build.

Details per provider: [`execution-providers.md`](execution-providers.md).

## 6. The Hex package

`mix.exs` `package[:files]`: `lib`, `.formatter.exs`, `mix.exs`, `README*`, `LICENSE*`,
`native/ortex/src/`, `config/config.exs`, `native/ortex/Cargo.lock`, `native/ortex/Cargo.toml`,
`native/ortex/.cargo/config.toml` (matches the published `0.1.10` tarball). Shipping
`config/config.exs` has no effect on users. Published requirements: `rustler ~> 0.27`, `nx ~> 0.6`.

Because the requirements are loose, a new application today resolves **Rustler 0.38.0 and Nx 0.13.1**
(not the versions in this repo's `mix.lock`). Verified on Elixir 1.20.4 / OTP 29.1.2 / macOS arm64:
that combination compiles (Rust warnings only) and runs `models/tinymodel.onnx`. It cannot be
combined with anything that requires Nx 1.0.

## 7. Releasing (current practice)

There is no release automation and no `CHANGELOG`. Releases so far: bump `version` in `mix.exs`, tag
`vX.Y.Z` (tags exist only for 0.1.8 to 0.1.10), `mix hex.publish` by the Hex owner (`travismorton`,
GitHub `@mortont`). PR #49 proposes a tag-driven workflow that builds NIFs for 5 targets x NIF 2.16/2.17
and a `RELEASE.md` runbook.

## 8. Precompiled NIFs (not yet available)

Requested in issue #27; implemented in open PR #49 with `rustler_precompiled`. Things to settle
before shipping precompiled artifacts:

* Artifacts must be self-contained. Static ONNX Runtime (the `none` variant) is; GPU variants are not.
  GPU builds need either `rustler_precompiled` *variants* that ship the ONNX Runtime side libraries,
  or remain source-build only.
* Which providers each artifact supports must be documented: PR #49 builds without Cargo features,
  so CoreML (macOS) and DirectML (Windows) providers would not register in precompiled builds.
* NIF versions (from `erl_nif.h` at each OTP tag): OTP 24 and 25 are 2.16; OTP 26, 27 and 28 are 2.17;
  OTP 29 is 2.18. A library built for a lower minor version loads on a newer runtime, so 2.16 and 2.17
  artifacts cover OTP 24 to 29. (PR #49's workflow comment says "2.16 covers OTP 24-26", which is
  wrong for OTP 26 but harmless.)
* Upstream ONNX Runtime stopped publishing x86_64 macOS binaries with 1.24, and `ort` rc.11+ has no
  Intel macOS downloads, so `x86_64-apple-darwin` artifacts only work while Ortex stays on rc.8.
* The checksum file must be generated for each release before `mix hex.publish` (PR #49 ships an
  empty `%{}` placeholder).
* glibc baseline: artifacts built on `ubuntu-22.04` need glibc 2.35+, which excludes Debian 11,
  Ubuntu 20.04, RHEL 8 and Amazon Linux 2. Build Linux artifacts in an older container (manylinux)
  or with `cargo-zigbuild` to widen support. After the `ort` upgrade the floor is set by pyke's
  static ONNX Runtime instead: glibc 2.39 / libstdc++ 13.2 (Ubuntu 24.04+, Debian 13+), whatever the
  build runner. Supporting older distributions then means linking Microsoft's ONNX Runtime (glibc 2.28+)
  and shipping it next to the NIF.
* rustler_precompiled facts (0.10.1): default NIF version list is `["2.15"]`, and a 2.15 NIF loads on
  OTP 22 to 29, so one artifact per target is enough; `variants` select alternative artifacts per
  target at compile time (the mechanism for CUDA builds); extraction unpacks every file in the
  tarball into `priv/native`, so side libraries can ship next to the NIF; require `>= 0.10.1`
  (earlier versions fail on GitHub's large response headers).
* The static ONNX Runtime still links `libstdc++` dynamically, so minimal runtime images (and Nerves
  root filesystems) must include it.

## 9. Cross-compiling, Nerves, Docker, offline

* **Cross-compiling / Nerves (issue #41).** The prebuilt table in section 3 has no 32-bit ARM, no
  musl and no RISC-V; those targets need an ONNX Runtime built for the target and `ORT_LIB_LOCATION`.
  For aarch64 Nerves (rpi4/5, CM4) a community recipe works with Hex `0.1.10`
  (`abelino/ortex_fw`, `config/target.exs`):

  ```elixir
  config :ortex, Ortex.Native,
    target: "aarch64-unknown-linux-gnu",
    env: [{"CC", ""}, {"CFLAGS", ""},
          {"CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER", "aarch64-nerves-linux-gnu-gcc"}]
  ```

  Why it is needed: Nerves exports `CC`/`CFLAGS` for the target, `cc`-based build scripts that run on
  the host (the `ort-sys` downloader's TLS stack, the unused `rustls`/`ring` dependency) pick them up
  and fail, and without `target:` the crate builds for the host. Ortex could derive all of this from
  the Nerves environment. The device needs `libstdc++`.
* **Docker (issue #43):** the build image needs Rust, a C/C++ toolchain for linking, CA certificates
  (for the download) and network access. After the `ort` upgrade it also needs OpenSSL headers on
  Linux (rc.10+ downloads through native TLS) and Rust 1.92+ (rc.14+). The runtime image needs the NIF
  and `libstdc++` (static CPU builds), plus `libonnxruntime*` and the CUDA libraries for GPU builds.
* **Offline builds:** download or build ONNX Runtime ahead of time and set `ORT_LIB_LOCATION`.
* **Large models and the dirty IO stack (issue #42):** ONNX Runtime's graph optimisation can recurse
  deeply. Session creation runs on a dirty IO scheduler thread (default stack 40 kilowords, 320 KiB
  on 64-bit), and a 300 MB model overflowed it and killed the VM. Workarounds today: a lower
  optimisation level, or a larger dirty IO stack with `+sssdio <kilowords>` (e.g. in
  `rel/vm.args.eex` or `ERL_FLAGS`). The fix is to build sessions on a Rust thread with a large stack.

## 10. After the `ort` rc.15 upgrade (planned)

What changes in this document once Ortex moves to `ort` rc.15 (details in `upgrading.md` section 3):

| Topic | rc.8 (today) | rc.15 |
| --- | --- | --- |
| Download host and format | `parcel.pyke.io`, `.tgz` | `https://cdn.pyke.io/0/pyke:ort-rs/ms@1.31.0/<target>[+<features>].tar.lzma2`, SHA-256 pinned, GitHub attestations |
| Download cache | `ort-sys` cache directory | `$ORT_CACHE_DIR`, else `~/.cache/ort.pyke.io` (Linux) or `~/Library/Caches/ort.pyke.io` (macOS); cache it in CI |
| Linking | static for CPU, dynamic for CUDA (`cu12`) | always a static core (`libonnxruntime.a`); CUDA/TensorRT packages add provider `.so` files loaded at run time |
| Side libraries next to the NIF | `libonnxruntime*` for CUDA builds | only `libonnxruntime_providers_*` (and WebGPU's Dawn library) |
| Custom ONNX Runtime | `ORT_LIB_LOCATION` | `ORT_LIB_PATH` (old name still read, removed before 2.0.0); `load-dynamic` with `ort::init_from(path)` or an absolute `ORT_DYLIB_PATH` |
| Build host | Rust 1.70+, no OpenSSL | Rust 1.92+; OpenSSL headers on Linux unless the `tls-rustls` feature is used |
| Prebuilt targets | 6 (incl. Intel macOS) | Apple Silicon macOS 13.4+, x86_64/aarch64 Linux (glibc 2.39+, AVX2 on x86_64), x86_64/aarch64 Windows; no Intel macOS, musl, armv7 or riscv |
| Rust 1.100 | dynamic builds fail (section 2) | supported |
| Feature/target validation | none | `ort` reports impossible provider sets as a link error, but `-undefined dynamic_lookup` (macOS) hides it; validate in Elixir and load the NIF in CI |
