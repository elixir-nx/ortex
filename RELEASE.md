# Release process

Ortex publishes a Rust NIF that depends on `onnxruntime`. As of `ort` 2.0.0-rc.x, `onnxruntime` is statically linked into the NIF binary, so the artifact tarballs contain a single self-contained shared library and no sidecar dependencies.

This document is the runbook for cutting a release.

## Supported precompile targets

| Target | Runner | Notes |
|---|---|---|
| `aarch64-apple-darwin` | `macos-14` | Apple Silicon, native build |
| `x86_64-apple-darwin` | `macos-13` | Intel Mac, native build |
| `x86_64-unknown-linux-gnu` | `ubuntu-22.04` | glibc >= 2.35; older distros fall back to source build |
| `aarch64-unknown-linux-gnu` | `ubuntu-22.04` (cross) | Cross-compiled via `cross-rs` |
| `x86_64-pc-windows-msvc` | `windows-2022` | Native build, MSVC ABI |

For each target we publish two NIF-version variants (`2.16` covering OTP 24-26, `2.17` covering OTP 27+), so each release produces 10 tarballs.

Targets outside this matrix automatically fall back to source build via the existing `ORTEX_BUILD` path. Consumers on those targets need a Rust toolchain — same as before precompilation existed.

## Cutting a release — step by step

### 1. Bump version

Edit `mix.exs` `@version` and update `CHANGELOG.md`. Commit on a release branch.

### 2. Push a pre-release tag

```bash
git tag v0.1.11-pre1
git push origin v0.1.11-pre1
```

The tag-push trigger in `.github/workflows/release.yml` runs the matrix build. Each successful job uploads its tarball to a **draft** GitHub release. Watch the workflow run; if any cell fails, fix and retag (`v0.1.11-pre2`, etc.).

### 3. Promote the draft release to a GitHub release (still pre-release flag)

In the GitHub UI, find the draft release for the pre-release tag and untoggle "Save as draft" (keep the "Set as a pre-release" flag). The artifacts are now downloadable at stable URLs.

### 4. Generate the checksum file

From a clean checkout of the release branch:

```bash
mix deps.get
mix rustler_precompiled.download Ortex.Native --all --print > checksum-Elixir.Ortex.Native.exs
```

The command pulls each tarball from the GitHub release URL pattern (defined by `base_url:` in `lib/ortex/native.ex`), computes its SHA-256, and writes the resulting map. Inspect the diff — every supported target × NIF-version combination should appear.

Commit the updated `checksum-Elixir.Ortex.Native.exs`.

### 5. Smoke-test the precompiled path locally

In a separate scratch directory:

```bash
# Without ORTEX_BUILD set — should download and use the precompiled NIF
mix new ortex_smoke && cd ortex_smoke
cat <<'EOF' >> mix.exs
# In deps/0:
{:ortex, path: "../ortex"}
EOF
mix deps.compile ortex --force
```

Should succeed without invoking Rust. If you have `cargo` on `PATH`, temporarily move it (`PATH=${PATH//$(dirname $(which cargo)):/}`) to confirm precompilation isn't silently falling back to source build.

### 6. Verify source fallback regression

```bash
ORTEX_BUILD=true mix deps.compile ortex --force
```

Should exercise the legacy path — `Rustler.Compiler.compile_crate/2` runs, `Ortex.Util.copy_ort_libs/0` runs (no-op when onnxruntime is statically linked), tests pass.

### 7. Promote the pre-release to a real release

Once smoke tests pass, push the final tag (`v0.1.11`), let the workflow re-run, promote the draft to a non-pre-release GitHub release. The checksum file already in the repo references the same artifact filenames; verify the SHA-256s still match (they should — same crate, same matrix).

### 8. Publish to Hex

```bash
mix hex.publish
```

The package now contains the `checksum-Elixir.Ortex.Native.exs` file; consumers will fetch precompiled NIFs by default.

## Updating the precompile matrix

Adding a new target means three things:

1. Add a new entry to `precompiled_targets` in `lib/ortex/native.ex`.
2. Add a corresponding `{ target: ..., os: ..., use-cross?: ... }` row to the matrix in `.github/workflows/release.yml`.
3. Re-run the release process — the new target's checksums end up in the next checksum file regeneration.

Removing a target is the inverse, with one caveat: existing consumers pinned to an older version of Ortex still rely on the artifact existing at the old release URL. Don't delete artifacts from past releases.

## Falling back to source build

Consumers who hit the source-build path get there in two ways:

1. Their target tuple isn't in the `precompiled_targets` list. RustlerPrecompiled detects this and prints a notice, then triggers a source build automatically.
2. They explicitly set `ORTEX_BUILD=true` before `mix deps.compile`. This is documented in the README as the manual override.

Both paths require a Rust toolchain and exercise the original `Rustler.Compiler.compile_crate/2` + `Ortex.Util.copy_ort_libs/0` flow. That path remains supported indefinitely — the precompiled artifacts are an optimisation, not a replacement.

## Common failure modes

* **"checksum mismatch" during install** — the GitHub release artifact was modified or rebuilt without the checksum file being regenerated. Solution: re-run step 4.

* **"no precompiled NIF available for ..."** — consumer is on a target outside the matrix. Solution: instruct them to set `ORTEX_BUILD=true` and ensure they have a Rust toolchain installed.

* **CI matrix cell fails for one target only** — fix it specifically; other targets' artifacts uploaded by the same workflow run are still valid. Re-tagging only re-runs *all* cells, which is wasteful but safe.

* **Old glibc on a Linux consumer** — the precompiled `x86_64-unknown-linux-gnu` artifact requires glibc >= 2.35 (set by the `ubuntu-22.04` runner image). To support older glibc, the workflow's Linux-x86_64 cell can be moved to a `manylinux_2_28` (or older) container; this is a follow-up worth doing if user reports come in.
