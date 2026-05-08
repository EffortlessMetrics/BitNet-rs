# BitNet-rs File Policy

Rust and `xtask` are the default implementation surface. Non-Rust
files are allowed only through a structured receipt in
`policy/non-rust-allowlist.toml`.

This is the same dual-rail design as the no-panic policy:

```text
implicit allow:
  every *.rs file and any Cargo.toml
explicit allow:
  every other tracked file must match an entry in
  policy/non-rust-allowlist.toml
```

`xtask check-file-policy` walks tracked files (via `git ls-files` if
the repo is a git checkout, otherwise the working tree) and reports
any non-Rust file that is not covered by an allowlist glob.

## Schema

```toml
[[allow]]
glob = "crates/bitnet-metal/**/*.metal"
kind = "gpu_shader"
owner = "gpu/metal"
surface = "gpu"
classification = "production"
reason = "Metal compute shader implementation surface."
covered_by = ["cargo check -p bitnet-metal"]
```

| Field            | Required | Meaning                                                   |
| ---------------- | -------- | --------------------------------------------------------- |
| `glob`           | yes      | Path glob (subset: `*`, `**`, `{a,b}`)                    |
| `kind`           | yes      | Short tag, e.g. `gpu_shader`, `ffi_surface`, `markdown`   |
| `owner`          | yes      | Team / area responsible                                   |
| `surface`        | yes      | High-level surface, e.g. `gpu`, `ffi`, `docs`, `policy`   |
| `classification` | yes      | One of: `production`, `test`, `documentation`, `tooling`, `config` |
| `reason`         | yes      | Why the surface is allowed in non-Rust form               |
| `covered_by`     | yes      | Verification commands or workflows that exercise it       |

## How the checker resolves files

1. Files ending in `.rs` are skipped (implicit allow for Rust source).
2. Any file named `Cargo.toml` is skipped (implicit allow; covered by
   `xtask check-lint-inheritance` instead).
3. Each remaining tracked file is matched against every `glob` in the
   allowlist; the first match wins.
4. Files that match nothing are reported as findings.

The matcher supports:

* `*` — any number of non-`/` chars
* `**` — any path sequence including `/`
* `{a,b,c}` — alternations

## Adding a new non-Rust file

If a new file legitimately belongs in the repository:

1. Choose the most specific existing entry that already covers your
   file. If your file is just one more shader, header, or markdown
   under an already-allowed glob, no policy change is needed.
2. If it is a genuinely new surface (e.g. a new GPU backend with its
   own shader directory), add a new `[[allow]]` block with a real
   `owner`, `surface`, `classification`, and `covered_by`.
3. Run `cargo run -p xtask -- check-file-policy` locally; the file
   should disappear from the findings.
4. Open the PR with the new entry. Policy review is part of normal
   PR review.

## What this policy is not

* It is not a content allowlist. It does not enforce "what you can
  write inside a file"; that is governed by Clippy, the no-panic
  checker, and the unsafe islands policy.
* It is not a security allowlist for binaries. Binary blobs and
  models are excluded by `.gitignore`; only metadata or fixtures
  small enough to commit are allowed.

## Status

PR 04 introduces the allowlist, the prose, and runs the checker
advisory. Promotion to `--fail-on-error` happens in the same PR
once the in-tree finding count is 0.

## Rust 1.95 rollout target state

The following changes are planned as part of the Rust 1.95 / 0.3.0 wave.
See `docs/development/RUST_1_95_ROLLOUT.md` for the full PR ladder.

### Allowlist tightening (PR 10)

PR 10 reviews the current `policy/non-rust-allowlist.toml` and:

- Removes stale entries that no longer match any tracked file.
- Narrows over-broad agent metadata globs where a more specific pattern fits.
- Adds `review_after` / `expires` fields to entries where the checker supports
  them.
- Verifies that production non-Rust surfaces have real `covered_by` commands.
- Ensures GPU shader, FFI, Python, and WASM surfaces are explicit entries, not
  hidden under broad `scripts/**` or `crates/**` catch-alls.

The following constraints apply to PR 10:

- No broad catch-all globs added.
- Checker policy is not weakened.
- Production GPU/FFI code is not reclassified as docs or tooling.

### Non-Rust surfaces that must remain explicit

The following surface categories must be individually allowlisted, not merged
into a single broad glob:

```text
GPU compute shaders (.metal, .glsl, .hlsl, .wgsl, .cl)
FFI headers and generated bindings (.h, .hpp, generated .rs via build.rs)
Python bindings and package metadata (setup.py, pyproject.toml, *.py)
WASM bindings (*.wasm, wasm-pack artifacts)
C++ cross-validation code (.cpp, .cc, .cxx)
Runtime config (e.g. ONNX/GGUF metadata, JSON schemas)
```

Each surface requires a `covered_by` entry that references an actual
verification command or workflow file.
