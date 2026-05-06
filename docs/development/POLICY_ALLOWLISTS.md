# Policy Allowlists

BitNet-rs governs exceptions to the strict policy through structured TOML
allowlists under `policy/`. Each exception is a *receipt*: it must name the
file or call site, the family of rule it relaxes, the owner who is on the
hook for retiring it, the reason, and (where appropriate) when it expires
and how it is otherwise covered.

This is the operating model:

> **Global deny by default. Local exception by structured receipt.**

Exceptions are fine. *Silent* exceptions are not.

## Files

| File | Purpose | Added in |
| --- | --- | --- |
| `policy/clippy-lints.toml` | Active lint baseline metadata + planned 1.94/1.95 flips | PR 1 |
| `policy/non-rust-allowlist.toml` | Allowed non-Rust programming/declarative files | PR 2 |
| `policy/no-panic-allowlist.toml` | Semantic allowlist for panic-family call sites | PR 3 |
| `policy/clippy-debt.toml` | Repo-local Clippy debt (path/glob + lint + expiry) | PR 5 onwards |

Each file has a `schema_version` field and is parsed by
`cargo xtask policy-report`. Schema breakage fails CI.

## Non-Rust allowlist (PR 2)

Identity: `path` *or* `glob`.

Required fields: `kind`, `owner`, `surface`, `classification`, `reason`.
Production/test/tooling surfaces also require `covered_by` (the commands
that exercise the file). Optional: `expires`.

The recognized BitNet `kind` vocabulary:

```
documentation
rust_manifest
rust_lockfile
ci_declarative
repo_config
policy_metadata
fixture_input
fixture_golden
gpu_shader
ffi_c
ffi_cpp
python_binding
wasm_binding
build_config
generated_metadata
asset
license
benchmark_data
```

Surfaces: `editor`, `ci`, `gpu`, `ffi`, `language-binding`, `repo-config`,
`policy`, `docs`, `fixtures`, `assets`, `benchmark`.

Classifications: `production`, `test`, `tooling`, `config`, `docs`,
`fixtures`, `assets`.

## No-panic allowlist (PR 3)

Identity: `path` + `family` + `selector` (kind, container, callee, optional
receiver fingerprint). `last_seen` (line/column) is **advisory** — it is a
hint to help locate the call site after refactors but is *not* part of the
matching key. Real motion that changes call-site meaning correctly fails
the check.

Families covered in the first wave:

```
unwrap
expect
panic_macro
todo
unimplemented
unreachable
```

`assert!`, `assert_eq!`, and `assert_ne!` are not in this first wave; they
are test oracles and migrate to fallible helpers under their own campaign.

## Expiry

Generated baseline debt should default to a short expiry
(`expires = "2026-07-01"`). Expired entries fail
`cargo xtask check-no-panic-family` and `cargo xtask check-file-policy`,
forcing renewal with an updated reason or removal of the underlying
violation.

## Reports

```
cargo xtask check-file-policy
cargo xtask check-no-panic-family
cargo xtask check-lint-policy
cargo xtask policy-report
```

All four are added incrementally over the rollout PR stack. Reports are
written to `target/bitnet/reports/` and uploaded as CI artifacts when the
`ripr` evidence reporting PR (PR 10) lands.
