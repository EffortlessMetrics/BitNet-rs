# BitNet-rs Clippy Policy

This is the policy contract for Clippy and related Rust shape lints in
BitNet-rs. It is enforced by:

* `[workspace.lints.clippy]` in the root `Cargo.toml` (lands staged in
  PR 08 and is promoted in PR 12).
* `cargo run -p xtask -- check-clippy-exceptions` (added in PR 06),
  which validates every receipt in `policy/clippy-exceptions.toml`
  and rejects bare `#[allow(clippy::...)]` shapes.

## Suppression style

Suppressions must be receipts, not decisions:

```rust
// allowed
#[expect(clippy::indexing_slicing, reason = "policy:clippy-0001")]
fn lookup(idx: usize) -> u8 {
    TABLE[idx]
}

// rejected
#[allow(clippy::indexing_slicing)]
fn lookup(idx: usize) -> u8 {
    TABLE[idx]
}
```

Every `#[expect(clippy::...)]` must reference an entry in
`policy/clippy-exceptions.toml`. Each entry carries:

| Field            | Required | Meaning                                                   |
| ---------------- | -------- | --------------------------------------------------------- |
| `id`             | yes      | Stable identifier, format `clippy-NNNN`                   |
| `lint`           | yes      | Clippy lint name, e.g. `clippy::indexing_slicing`         |
| `path`           | yes      | File path or glob the receipt covers                      |
| `classification` | yes      | One of: `generated_table`, `ffi_boundary`, `kernel`, etc. |
| `owner`          | yes      | Team / area responsible                                   |
| `reason`         | yes      | Why the exception is acceptable today                     |
| `expires`        | yes      | ISO date by which the receipt must be renewed or removed  |
| `selector`       | optional | Selector that pins the exception to specific code shape   |
| `last_seen`      | optional | Advisory line/column hint                                 |

## Test carveouts

BitNet-rs intentionally **does not** add Clippy test carveouts:

```toml
# NOT used
allow-unwrap-in-tests = true
allow-expect-in-tests = true
allow-panic-in-tests = true
allow-indexing-slicing-in-tests = true
allow-dbg-in-tests = true
```

Tests are part of the contract. Setup, fixture loading, parsing,
indexing, and helper plumbing should be fallible. The
`bitnet-test-support::assertions` module (PR 09) provides `ensure`,
`ensure_eq`, `require_some`, and `require_ok` to make panic-free
tests practical.

The current `clippy.toml` retains
`allow-expect-in-tests = true` / `allow-unwrap-in-tests = true` for
the staging window only; PR 09–11 remove the dependency on these
carveouts and PR 12 deletes them.

## `unsafe_code`

The workspace lints set:

```toml
unsafe_code = "deny"
unsafe_op_in_unsafe_fn = "deny"
```

`forbid` is **not** used. BitNet-rs has legitimate unsafe-adjacent
surfaces: FFI, GPU kernels, language bindings, memory mapping, C ABI,
and SIMD. Each unsafe island must be narrow, documented, and
eventually receipted via `policy/unsafe-allowlist.toml` (added later
in the rollout once the current islands are inventoried).

## Stages

PR 08 (Stage A) introduces the explicit profile. The staged lints
that arrive after MSRV bumps (1.94, 1.95) are listed in
`policy/clippy-lints.toml`. PR 12 (the strict flip) promotes
panic-family lints from `warn` to `deny` and adds the rest of the
shared baseline.

## When to add an exception

Order of preference:

1. Refactor the code so the lint stops triggering.
2. Use a fallible helper (`?`, `ensure`, `ok_or_else`).
3. Add a temporary `#[expect]` with a real receipt and an expiry.
4. Last resort: file `policy/clippy-debt.toml` debt for a whole
   crate or path, with an owner and an expiry.

There is no "permanent allow" path: receipts must be reviewed by their
expiry date.
