# BitNet-rs Clippy Policy

This is the policy contract for Clippy and related Rust shape lints in
BitNet-rs. It is enforced by:

* `[workspace.lints.clippy]` in the root `Cargo.toml` (lands staged in
  PR 08 and is promoted in PR 12).
* `cargo run --no-default-features -p xtask -- check-clippy-exceptions` (added in PR 06),
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

The Rust 1.95 follow-up PR 6 removed the last
`allow-expect-in-tests = true` / `allow-unwrap-in-tests = true` entries from
`clippy.toml`. The workspace still carries broader panic-family debt under the
staged lint profile; that debt is tracked by the no-panic checker and later
owner-lane burndown, not by reintroducing test carveouts.

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

## Rust 1.95 rollout target state

The following changes are planned as part of the Rust 1.95 / next minor wave.
See `docs/development/RUST_1_95_ROLLOUT.md` for the full PR ladder and
CI economics framing.

Current `main` has `clippy.toml` pinned to `msrv = "1.95.0"` and no longer
uses test unwrap/expect carveouts. `policy/clippy-lints.toml` records the Rust
1.95 MSRV. PR 5 measured the Rust 1.94/1.95 ratchets before activation and
promoted only the clean lints from staged policy into `[workspace.lints.clippy]`.

### Lint ratchets (PR 5)

The following staged lints are active after the PR 5 measurement pass:

| Lint | Level | Reason |
|---|---|---|
| `same_length_and_capacity` | `deny` | Catch raw-parts reconstruction mistakes (also staged at 1.94) |
| `manual_ilog2` | `warn` | Prefer standard integer log helper (also staged at 1.94) |
| `decimal_bitwise_operands` | `warn` | Make bit masks visually inspectable (also staged at 1.94) |
| `manual_take` | `warn` | Use standard ownership helper instead of local reimplementation |
| `needless_type_cast` | `warn` | Avoid stale numeric type drift (also staged at 1.94) |

The following lints remain deferred with policy debt or a toolchain note:

| Lint | Status | Reason |
|---|---|---|
| `manual_checked_ops` | deferred | Measured findings in kernel arithmetic and model config paths need invariant-preserving cleanup. |
| `duration_suboptimal_units` | deferred | Measured duration-unit findings need workspace-wide cleanup rather than partial activation. |
| `unnecessary_trailing_comma` | deferred | Measured kernel formatting findings are mechanical but noisy and belong in their own cleanup. |
| `manual_pop_if` | deferred | The installed Rust 1.95 Clippy reports this lint as unknown, so it remains staged until the toolchain exposes it. |

Because CI Core uses `RUSTFLAGS = "-Dwarnings"` and Clippy with
`-D warnings`, warning-level Clippy lints can still behave as hard failures in
default lanes. If measurement finds hits, PR 5 either fixes them in scope or
keeps the lint staged with a real debt entry.

### `disallowed_fields` (PR 5 prerequisite)

`disallowed_fields` is not activated globally until protected seam definitions
are present in `clippy.toml`. Candidate seams are listed in
`docs/development/RUST_1_95_ROLLOUT.md`. Activation is a deliberate step after
the seams are specified, not an automatic promotion.

### Test carveout removal (PR 6)

PR 6 removed the remaining Clippy test unwrap/expect carveouts from
`clippy.toml`. The first migrated helper slice is
`bitnet-test-support::assertions`, whose unit tests now return
`anyhow::Result<()>` and exercise `ensure`, `ensure_eq`, `require_some`, and
`require_ok` without relying on unwrap/expect.

### Clippy debt and exceptions (PR 5–6 and later)

`policy/clippy-debt.toml` currently contains placeholder entries only.
The 1.95 wave requires real entries with `owner`, `reason`, and `expiry`
for any debt that cannot be resolved immediately. Placeholder entries
that lack these fields are invalid under the updated checker.

`policy/clippy-exceptions.toml` currently has no entries. Exceptions added
during the 1.95 wave must follow the exact `#[expect]` receipt schema with
a `clippy-NNNN` identifier.
