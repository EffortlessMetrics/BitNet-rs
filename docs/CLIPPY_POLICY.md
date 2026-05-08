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

## Rust 1.95 rollout target state

The following changes are planned as part of the Rust 1.95 / 0.3.0 wave.
See `docs/development/RUST_1_95_ROLLOUT.md` for the full PR ladder.

### Lint ratchets (PR 5)

The lints staged in `policy/clippy-lints.toml` for MSRV 1.95 will be
promoted to `[workspace.lints.clippy]`:

| Lint | Level | Reason |
|---|---|---|
| `same_length_and_capacity` | `deny` | Catch raw-parts reconstruction mistakes (also staged at 1.94) |
| `manual_checked_ops` | `warn` | Prefer checked arithmetic over manual divide-by-zero guards |
| `manual_take` | `warn` | Use standard ownership helper instead of local reimplementation |
| `manual_pop_if` | `warn` | Use collection APIs that encode predicate-and-pop intent |
| `duration_suboptimal_units` | `warn` | Make durations legible without mental unit conversion |
| `needless_type_cast` | `warn` | Avoid stale numeric type drift (also staged at 1.94) |
| `unnecessary_trailing_comma` | `warn` | Keep format macro calls clean |

Lints are only promoted after a clean measurement pass confirms zero or
cheap-to-fix violations in the workspace.

### `disallowed_fields` (PR 5 prerequisite)

`disallowed_fields` is not activated globally until protected seam definitions
are present in `clippy.toml`. Candidate seams are listed in
`docs/development/RUST_1_95_ROLLOUT.md`. Activation is a deliberate step after
the seams are specified, not an automatic promotion.

### Test carveout removal (PR 6)

The following lines in `clippy.toml` are a staging window and will be removed
in PR 6:

```toml
allow-expect-in-tests = true
allow-unwrap-in-tests = true
```

PR 6 also adds any remaining ergonomic fallible helpers to the appropriate
test-support crate and converts a first narrow batch of tests before removal.

### Clippy debt and exceptions (PR 5–6 and later)

`policy/clippy-debt.toml` currently contains placeholder entries only.
The 1.95 wave requires real entries with `owner`, `reason`, and `expiry`
for any debt that cannot be resolved immediately. Placeholder entries
that lack these fields are invalid under the updated checker.

`policy/clippy-exceptions.toml` currently has no entries. Exceptions added
during the 1.95 wave must follow the exact `#[expect]` receipt schema with
a `clippy-NNNN` identifier.
