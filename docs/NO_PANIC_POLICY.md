# BitNet-rs No-Panic Family Policy

This is the dual-rail policy that governs panic-family API shapes
(`unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`,
`unreachable!`, and later `indexing`, `string_slice`, `get_unwrap`,
`unchecked_time_subtraction`, `unwrap_unchecked`).

```text
Clippy:
  immediate code-shape detector inside the compiler
xtask check-no-panic-family:
  owner / reason / selector / expiry control plane
```

## Why two rails?

Clippy gives instant feedback inside the compile cycle; it does not
have an opinion about *who* owns a particular use of `unwrap()` or
*when* the team has agreed to remove it. The semantic checker fills
that gap by maintaining `policy/no-panic-allowlist.toml`, where each
exception is a receipt with:

* `id` — stable identifier
* `path` — file path the receipt applies to
* `family` — one of the panic families above
* `snippet` — exact normalized source snippet for the finding
* `count` — number of matching occurrences covered by this receipt
* `classification` — `test_helper`, `kernel`, `ffi_boundary`, etc.
* `owner` — team / area responsible
* `explanation` — why this exception is acceptable now
* `expires` — ISO date by which the receipt must be renewed or removed
* `selector` — optional pin describing the specific call site
  (`kind`, `container`, `callee`, `receiver_fingerprint`)
* `last_seen` — advisory line/column

Identity is exact and counted:

```text
path + family + selector.kind + selector.callee + snippet + count
```

`last_seen` is not part of identity; it is advisory location evidence only.

## Workflow

1. Run `cargo run --no-default-features -p xtask -- check-no-panic-family` locally; this
   reports unallowlisted findings without changing the allowlist.
2. If a finding is real debt that cannot yet be removed, inspect
   `target/bitnet/reports/no-panic-proposed-allowlist.toml`. That file is
   advisory only; it gives humans exact counted receipt shapes to review.
3. The real allowlist is `policy/no-panic-allowlist.toml`. Auto-tools
   never write to it; humans review each entry.
4. The generated baseline is `policy/no-panic-baseline.toml`. It carries
   existing exact-counted debt and is marked generated in `.gitattributes`.
5. Refresh the baseline with
   `cargo run --no-default-features -p xtask -- no-panic baseline`. A normal
   refresh may only drop disappeared entries; it refuses to absorb new debt.
   `--reset` is reserved for the dedicated baseline reset PR.
6. The allowlist policy mode is `no-new-debt`, so any finding outside the
   allowlist and baseline fails the checker.

## Family staging

PR 05 ships these families:

```text
unwrap, expect, panic_macro, todo, unimplemented, unreachable
```

Later PRs promote:

```text
indexing, string_slice, get_unwrap,
unchecked_time_subtraction, unwrap_unchecked
```

`assert!`/`assert_eq!` are intentionally not in scope today. They are
test oracles in the current codebase; future fallible-assertion
migration is a separate decision.

## Tests

Tests are part of the contract. `unwrap`/`expect` in tests are not
free of cost — they hide real failure modes behind a single line of
panic. Use the helpers in `bitnet-test-support::assertions` (`ensure`,
`ensure_eq`, `require_some`, `require_ok`) to make tests fallible
where it makes sense. The Rust 1.95 PR 6 removed the remaining
`clippy.toml` test carveouts (`allow-unwrap-in-tests` and
`allow-expect-in-tests`). Remaining panic-family findings are handled by
exact identity, baseline, and owner-lane burndown instead of reintroducing
test-specific Clippy exemptions.

## When `panic` is genuinely acceptable

For each retained panic in FFI, GPU, or hardware-only code paths the
receipt must answer:

* Why is panic acceptable here?
* What boundary contains it (FFI, kernel guard, etc.)?
* Why is `Result` not better?
* Who owns it?
* When does the receipt expire?

Any "we panic so the upstream API is simpler" rationale is rejected.

## Rust 1.95 rollout target state

The following changes are part of the Rust 1.95 / next minor wave.
See `docs/development/RUST_1_95_ROLLOUT.md` for the full PR ladder and
CI economics framing.

Current `main` has the no-panic allowlist present, exact counted identity in
the checker, and a generated `policy/no-panic-baseline.toml`. Policy mode is
`no-new-debt`: existing findings are consumed by exact allowlist or baseline
counts, and anything left is new debt.

### Identity hardening (PR 7)

Allowlist identity is now exact and counted before bulk baseline work begins:

```text
path
family
selector_kind
selector_callee
snippet
count
```

Matching is **counted and consumptive**:

1. Consume exact allowlist count slots.
2. Then consume baseline count slots, unless in blocking mode.
3. Anything remaining is new debt.

This prevents one allow entry from accidentally covering unrelated calls in the
same file with the same callee.

Required tests for this contract:

```text
allowlist_entry_requires_exact_snippet
allowlist_count_is_consumed_per_occurrence
allowlist_does_not_cover_same_file_same_callee_different_snippet
baseline_generation_subtracts_allowlisted_counts
duplicate_allowlist_keys_are_rejected
blocking_mode_ignores_baseline_but_honors_counted_allowlist
```

### Baseline and no-new-debt mode (PR 8)

After identity hardening, a generated baseline was created from current `main`
and the policy mode was set to `no-new-debt`. The baseline file is marked
generated in `.gitattributes` so it collapses in GitHub review:

```gitattributes
policy/no-panic-baseline.toml text eol=lf linguist-generated=true
```

Baseline refresh may only drop disappeared entries, never absorb new findings:

```bash
cargo run --locked -p xtask --no-default-features -- no-panic baseline
```

Use `--reset` only when the PR's explicit purpose is to regenerate the
baseline. Allowlist entries added after baseline generation must be
exact-counted and explicitly reviewed.

### Diagnostic improvements (PR 9)

PR 9 adds:

* Missing baseline setup error (clear message when baseline file absent).
* Stale baseline entries in Markdown/JSON reports.
