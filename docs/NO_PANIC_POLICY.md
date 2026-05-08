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
* `classification` — `test_helper`, `kernel`, `ffi_boundary`, etc.
* `owner` — team / area responsible
* `explanation` — why this exception is acceptable now
* `expires` — ISO date by which the receipt must be renewed or removed
* `selector` — optional pin describing the specific call site
  (`kind`, `container`, `callee`, `receiver_fingerprint`)
* `last_seen` — advisory line/column

Identity is `path + family + selector`. `last_seen` is not part of
identity, so the receipt keeps matching across small edits.

## Workflow

1. Run `cargo run -p xtask -- check-no-panic-family` locally; this
   reports unallowlisted findings without changing the allowlist.
2. If a finding is real debt that cannot yet be removed, run
   `cargo run -p xtask -- no-panic propose` (later PR) — it writes a
   draft receipt to `target/bitnet/reports/no-panic-proposed-allowlist.toml`
   for the human to edit and copy in.
3. The real allowlist is `policy/no-panic-allowlist.toml`. Auto-tools
   never write to it; humans review each entry.
4. CI runs the checker with `--fail-on-error` once the rollout reaches
   PR 10 / 11 / 12. Until then it runs advisory.

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
where it makes sense. The current carveouts in `clippy.toml`
(`allow-unwrap-in-tests`, `allow-expect-in-tests`) are temporary
during PR 09–11 and removed in PR 12.

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

The following changes are planned as part of the Rust 1.95 / 0.3.0 wave.
See `docs/development/RUST_1_95_ROLLOUT.md` for the full PR ladder.

### Identity hardening (PR 7)

Current allowlist identity is `path + family + selector`. Before bulk baseline
work begins, identity must expand to include `snippet` and `count`:

```text
path
family
selector_kind
selector_callee
snippet
count
```

Matching becomes **counted and consumptive**:

1. Consume exact allowlist count slots.
2. Then consume baseline count slots, unless in blocking mode.
3. Anything remaining is new debt.

This prevents one allow entry from accidentally covering unrelated calls in the
same file with the same callee.

Required tests before PR 7 merges:

```text
allowlist_entry_requires_exact_snippet
allowlist_count_is_consumed_per_occurrence
allowlist_does_not_cover_same_file_same_callee_different_snippet
duplicate_allowlist_keys_are_rejected
blocking_mode_ignores_baseline_but_honors_counted_allowlist
```

### Baseline and no-new-debt mode (PR 8)

After identity hardening, a generated baseline is created from current `main`
and the policy mode is set to `no-new-debt`. The baseline file is marked
generated in `.gitattributes` so it collapses in GitHub review:

```gitattributes
policy/no-panic-baseline.toml text eol=lf linguist-generated=true
```

Baseline refresh may only drop disappeared entries, never absorb new findings.
Allowlist entries added after baseline generation must be exact-counted and
explicitly reviewed.

### Diagnostic improvements (PR 9)

PR 9 adds:

- Missing baseline setup error (clear message when baseline file absent).
- Stale baseline entries in Markdown/JSON reports.
- Baseline refresh delta details (what appeared, what disappeared).
- Blocking-mode baseline messaging (explain that baseline is ignored in
  blocking mode).

### First burndown lane (PR 14)

The first burndown PR targets one narrow lane. Good starting candidates:

```text
bitnet-atomic-file-core
bitnet-http-retry
bitnet-api-key-auth-core
bitnet-client-ip-core
bitnet-request-router-core
bitnet-server-health-types-core
xtask policy/report helpers
```

Avoid starting with `bitnet-kernels`, FFI, GPU backends, large CLI/server
integration tests, or Python/WASM bindings.

Replacement strategy:

```text
production unwrap/expect → ? / ok_or_else / typed error
test setup unwrap/expect → fallible helper
intentional retained panic → exact allowlist with count + expiry
```
