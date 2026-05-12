# `ripr` Evidence Policy

`ripr` is static mutation-exposure analysis. It asks:

> Does the changed Rust behavior appear gripped by a meaningful test
> discriminator?

It catches much of the same signal mutation testing catches -- weak
test/oracle exposure -- but earlier and cheaper, because it runs statically and
can run per PR.

Mutation testing remains the runtime empirical backstop, especially for nightly
and release readiness. The CI design uses `ripr` to shift mutation signal left,
not to pretend mutation is unnecessary. See also `docs/ci/ripr.md`.

## What ripr produces

`ripr check` emits, per finding:

| Severity              | Meaning                                                   |
| --------------------- | --------------------------------------------------------- |
| `exposed`             | The change is reachable and a static test discriminator covers it. |
| `weakly_exposed`      | A discriminator exists but its observation is shallow.    |
| `reachable_unrevealed`| Reachable from a test, but no observable assertion grips it. |
| `no_static_path`      | No static call path from any test reaches the change.     |
| `infection_unknown`   | Cannot statically decide whether the input would propagate. |
| `propagation_unknown` | Cannot statically decide whether output reaches an oracle. |
| `static_unknown`      | The analysis bailed out (e.g. macro, dynamic dispatch).   |

## Posture in this rollout

PR 13 ships ripr **advisory only**. The workflow at
`.github/workflows/ripr.yml`:

* runs on `pull_request` against production Rust diffs;
* invokes `ripr check` with `ripr.toml`;
* uploads `target/ripr/ripr.{json,sarif,md}` as artifacts;
* writes a step summary;
* exits 0 regardless of finding severity.

The Rust 1.95 rollout changes this from "binary may be absent" to "install and
run `ripr` consistently." When analysis is skipped by policy or cannot run, the
workflow records that fact in the step summary. Skipped analysis must not be
reported as passed proof.

## Why advisory first

* The tool can be noisy on a 200-crate workspace until baseline
  behavior is understood.
* BitNet-rs has a lot of macro-heavy code (BDD grids, runtime
  feature flags) that may produce `static_unknown` findings.
* The team needs to land suppressions in
  `policy/ripr-suppressions.toml` for known-acceptable gaps before
  promotion.

## Promotion path

1. Run advisory for at least one full sprint cycle.
2. Triage `target/ripr/` artifacts; record acknowledged gaps in
   `policy/ripr-suppressions.toml` with owner, reason, and expiry.
3. Once the noise floor is understood, promote `exposed = "notice"`
   to `notice` (no change), and lift `weakly_exposed` and
   `reachable_unrevealed` to `warning` annotations on review.
4. Eventually, set `[policy] fail_on = ["weakly_exposed",
   "reachable_unrevealed"]` so PR review can be evidence-driven
   without being blocking.

## What ripr is not

* Not a substitute for mutation testing (which still runs for targeted risk
  PRs, nightly, and release readiness).
* Not a coverage report.
* Not a code reviewer — its annotations are inputs for review, not
  decisions.

## Suppression schema

See `policy/ripr-suppressions.toml`. Each suppression records:

```toml
[[suppress]]
id      = "ripr-0001"
path    = "crates/X/src/Y.rs"
finding = "no_static_path"
owner   = "core/runtime"
reason  = "Reachable only via dyn dispatch; covered by integration test foo."
expires = "2026-08-01"
```

Suppressions, like every other policy receipt in BitNet-rs, must
have an owner, a reason, and an expiry.

## Toolchain dependency

`ripr` requires Rust `1.93` or newer. The current workspace MSRV is 1.95.0; the
Rust 1.95 / next minor rollout keeps the floor ahead of that requirement before
the real advisory `ripr` provisioning PR.
