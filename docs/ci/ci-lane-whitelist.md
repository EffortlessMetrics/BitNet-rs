# CI Lane Whitelist

This is the governed map of every CI lane in BitNet-rs. Each lane in the
whitelist (`policy/ci-lane-whitelist.toml`) is mapped to:

```text
CI item -> purpose -> proof obligation -> cost -> trigger -> owner -> evidence -> duplicate-of -> review date
```

The whitelist exists to prevent two failure modes:

1. **Duplicate proof obligations under different workflow names.**
   Two lanes that exercise the same surface should be marked
   `duplicate_of` and consolidated.
2. **Expensive lanes silently becoming default PR work.**
   Default-PR lanes must be cheap (low LEM); expensive lanes must be
   gated on labels, paths, or scheduled triggers.

## LEM (Linux-Equivalent Minutes)

LEM normalizes wall-clock minutes by runner cost so Linux, Windows,
macOS, Docker, GPU, and external review lanes can be compared in one
unit. Multipliers live in `[runner_multipliers]` of the whitelist.

Practical bands:

| LEM       | Posture                                           |
| --------- | ------------------------------------------------- |
| 0 - 25    | preferred default (frontdoor lanes)               |
| 26 - 35   | acceptable default                                |
| 36 - 75   | elevated; warrants justification                  |
| 76 - 125  | high; suggests `ci-budget-ack` label              |
| > 125     | hard ceiling; requires `full-ci` / override label |

## Tiers

* `frontdoor` — required on every PR, cheap, fast feedback
* `frontdoor-advisory` — runs on every relevant PR but not blocking
* `risk-gated` — runs only when the touched paths or labels indicate need
* `expensive` — opt-in via labels or scheduled runs; never default
* `deep` — main / nightly / release / labeled only

## Schema fields

| Field              | Required | Meaning                                                       |
| ------------------ | -------- | ------------------------------------------------------------- |
| `id`               | yes      | Stable identifier referenced by exceptions and dependents     |
| `workflow`         | yes      | Path to the workflow definition                               |
| `job`              | yes      | Job name within the workflow                                  |
| `kind`             | yes      | One of: `control`, `rust`, `lint`, `docs`, `feature`, `policy`, `gpu`, `ffi`, `compatibility`, `oracle-gap`, `platform` |
| `tier`             | yes      | One of: `frontdoor`, `frontdoor-advisory`, `risk-gated`, `expensive`, `deep` |
| `default_pr`       | yes      | Whether the lane runs on every PR                             |
| `blocking`         | yes      | Whether the lane gates merge                                  |
| `runner`           | yes      | Runner family used for LEM cost                               |
| `base_lem`         | one of   | Static LEM estimate (preferred for Linux lanes)               |
| `base_minutes`     | one of   | Wall-clock minutes (used when LEM not yet calibrated)         |
| `owner`            | yes      | Team / area responsible                                       |
| `intent`           | yes      | One sentence: what does this lane buy?                        |
| `failure_mode`     | yes      | One sentence: what regression slips if this lane is missing?  |
| `proof_obligation` | yes      | What the job actually runs                                    |
| `evidence`         | yes      | Artifacts or logs that prove the obligation                   |
| `allowed_triggers` | yes      | `pull_request`, `push`, `workflow_dispatch`, `pull_request:labeled`, `pull_request:path-gated` |
| `labels`           | no       | Labels that gate the lane (when not default)                  |
| `duplicate_of`     | yes      | Other lane IDs that overlap proof; `[]` if none               |
| `review_after`     | yes      | ISO date for next routine review                              |
| `expires`          | yes      | ISO date by which lane must be reviewed or removed            |
| `expensive`        | no       | Marks lanes treated as costly in budget planning              |
| `expensive_reason` | no       | Human-readable explanation                                    |

## Exceptions

`policy/ci-whitelist-exceptions.toml` records every lane that
intentionally violates a default rule (for example, an expensive lane
left in the default PR set, or a duplicate that we accept temporarily).
Each exception requires `owner`, `reason`, `created`, `review_after`,
and `expires`.

## How to update

When adding or modifying a CI workflow:

1. Add or update the corresponding `[[lane]]` entry in
   `policy/ci-lane-whitelist.toml`.
2. If the lane runs by default and is expensive, add an exception in
   `policy/ci-whitelist-exceptions.toml` with a real expiry.
3. If the lane duplicates an existing proof obligation, set
   `duplicate_of` to the other lane's `id`.
4. Confirm `cargo run -p xtask -- ci-lane-whitelist check` passes
   (added in PR 02 of the rollout).

## Status

* PR 01 (this PR): policy files only, advisory.
* PR 02: `xtask ci-lane-whitelist check` enforces structure and detects
  workflow drift from the whitelist.
