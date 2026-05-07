# Learned LEM budgets

PR 20 of the strict policy / CI economics rollout. Replaces the
static lane costs in `policy/ci-lanes.toml` with a rolling
estimate computed from observed actuals.

## Model

```text
estimate = max(static_floor, p50_recent_actual * 1.15)
warning  = p90_recent_actual
hard     = p95_recent_actual
```

The 1.15× factor on the median is intentional: real CI runtime
distributions are right-skewed (cache misses, queueing, runner
flake), so estimating at the median underbudgets ~50 % of PRs. A
15 % cushion on the p50 still reads "tight but realistic" while
keeping the planner's headline number close to lived experience.

## Inputs

| Source                              | Role                                 |
| ----------------------------------- | ------------------------------------ |
| `.ci/metrics/ci-lane-history.jsonl` | Append-only JSONL of historical actuals |
| `policy/ci-lanes.toml`              | Static lower bounds per lane          |

The history file is written by the actuals collector
(`xtask ci actuals`, PR 16). Each line is one record:

```json
{"lane":"ci-core-build-test","actual_lem":18.6,"conclusion":"success"}
```

Comments (lines starting with `#`) and blank lines are skipped.
Records without a `lane` field are skipped.

## Outputs

`target/ci/ci-lane-estimates.json`:

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-07T09:50:00Z",
  "window_runs": 50,
  "lanes": {
    "ci-core-build-test": {
      "lane": "ci-core-build-test",
      "samples": 50,
      "p50": 14.0,
      "p90": 21.5,
      "p95": 25.0,
      "static_floor": 22.0,
      "estimate": 22.0
    }
  }
}
```

`samples` is the number of records that contributed (capped by
`--window`, default 50).

## CLI

```bash
cargo run -p xtask -- ci estimate \
  --history .ci/metrics/ci-lane-history.jsonl \
  --lanes-toml policy/ci-lanes.toml \
  --json-out target/ci/ci-lane-estimates.json \
  --window 50
```

## Posture in this PR

PR 20 ships:

* the calculation (`xtask/src/ci/estimate.rs` + 6 unit tests);
* the consumer surface (the JSON report under
  `target/ci/ci-lane-estimates.json`);
* this prose policy.

It does **not** wire the learned estimate into `xtask ci plan` yet.
That switch needs at least one full sprint of `ci-actuals.json`
data to be meaningful, and the planner-side change is reversible
in one commit when the data is ready.

The promotion path is:

1. Land `xtask ci actuals` collection on at least the ten most
   active default-PR lanes.
2. Aggregate the per-job records into `.ci/metrics/ci-lane-history.jsonl`
   in nightly maintenance.
3. Run `xtask ci estimate` weekly, sanity-check the output.
4. Switch the planner to read learned estimates as the headline
   LEM with the static floor as a fallback.

## Why JSONL, not a database

`.ci/metrics/ci-lane-history.jsonl` is intentionally an append-only
text file:

* commits cleanly; reviewers can eyeball the diff;
* survives any change of CI provider;
* easy to grep / cut / awk for ad-hoc analysis;
* the file is < 10 MB even at one record per PR per lane for a
  full year of activity.

When the volume genuinely exceeds what plain text can handle, the
schema can lift to Parquet / DuckDB without changing the consumer
surface (the lane name is the only join key).
