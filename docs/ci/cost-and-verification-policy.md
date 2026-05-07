# CI Cost and Verification Policy

BitNet-rs intentionally targets CI cost per ordinary pull request far below the
default cost profile common in large agentic or infrastructure-heavy
repositories.

Our target is not "cheap because lightly tested." It is the opposite:

> We want **stronger verification** than a conventional PR workflow, but we
> want it delivered through fast, deterministic, Rust-native tests that are
> scoped to the actual risk surface of the change.

At our volume, CI spend compounds quickly. A workflow that seems acceptable at
low PR volume becomes unreasonable when many human and agent-authored branches
are iterating in parallel. Cost discipline is therefore part of correctness
discipline: if verification is too expensive, people avoid running it; if it is
cheap, deterministic, and well-scoped, it becomes part of the normal
development loop.

## Cost target

For ordinary PRs, our operating target is:

- **Preferred:** well below `$1` per PR
- **Normal Rust PR target:** materially below `$0.50`
- **Docs / tracking PR target:** pennies
- **High-risk or explicitly labeled PRs:** may use more budget, but only when
  the extra verification is tied to a real risk surface

The `$1` mark is a ceiling, not the goal.

## Why the budget target is aggressive

Our CI budget target is intentionally aggressive — but **not because we want
less verification**.

We believe the opposite. Agentic development requires *more* verification than
traditional software development, and likely more verification than most
current agentic repositories are doing today. More generated branches, more
rapid iteration, more integration edges, and more repeated PR attempts all
increase the need for automated proof. Review alone does not scale to that
volume.

OpenClaw is a useful benchmark **not because we think they are wrong to spend
heavily on verification**, but because their published cost curve shows what
happens when verification demand rises faster than verification efficiency.
They published a Blacksmith runner bill of roughly `$511k`; using commit
volume since February as the denominator, that maps directionally to about
`$20 per commit` on Blacksmith runners alone. Because OpenClaw appears to
squash-merge PRs, commit cost is a reasonable proxy for per-PR cost — though
the figure should be treated as **directional rather than exact**.

That number is not evidence that OpenClaw is doing CI wrong. It is evidence
that **verification demand is rising faster than verification efficiency**.
The lesson is not "verify less." The lesson is that serious agentic
workflows need a better verification cost model. The question is not
verification vs. cost. The question is:

```text
expensive broad verification
   vs.
cheap, scoped, high-frequency verification
```

BitNet-rs is targeting a **different verification economics model**, not less
verification:

- ordinary PRs should stay well below `$1`,
- normal Rust PRs should usually land well below `$0.50`,
- docs / tracking PRs should cost pennies,
- expensive lanes should run when they are relevant, not by default,
- high-cost validation should require explicit labels, main-branch execution,
  nightly execution, release gates, or campaign gates.

The goal is not to spend less by testing less. **The goal is to spend less on
unrelated work so we can afford more verification where the change actually
creates risk.**

> Source note: the OpenClaw comparison is based on their published Blacksmith
> runner cost of approximately `$511k`, divided by observed commit volume
> since February. Because OpenClaw appears to squash-merge PRs, commit count
> is used as a directional proxy for merged-PR count. The figure refers to
> Blacksmith runner cost alone and should not be treated as total CI cost.

## Why Rust and ripr matter

A major reason BitNet-rs is written in Rust is that Rust changes the cost
curve of verification.

Rust lets us push a large share of correctness checking into fast,
deterministic, local validation:

- type and ownership checks at compile time,
- crate-local unit tests,
- feature-gated compile checks,
- small oracle tests,
- bounded property tests,
- deterministic receipt and schema tests,
- precise package and dependency selection.

That means we can run deep checks without needing every ordinary PR to
download models, build external C++ references, start Docker images,
provision macOS runners, or touch live hardware.

### ripr is mutation-testing-lite at static-analysis prices

The CI design principles in this document are adapted from the
[ripr](https://github.com/EffortlessMetrics/ripr) project, which we also use
as tooling. ripr is one of the main reasons this CI strategy is
economically viable. It is **not** generic CI routing.

Coverage tells us code executed. Traditional mutation testing tells us
whether tests fail when a concrete mutant is run. Both are useful, but they
sit at different points on the cost curve:

```text
coverage:
  cheap, but often too weak as an oracle signal
ripr:
  mutation-testing-shaped static exposure signal
mutation testing:
  strong runtime confirmation, but expensive
```

`ripr` is the middle layer. It analyzes the diff, builds mutation-shaped
static probes, and asks whether the changed behavior appears exposed to a
meaningful test discriminator. The PR-time question it answers is:

> For the behavior changed in this diff, do the current tests appear to
> contain a discriminator that would notice if that behavior were wrong?

That is exactly the kind of signal agentic development needs: fast, local,
targeted, and cheap enough to run while a PR is still being drafted.

`ripr` does **not** run mutants, does **not** report `killed` / `survived`
outcomes, and does **not** replace execution-backed mutation testing. It
*shifts mutation-testing-shaped feedback earlier and cheaper*. Full mutation
testing remains valuable for calibration, nightly, and high-risk changes.

### The verification ladder

| Signal                                 |        Cost | Use                                      |
| -------------------------------------- | ----------: | ---------------------------------------- |
| `cargo check` / clippy                 |         low | type / lint correctness                  |
| unit / oracle tests                    |         low | deterministic behavior proof             |
| `ripr`                                 |  low-medium | static mutation-shaped oracle-gap signal |
| property tests                         |      medium | bounded-input confidence                 |
| coverage                               | medium-high | execution surface                        |
| mutation testing                       |        high | runtime adequacy confirmation            |
| crossval / hardware / model validation |        high | external parity and platform proof       |

The strategic claim is:

```text
Rust makes correctness checks fast.
ripr makes oracle gaps visible early.
LEM budgeting makes verification economics explicit.
CI routing spends expensive lanes only where they buy signal.
```

Together, they let us **increase** verification density without letting CI
spend scale linearly with PR volume. The goal is more proof per CI minute —
enough verification for the agentic age, paid for by changing the cost curve
of verification.

## Coverage reporting (Codecov)

Coverage is one signal on the verification ladder: it measures **execution
surface**, not test adequacy or model quality.

### What coverage answers

Coverage tells us: _Did tests exercise this Rust code?_

It does **not** tell us:

- whether tests would catch the wrong behavior (see `ripr`, property tests)
- whether the inference engine produces correct output (see crossval, hardware validation)
- whether GPU backends are correct (see GPU scaffolding status in README)
- whether model predictions are sound (see model validation in `docs/howto/`)

### Coverage in BitNet-rs

Coverage runs are **gated by label or main branch**:

- **PR runs:** only when explicitly labeled `coverage` or `full-ci`
- **Main runs:** automatic after every merge (cost: ~45 LEM, included in
  release validation)
- **Flag:** `rust-cpu` — CPU path execution surface only
- **Threshold policy:** currently informational; will ratchet after baseline
  collection

Coverage artifacts (`coverage.json`, `coverage.txt`, `lcov.info`, `coverage-report`) are stored on every run, enabling trend analysis and per-crate surface inspection.

### Codecov configuration

Codecov integration is configured in `codecov.yml` with:

- **Project status:** tracks overall coverage %
- **Patch status:** tracks changes in PR diffs
- **Comments:** disabled — the GitHub check and Codecov dashboard are the
  primary signals
- **Flags:** scoped to `rust-cpu` for now; GPU flags deferred until backend
  validation is real

### Coverage is not

Coverage is explicitly **not** responsible for:

- CUDA, Metal, OpenCL, ROCm validation (GPU backends are still scaffold)
- model quality or inference correctness (see crossval, hardware receipts)
- test design adequacy (see `ripr` and property tests)
- production inference performance (see hardware validation, runtime receipts)

### Future: baseline and ratchet

After 10–20 runs with real project coverage, we will review:

1. Coverage % distribution across crate types
2. Lowest-covered core paths
3. Runtime cost and flake rate
4. Whether `--ignore-run-fail` is masking relevant failures

Then we will decide whether to tighten thresholds and move from informational
to enforced status. Decisions will be based on observed data, not aspiration.

## Why verification needs to increase

Agentic development changes the shape of risk.

More code can be produced more quickly, but that means more integration edges,
more generated changes, more repeated PR attempts, and more cases where review
alone is not enough. The answer is not to trust less or slow everything down
manually. The answer is to make verification cheaper, sharper, and harder to
bypass.

For BitNet-rs, that means ordinary PRs should run tests that are:

- deterministic,
- local,
- Rust-native,
- model-free,
- hardware-free,
- scoped to the changed crates and their dependents,
- able to catch real regressions before merge.

Expensive validation still matters, but it belongs on the right lanes: main,
nightly, release, campaign, hardware, or explicit labels such as `full-ci`,
`gpu-ci`, `crossval`, `coverage`, or `model-validation`.

## Linux-equivalent minutes (LEM)

We track CI in **Linux-equivalent minutes** because raw wall-clock minutes
hide runner cost. A 10-minute macOS job and a 10-minute Linux job are not
economically equivalent.

LEM gives us one planning unit:

```text
LEM = wall_minutes × runner_multiplier
```

GitHub-hosted runner multipliers (rough planning placeholders):

| Runner            | Multiplier |
| ----------------- | ---------: |
| Linux             |        1.0 |
| Linux + GHA cache |        1.0 |
| GPU Docker (gha)  |       ~6.0 |
| macOS-14 (M1)     |       10.0 |
| Windows           |        2.0 |

Use cases for LEM:

1. **Forecast** PR cost before the PR runs (see `.github/workflows/pr-plan.yml`).
2. **Compare** optional lanes fairly when deciding what to gate behind labels.
3. **Prevent** expensive labels from silently turning ordinary PRs into
   high-cost validation runs.
4. **Calibrate** budgets against observed spend before introducing hard
   budget enforcement.

We deliberately start with LEM **visibility** rather than LEM **enforcement**.
The current repo-level evidence does not yet provide durable enough timing,
queue, cache-hit, failure-rate, flake, and MTTR data to manage CI as a
complete operating system. The path forward is: collect that data, tighten
the default PR lane, move exhaustive lanes to main/nightly/labels, and only
then enforce learned budgets with guardrails.

## CI lane policy

Ordinary PR CI should answer:

> Did this change plausibly break the changed crate, its direct dependents,
> or the canonical CPU path?

It should not answer every question the project can ask.

Broader validation is still required, but it is routed:

| Lane           | Purpose                                       |
| -------------- | --------------------------------------------- |
| Ordinary PR    | Fast, scoped correctness gate                 |
| Main           | Broader integration confidence                |
| Nightly        | Exhaustive / expensive validation             |
| Labeled PR     | Explicit high-risk or campaign validation     |
| Hardware lane  | Live backend and device proof                 |
| Release lane   | Final compatibility, coverage, audit surface  |

This keeps verification strong without making every PR pay for every lens.

## What this does not mean

This policy does **not** mean:

- skipping tests because they are inconvenient,
- hiding failures in non-blocking jobs,
- relying only on happy-path smoke tests,
- avoiding cross-validation,
- avoiding coverage,
- avoiding hardware validation.

It means each check must run where it provides the most value for its cost.

A PR that touches QK256 layout should run QK256 layout fixtures, scalar oracle
tests, and low-case property checks. It should not automatically build every
GPU Docker image.

A tokenizer PR should run tokenizer fixtures. It should not install large
Python stacks or fetch external models unless explicitly requested.

A docs-only PR should run docs and tracking checks. It should not compile the
Rust workspace.

## Operating metric

We optimize for:

```text
proof per CI minute
```

A useful CI minute either:

1. blocks a likely bad merge,
2. proves a meaningful invariant,
3. narrows the cause of a failure,
4. updates a durable signal such as timing, flake, coverage, or compatibility
   state.

CI minutes spent on duplicated checks, no-op jobs, broad unrelated workflows,
unnecessary model downloads, or non-blocking confirmation lanes are waste.

The expected result is a CI system that is **cheaper than conventional broad
PR validation, but stronger where it matters**.

## CI is part of the architecture

CI is not a billing concern bolted on after the system is built. We treat
cost, latency, determinism, and proof strength as design constraints, not
after-the-fact billing concerns. The test rig is part of the machine.

## See also

- [`docs/ci/labels.md`](./labels.md) — cost-aware CI labels and what they
  authorize.
- [`docs/development/validation-ci.md`](../development/validation-ci.md) —
  validation lanes and how they integrate with CI.
- [`docs/development/ci-integration.md`](../development/ci-integration.md) —
  how to wire new checks into the CI portfolio.
- [`docs/reference/validation-gates.md`](../reference/validation-gates.md) —
  the validation gate surface.
- [PR Plan workflow](../../.github/workflows/pr-plan.yml) — advisory per-PR
  Linux-equivalent-minute (LEM) estimate posted to the run summary.
- [ripr](https://github.com/EffortlessMetrics/ripr) — source of the CI
  design principles used here, and the mutation-testing-lite tooling that
  makes this verification ladder economically viable.
