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

By comparison, OpenClaw has been cited as an example of how expensive modern
agentic CI can become: their published Blacksmith spend of approximately
`$511k` mapped to roughly `$20` per commit (squash-merged PRs) since February.
Whether or not that number is typical, it illustrates the failure mode
clearly. In high-volume agentic workflows, broad CI that feels acceptable at
low volume becomes a material operating cost very quickly.

That comparison is not included to criticize a different repository or runner
choice. It is a reminder of the operating reality: in the agentic age, PR
volume rises, verification demand rises, and default CI economics can break
quickly if every branch triggers broad, expensive validation.

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

## Why Rust is central to the cost model

A major reason BitNet-rs is written in Rust is that Rust lets us move a large
amount of verification into fast compile-time and unit-level checks.

Rust gives us:

- strong type and ownership guarantees before runtime,
- fast deterministic unit tests,
- precise crate-level test selection,
- feature-gated compile checks,
- lightweight property and oracle tests,
- reliable local reproduction of CI failures.

That changes the economics. We can run deep correctness checks without needing
to download large models, build external C++ references, start Docker images,
or provision special hardware for every ordinary PR.

The goal is not fewer tests. The goal is **more proof per CI minute**.

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
