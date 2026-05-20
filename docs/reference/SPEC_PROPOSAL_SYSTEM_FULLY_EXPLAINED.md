# The spec/proposal system, fully explained

The repo source-of-truth stack exists to keep each artifact focused on a single
kind of truth and avoid blending roadmap intent, behavior contracts,
architecture decisions, PR sequencing, execution state, and proof in one
monolithic document.

## Core principle

> Do not make every document do every job.

Each artifact answers one question:

- Proposal/PRD: **why**
- Spec: **what behavior must be true**
- ADR: **what durable decision was made**
- Plan: **how work is sequenced into PR-sized steps**
- Active goal: **what is being executed now**
- Support tiers: **what users may believe and what proves it**
- Policy ledgers: **what exceptions and obligations exist**
- Closeout: **what actually happened**

## Source-of-truth stack

```text
Roadmap
  -> Proposal / PRD
    -> Specs
      -> ADRs where needed
        -> Implementation plan
          -> Active goal manifest
            -> Issues / PRs
              -> Proof commands
              -> CI lanes
              -> support-tier updates
              -> policy receipts
                -> Closeout / handoff
```

## Operating guidance for agents

1. Read the active goal manifest.
2. Follow linked plan/spec/ADR artifacts.
3. Execute one ready work item per PR.
4. Run listed proof commands plus `git diff --check`.
5. Update support-tier/policy artifacts only when claims or policy change.
6. Do not hand-edit generated status.
7. Do not invent repository rules; verify commands, lanes, and policies first.

## Non-duplication rule

Keep each fact in one canonical location and link to it from adjacent artifacts.
Avoid copying support-tier claims into specs, CI ownership into plans, or PR
queue state into proposals.

## Minimal rollout order

1. Define docs model and templates.
2. Add document artifact ledger.
3. Add doc-artifact validation checks.
4. Add active goal manifest.
5. Add goal validation checks.
6. Add proposal/spec artifacts for a real lane.
7. Add support tiers and policy ledgers.
8. Wire CI checks from advisory to required.

## Short mental model

```text
Proposal = why.
Spec = what.
ADR = durable decision.
Plan = how.
Active goal = what Codex is doing now.
Support tiers = what users may believe.
Policy ledgers = what exceptions and proof obligations exist.
CI = what proved it.
Closeout = what happened.
```

This system works when artifacts are linked, validated in CI, and prevented from
owning duplicate truths.
