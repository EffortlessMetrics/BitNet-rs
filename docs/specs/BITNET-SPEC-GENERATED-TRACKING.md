# BITNET-SPEC-GENERATED-TRACKING: Generated Tracking Conflict Handling

Status: proposed
Owner: release/ci
Created: 2026-05-19
Linked proposal: n/a
Linked specs:
[BITNET-SPEC-0001](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md),
[BITNET-SPEC-PR-QUEUE-DISPOSITION](BITNET-SPEC-PR-QUEUE-DISPOSITION.md)
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: `policy/generated-tracking.toml`

## Purpose

Generated campaign dashboards are derived status. They must not become a hand
authored queue, claim ledger, or conflict-resolution surface. This spec defines
how agents and maintainers handle generated tracking files when branches,
rebases, campaign rows, or queue work collide.

The rule is direct: fix the source manifest, event, generator, or checker, then
regenerate. Do not edit generated output to make a conflict disappear.

## Source-Of-Truth Authorities

Generated tracking truth lives in:

- campaign `active.toml` manifests;
- campaign `events/*.toml` lifecycle records;
- tracker infrastructure generator and checker code;
- this spec;
- `policy/generated-tracking.toml`;
- [Tracker Model](../tracking/TRACKER_MODEL.md);
- [Source Of Truth And Claim Boundaries](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md).

Generated files under `docs/tracking/generated/` and
`docs/tracking/campaigns/*/generated/` are derived views. They can be committed
when the generator produces them, but they do not own campaign state.

## Generated Files Are Not Manually Authored

Agents and maintainers must not hand-edit generated tracking files to:

- remove a lane row;
- mark a work item complete;
- hide a skipped or blocked lane;
- resolve a rebase conflict by choosing one branch's rendered table;
- change claim wording;
- close or supersede source PR work;
- make campaign doctor pass without updating the source.

If generated output is stale, run the generator. If generated output is wrong,
fix the manifest, event source, generator, or checker.

## Conflict Handling

When a generated tracking file conflicts:

1. Preserve every active row, lane, and work item that remains true.
2. Inspect the relevant campaign `active.toml` and `events/*.toml` sources.
3. Update the source manifest or event records, not the generated table.
4. Run the generator or checker named by the affected campaign.
5. Review the generated diff for source-faithful output.
6. Do not hand-delete another lane's row to resolve the conflict.

If two branches claim ownership of the same campaign item, stop the merge or
port and resolve ownership in the source manifests before generating output.

## Allowed Generated Diffs

A generated tracking diff is allowed only when:

- it was produced by the campaign generator;
- it matches the committed campaign manifests and events;
- it preserves active rows that remain true;
- it does not promote product, support, hardware, runtime, or PR disposition
  claims beyond the source data;
- the PR records the generator or checker command used.

## Acceptance Examples

| Case | Required handling |
| --- | --- |
| Generated dashboard is stale after an `active.toml` edit | Run generator and commit generated output |
| Generated dashboard conflicts during rebase | Reconcile manifest/event sources, then regenerate |
| Another branch added a valid campaign row | Preserve the row unless the source says it closed |
| A row looks inconvenient for queue burn-down | Keep it; queue disposition requires source proof |
| Generator omits a true active row | Fix generator or source parsing, then regenerate |
| Hand edit would make campaign doctor pass | Reject; fix source or generator |

## Proof Commands

Current generated-tracking validation:

```bash
cargo run --locked -p xtask --no-default-features -- campaign generate --check
cargo run --locked -p xtask --no-default-features -- campaign doctor
git diff --check
```

When a PR intentionally updates generated tracking, it must run the campaign
generator without `--check` before the check form.

## Non-Goals

- Do not change campaign generator behavior in this spec.
- Do not change today's campaign state.
- Do not edit generated dashboards in this PR.
- Do not encode current open PR order.
- Do not use generated tracking as proof of support, speed, residency, or
  product readiness without the underlying proof surface.
