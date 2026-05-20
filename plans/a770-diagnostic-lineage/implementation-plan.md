# A770 Diagnostic Lineage Implementation Plan

Status: proposed
Owner: release/runtime
Created: 2026-05-19
Linked proposal: n/a
Linked specs:
[BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE](../../docs/specs/BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE.md),
[BITNET-SPEC-PR-QUEUE-DISPOSITION](../../docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md)
Linked ADRs:
[BITNET-ADR-0007](../../docs/adr/BITNET-ADR-0007-a770-diagnostics-are-lineage.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: `policy/a770-diagnostic-lineage.toml`

## Goal

Make A770 diagnostic queue handling durable and repeatable without treating the
branch chain as a bulk close target or as product proof.

## Scope

This plan sequences operating-rule work only. It does not port runtime code,
merge A770 runtime fixes, promote A770 support, or reorder today's live queue.

## PR 1 - Lineage Spec, ADR, Plan, And Policy

Purpose: encode the durable rule set before the next runtime salvage slice.

Files:

```text
docs/specs/BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE.md
docs/adr/BITNET-ADR-0007-a770-diagnostics-are-lineage.md
plans/a770-diagnostic-lineage/implementation-plan.md
policy/a770-diagnostic-lineage.toml
docs/specs/INDEX.md
```

Acceptance:

```text
durable diagnostic content is defined
transient diagnostic content is defined
bulk close/reopen/recreate/restack are forbidden by default
closed ancestor and open descendant are lineage states, not dispositions
diagnostic evidence cannot promote support, quality, speed, or residency claims
```

Verification:

```powershell
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

## PR 2 - Checker Integration

Purpose: make the lineage policy machine-checkable after the spec has landed.

Allowed implementation:

```text
load policy/a770-diagnostic-lineage.toml
extend the PR disposition checker to require A770 durable/transient classification
reject A770 close records that cite stale/closed-parent/diagnostic-only as close reasons
require landed successor, duplicate, tracking issue, historical report, or audit record
```

Not allowed:

```text
query live GitHub by default
rerun CI for archaeology
bulk-write labels or close comments
promote diagnostic evidence into support claims
```

## PR 3 - Optional Historical Report

Purpose: capture any historical-only A770 diagnostic evidence that is safe to
close without hiding future work.

Allowed files:

```text
docs/tracking/a770-diagnostic-lineage/*.md
docs/tracking/a770-diagnostic-lineage/*.toml
```

Acceptance:

```text
each closed item cites a valid disposition
future work links to a live successor PR or issue
durable runtime content is not demoted into source material
no generated dashboard is hand-edited
```

## Runtime Salvage Boundary

Runtime salvage starts only after current-main merge candidates are drained and
this lineage rule is available. Pick one narrow dependency slice at a time.

Candidate slices include tied `lm_head`, tokenizer duplicate specials,
effective logits assertion, prompt prefill, I2S trailer scale, act-parallel
QK256 layout, GGUF BPE tokenizer, activation quant oracle, and
activation-quantized QK256 dispatch.

Each runtime salvage PR must identify:

```text
source PR
source commit
current-main successor
files changed
runtime behavior changed
tests proving behavior
claim boundary
source PR disposition
```

## Stop Rules

- Do not process the A770 stack as a batch.
- Do not close because old, stale, behind main, closed parent, or needs restack.
- Do not create replacement PRs without source links and disposition rules.
- Do not run CI for archaeology.
- Do not claim A770 product support from diagnostic lineage.
