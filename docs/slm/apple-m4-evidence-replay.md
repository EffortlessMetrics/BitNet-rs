# Apple M4 Evidence Replay Bundles

`bitnet mac evidence replay` audits committed Apple M4 evidence bundles. A
bundle manifest records the exact commands, source and binary identity,
model/tokenizer identity, receipt inputs, dashboard outputs, expected regression
result, and claim boundary needed to reproduce or review an evidence refresh.

The command is intentionally dry-run only:

```bash
bitnet mac evidence replay \
  --bundle ci/hardware/apple-m4-mac-mini/2026-05-22T0400Z/evidence-replay/dense-slm-q8-eval/manifest.json \
  --dry-run \
  --json
```

Dry-run replay validates the manifest shape, SHA256-pinned receipt inputs,
dashboard outputs, receipt contracts, and expected regression metadata. It
writes an `apple_m4_evidence_replay_dry_run` receipt and can be checked with:

```bash
bitnet mac receipts-check target/apple-m4-inference-ops/evidence-replay-dry-run.json --json
```

The first committed bundle covers the dense SLM q8 eval-v2 refresh:

```text
ci/hardware/apple-m4-mac-mini/2026-05-22T0400Z/evidence-replay/dense-slm-q8-eval/manifest.json
```

It references:

| Role | Path |
|---|---|
| latest receipt | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` |
| baseline receipt | `ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` |
| dashboard JSON | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/regression-dashboard/regression-dashboard.json` |
| dashboard Markdown | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/regression-dashboard/regression-dashboard.md` |

## Claim Boundary

Replay dry-runs do not run a model, download artifacts, execute the regression
command, or validate uncommitted local artifacts. They only prove that the
committed bundle manifest points at receipt and dashboard inputs that can be
audited with the local receipt validators. Dense SLM and BitNet evidence remain
separate, and replay bundles do not enable BitNet chat, BitNet serve, Metal,
QK256, Neural Engine, MPSGraph, MacBook, broad quality, broad performance, or
speedup claims.
