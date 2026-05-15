# Apple M4 Inference Ops Campaign

Campaign ID: `apple-m4-inference-ops`

Status: complete

## Objective

Turn the completed Apple M4 dense SLM and BitNet proof surfaces into a durable
operator layer: status, report inventory, advisory refresh, regression
dashboarding, disk/cache posture, and an operator envelope v2 with explicit
claim boundaries.

## Scope Boundary

This is an M4 Mac mini operations campaign. It consumes existing dense SLM and
BitNet evidence but does not reopen completed runtime proof campaigns. It must
not use dense Qwen evidence as BitNet evidence, and it must not enable BitNet
chat or serve.

No generic required PR check may download models, run live M4 inference, run
long resident soaks, or claim broad performance. Live M4 refreshes stay local,
advisory, scheduled, or release-only.

## End State

- `bitnet mac status` gives operators one receipt-backed readiness summary.
- Advisory/nightly report refresh manifests cover dense SLM and BitNet report
  families without mixing claim scopes.
- Regression dashboard artifacts compare matching reports over time for
  quality, TTFT, input/output/decode throughput, memory, and reliability drift.
- Operator envelope v2 maps supported M4 commands to model/tokenizer/backend/
  fallback/timing/memory/token-ID receipt requirements and claim boundaries.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-INF-OPS-001 | merged | PR #4960 added `bitnet mac status` with an `apple_m4_inference_status` receipt covering dense SLM, BitNet, disk/cache, report inventory, commands, and claim boundaries. |
| M4-INF-OPS-002 | merged | PR #4963 added advisory/nightly report refresh manifest generation for committed M4 dense SLM and BitNet report families. |
| M4-INF-OPS-003 | merged | PR #4967 added compact regression dashboard artifacts across dense SLM and BitNet reports while keeping evidence families separate. |
| M4-INF-OPS-004 | merged | PR #4969 published operator envelope v2 mapping commands to receipts, gates, and unsupported claims. |

## Review Policy

Each PR owns one item. Parser, schema, fixture, and receipt validation belong in
generic CI. Live model execution and timing refreshes belong in local,
advisory, scheduled, or release lanes.
