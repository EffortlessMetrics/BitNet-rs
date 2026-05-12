# Apple M3 MacBook Air Campaign

Campaign ID: `apple-m3-macbook-air`

Status: active

## Objective

Turn the available M3 MacBook Air into a disciplined Apple Silicon lane for
machine-profile evidence, dense SLM cross-checks, large BitNet artifact
qualification, and M4 strict-proof handoff planning without converting MacBook
receipts into M4 Mac mini performance or BitNet local-answer claims.

## Why This Exists

The existing M3 MacBook Air roadmap names the right sequence, but the lane needs
the same campaign-local control plane used by the other active hardware efforts.
The MacBook has enough local storage for larger candidate artifacts, but it is a
mobile, fanless Apple Silicon host. Its receipts need power, thermal, storage,
cache, backend, fallback, and cleanup context before any timing or artifact
decision can be trusted.

This campaign sits between the completed M4 dense SLM lanes and the Apple BitNet
artifact sweep. It proves what the MacBook observed, then hands accepted
artifacts to separate strict proof items.

## End State

- A real M3 MacBook Air machine-profile receipt is committed with
  `inference_run=false`.
- M3 Air dense Qwen receipts use an explicit MacBook backend label and record
  power, thermal, storage, model, tokenizer, and fallback context.
- Larger BitNet candidate downloads are accepted, rejected, or blocked with
  source, revision, size, SHA256, tokenizer authority, prompt output, and
  cleanup status.
- Accepted artifacts feed separate M4 Mac mini strict Apple CPU/NEON proof
  items; M3 evidence does not become M4 evidence by wording.
- The MacBook lane remains storage-aware and never commits model binaries.

## Hard Constraints

- This is the Apple M3 MacBook Air lane, not the M4 Mac mini product,
  performance, or strict-proof lane.
- Do not claim BitNet local-answer quality from dense Qwen SLM receipts.
- Do not claim M4 Mac mini performance, broad Apple Silicon performance, QK256
  support, full Apple Metal inference, Neural Engine execution, or MPSGraph model
  inference from this lane.
- Do not weaken existing M4 receipt checks to make M3 receipts fit; add the
  smallest MacBook-specific label or validation path instead.
- Do not add live model downloads, large artifact sweeps, or hardware timing
  runs to generic required CI.
- Never commit model binaries.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M3MBA-001 | in_progress | Add the campaign control plane and roadmap linkage for the M3 MacBook Air lane. |
| M3MBA-002 | proposed | Commit the real M3 Air machine-profile receipt with storage, cache, power, thermal, and visibility fields. |
| M3MBA-003 | proposed | Add or confirm the explicit `apple-m3-air-cpu-neon` receipt label without weakening M4 validation. |
| M3MBA-004 | proposed | Mirror the known dense Qwen SLM route on M3 Air with smoke and operator receipts. |
| M3MBA-005 | proposed | Qualify the official Microsoft 2B I2_S BitNet artifact on M3 Air under tokenizer authority. |
| M3MBA-006 | proposed | Evaluate the smaller 0.7B 1bitLLM control candidate after Microsoft 2B is accepted, rejected, or blocked. |
| M3MBA-007 | proposed | Run only diagnostic TL1/TL2 checks for the 3B candidate and record the I2_S non-claim. |
| M3MBA-008 | proposed | Hand accepted artifacts to separate M4 strict-proof items without claiming proof in this lane. |

## Review Policy

Each PR should own one work item and should leave either passing evidence or a
blocker report. Hardware and artifact PRs must record the exact command, host,
artifact identity when relevant, receipt path, cleanup status, and claim
boundary. A skipped live run is acceptable only when the blocker is explicit and
the next smallest fix is named.

Generic CI should cover tracker validity, schema shape, parser behavior, and
synthetic receipt checks. Live M3 model runs, large downloads, timing receipts,
and artifact sweeps are local or scheduled Apple-hardware evidence, not ordinary
CI requirements.

## Claim Boundary

Do not claim:

```text
BitNet local-answer quality from dense Qwen evidence
M4 Mac mini performance from MacBook timing
QK256 support on Apple Silicon
full Apple Metal inference
Neural Engine execution
MPSGraph model inference
broad Apple Silicon performance
```

Do claim only:

```text
M3 MacBook Air machine facts, dense SLM cross-checks, artifact decisions,
or handoff readiness when the named receipt or report provides that evidence
```
