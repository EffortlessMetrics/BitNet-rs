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
| M3MBA-001 | merged | Add the campaign control plane and roadmap linkage for the M3 MacBook Air lane. |
| M3MBA-002 | proposed | Commit the real M3 Air machine-profile receipt with storage, cache, power, thermal, and visibility fields. |
| M3MBA-003 | proposed | Add or confirm the explicit `apple-m3-air-cpu-neon` receipt label without weakening M4 validation. |
| M3MBA-004A | proposed | Mirror the known dense Qwen SLM smoke route on M3 Air. |
| M3MBA-004B | proposed | Run the bounded dense Qwen operator profile only after smoke passes. |
| M3MBA-005A | proposed | Record official Microsoft 2B I2_S artifact identity, source revision, size, hash, and storage context. |
| M3MBA-005B | proposed | Record Microsoft 2B tokenizer/pre-tokenizer authority and bad/no-authority rejection evidence. |
| M3MBA-005C | proposed | Decide Microsoft 2B reference output acceptance, rejection, or blocker state. |
| M3MBA-006 | proposed | Evaluate the smaller 0.7B 1bitLLM control candidate after Microsoft 2B is accepted, rejected, or blocked. |
| M3MBA-007 | proposed | Run only diagnostic TL1/TL2 checks for the 3B candidate and record the I2_S non-claim. |
| M3MBA-008 | proposed | Hand accepted artifacts to separate M4 strict-proof items without claiming proof in this lane. |
| M3MBA-009 | proposed | Synthesize M3 dense SLM behavior against M4 and SLM CPU evidence after dense smoke/operator evidence exists. |
| M3MBA-010 | proposed | Audit MacBook model-cache retention and cleanup after the first large BitNet download. |

## Phase Roadmap

| Phase | Work item(s) | Purpose | Committed output |
|---|---|---|---|
| Foundation | M3MBA-001, M3MBA-002, M3MBA-003 | Make the M3 Air a receipt-backed evidence source before model timing exists. | Campaign tracker, real machine profile, explicit MacBook backend label. |
| Dense control | M3MBA-004A, M3MBA-004B | Mirror the established dense Qwen SLM path on the exact MacBook host in smoke then operator steps. | Smoke/operator receipts, receipt-check output, thermal and power context. |
| BitNet artifact qualification | M3MBA-005A, M3MBA-005B, M3MBA-005C, M3MBA-006, M3MBA-007 | Use the MacBook storage budget to identify, authorize, then accept, reject, or block candidate artifacts. | Candidate reports with source, revision, SHA256, tokenizer authority, prompt output, and cleanup state. |
| Storage hygiene | M3MBA-010 | Keep the MacBook lane usable for large artifacts without hiding local cache state. | Artifact ledger audit with retained/deleted state and free-space floor. |
| Cross-lane synthesis | M3MBA-009 | Compare M3 dense SLM behavior against M4 and SLM CPU evidence without broad claims. | Synthesis report naming comparable receipts and non-comparable gaps. |
| Strict-proof handoff | M3MBA-008 | Convert accepted artifact evidence into separate M4 proof work. | Handoff report only; no manufactured M4 receipt. |

## Milestone Gates

The lane advances only when the previous gate leaves durable committed evidence.
Local cache state, terminal output, and downloaded model files are not enough.

| Gate | Required before advancing |
|---|---|
| Machine readiness | Real profile receipt records model identifier, chip, core split, memory, macOS version, cache root, free disk, power, thermal state when available, CPU/NEON visibility, Metal visibility, MPSGraph visibility when available, and `inference_run=false`. |
| Receipt label readiness | `apple-m3-air-cpu-neon` or a documented successor is accepted without weakening `apple-m4-cpu-neon` validation. |
| Dense smoke readiness | Dense Qwen smoke receipt passes validation or leaves a blocker report with backend, fallback, model hash, tokenizer metadata, power, thermal, and storage context. |
| Dense operator readiness | Dense Qwen operator receipt exists only after smoke passes, and records allocation-audit context, repeat count, token budget, thermal/power state, and comparison-grade vs diagnostic status. |
| Microsoft identity readiness | Official Microsoft 2B I2_S evidence records source revision, filename, size, SHA256, cache root, free-space before/after, and shared artifact-gate references. |
| Microsoft authority readiness | Tokenizer/pre-tokenizer authority and bad/no-authority rejection evidence are recorded before reference output is treated as acceptance evidence. |
| BitNet screening readiness | Official Microsoft 2B I2_S reference outputs record answer-gate result or failing prompt IDs and cleanup status. |
| Handoff readiness | Only accepted artifacts are named in separate M4 strict-proof work, and the handoff requires fresh M4 receipts before any M4 claim. |

## Output Map

| Work item | Primary output |
|---|---|
| M3MBA-002 | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/machine-profile.json` |
| M3MBA-003 | Receipt validator/test fixture or documented schema evidence proving `apple-m3-air-cpu-neon` support without weakening `apple-m4-cpu-neon`. |
| M3MBA-004A | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json` and `docs/reports/apple-silicon-macbook-m3-air-qwen-smoke.md` |
| M3MBA-004B | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-operator.json` and `docs/reports/apple-silicon-macbook-m3-air-qwen-operator.md` |
| M3MBA-005A | `docs/reports/apple-silicon-macbook-m3-air-microsoft-2b-i2s.md` identity/hash section |
| M3MBA-005B | `docs/reports/apple-silicon-macbook-m3-air-microsoft-2b-i2s.md` tokenizer authority section |
| M3MBA-005C | `docs/reports/apple-silicon-macbook-m3-air-microsoft-2b-i2s.md` reference output decision section |
| M3MBA-006 | `docs/reports/apple-silicon-macbook-m3-air-1bitllm-07b.md` |
| M3MBA-007 | `docs/reports/apple-silicon-macbook-m3-air-3b-tl-diagnostic.md` |
| M3MBA-008 | `docs/reports/apple-silicon-macbook-m3-air-m4-proof-handoff.md` |
| M3MBA-009 | `docs/reports/apple-silicon-macbook-m3-air-slm-synthesis.md` |
| M3MBA-010 | `docs/reports/apple-silicon-macbook-m3-air-storage-audit.md` |

## Tactical Order

The first live sequence is:

1. `M3MBA-002` records host facts and free disk without inference.
2. `M3MBA-003` adds or confirms the explicit M3 Air CPU/NEON receipt label.
3. `M3MBA-004A` runs the dense Qwen smoke control path before any BitNet artifact sweep.
4. `M3MBA-004B` runs the bounded dense Qwen operator profile only after smoke passes.
5. `M3MBA-005A` records official Microsoft 2B I2_S source, revision, size, hash, and storage context.
6. `M3MBA-005B` records tokenizer/pre-tokenizer authority and rejection evidence.
7. `M3MBA-005C` records prompt-suite reference output and accepts, rejects, or blocks the candidate.
8. `M3MBA-010` audits cache retention before secondary large downloads.
9. `M3MBA-006` evaluates the smaller 0.7B control candidate.
10. `M3MBA-009` summarizes M3 dense SLM behavior against comparable M4 and SLM CPU evidence.
11. `M3MBA-007` keeps 3B TL routes diagnostic-only.
12. `M3MBA-008` opens M4 proof handoff only for accepted artifacts.

Do not skip the dense control path and jump straight to BitNet downloads. The
dense run proves the MacBook runner, receipts, cache policy, backend labels, and
operator flow before larger artifacts consume the local storage budget.

`M3MBA-010` is a blocker for secondary large downloads. `M3MBA-006` and
`M3MBA-007` may proceed only after the storage audit records retained/deleted
artifacts, free-space before/after, and whether the lane has headroom for
another candidate.

`M3MBA-008` is conditional on an accepted artifact. If the Microsoft path and
secondary candidates are rejected or blocked, close the handoff with a
no-accepted-artifact report instead of opening a proof item.

## Report Minimum Sections

M3 evidence reports should use consistent headings so reviewers can compare
machine, dense SLM, and BitNet artifact PRs without reconstructing local state.
Every report should include:

```text
Work item and claim boundary
Host profile and power/thermal context
Commands run and exit status
Artifact identity, source revision, size, SHA256, and cache root when relevant
Tokenizer and pre-tokenizer authority when relevant
Receipt or prompt-suite outputs, including failing prompt IDs when relevant
Storage before/after and cleanup or retention status
Comparison-grade vs diagnostic-only decision
Next dependency unblocked or blocker named
```

## Authority And Dependencies

`apple-m3-macbook-air` is the live execution authority for this MacBook. New
M3 Air machine-profile, dense SLM, and large-artifact evidence should be opened
here first.

`apple-silicon-macbook` remains the umbrella and historical MacBook campaign.
It should point to this campaign for new M3 Air execution work instead of
duplicating live items.

`apple-bitnet-artifact-sweep` and `model-artifacts` remain the shared artifact
and answer-gate authorities. M3 BitNet items must consult
`docs/model-artifacts/ANSWER_ARTIFACT_GATE.md` and
`ci/model-artifacts/model-kernel-compatibility.toml` before turning a local
MacBook run into candidate acceptance, rejection, or handoff evidence.

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
