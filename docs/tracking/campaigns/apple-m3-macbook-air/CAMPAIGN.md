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
- Do not use near-completion timeouts as M3 lane cost control; route the job
  away, choose a smaller profile, or cap healthy selected runs with cushion.
- Never commit model binaries.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M3MBA-001 | merged | Add the campaign control plane and roadmap linkage for the M3 MacBook Air lane. |
| M3MBA-002 | merged | Commit the real M3 Air machine-profile receipt with storage, cache, power, thermal, and visibility fields; merged in #4592. |
| M3MBA-003 | merged | Add or confirm the explicit `apple-m3-air-cpu-neon` receipt label without weakening M4 validation; merged in #4596. |
| M3MBA-012 | merged | Specify the dense SLM harness contract before live M3 smoke runs, including `mac validate`, `mac receipts-check`, synthetic CI receipt evidence, and local-only timing boundaries. |
| M3MBA-013 | merged | Encode the M3 Air CI completion policy so selected long jobs are routed, preflighted, and capped from completed-run evidence instead of ending just before receipts. |
| M3MBA-014 | merged | Preserve M3 Air Mac validate and receipt-check identity before live model work. |
| M3MBA-004A | merged | Mirror the known dense Qwen SLM smoke route on M3 Air. |
| M3MBA-004B | merged | Run the bounded dense Qwen operator profile only after smoke passes. |
| M3MBA-005A | merged | Record official Microsoft 2B I2_S artifact identity, source revision, size, hash, and storage context. |
| M3MBA-005B | merged | Record Microsoft 2B tokenizer/pre-tokenizer authority and bad/no-authority rejection evidence. |
| M3MBA-005C | merged | Decide Microsoft 2B reference output acceptance, rejection, or blocker state. |
| M3MBA-006 | blocked | Evaluate the smaller 0.7B 1bitLLM control candidate after Microsoft 2B is accepted, rejected, or blocked; blocked because the official repo has no GGUF artifact at the recorded revision. |
| M3MBA-007 | blocked | Run only diagnostic TL1/TL2 checks for the 3B candidate and record the I2_S non-claim; blocked because the official repo has no TL1/TL2 GGUF and local free space is below the safe large-candidate floor for the available safetensors shards. |
| M3MBA-008 | merged | Hand the accepted Microsoft 2B I2_S M3 reference-runner artifact to separate M4 strict-proof work without claiming proof in this lane; merged in #5372. |
| M3MBA-009 | merged | Synthesize M3 dense SLM behavior against M4 and SLM CPU evidence after dense smoke/operator evidence exists; merged in #4902. |
| M3MBA-010 | merged | Audit MacBook model-cache retention and cleanup after the first large BitNet download; merged in #4839. |
| M3MBA-011 | merged | Deepen the M3 Air roadmap into operating horizons, stop/go gates, parallel lanes, and local resource budgets. |
| M3MBA-016 | merged | Add bounded M3 Air Metal/MPSGraph backend visibility preflight receipts without model loads, downloads, or performance claims; merged in #5043. |
| M3MBA-017 | merged | Align M3 Air device identity help and rejection surfaces across CLI config and Mac wrappers; merged in #5148. |
| M3MBA-018 | merged | Refresh the roadmap and campaign state after M3MBA-017 and align staged M3 workflow evidence preservation with the selected-long-job policy; merged in #5225. |
| M3MBA-019 | merged | #5388 refreshes the post-handoff roadmap so next M3 work is split into device-model hardening, accuracy comparison, bounded performance, artifact-unblock, and M4 handoff alignment tracks. |
| M3MBA-020 | merged | #5937 encodes the post-handoff execution queue as concrete follow-on work items instead of leaving the lane at a roadmap reset. |
| M3MBA-021 | merged | #5952 extends the shared device/profile model with a structured M3 Air host profile contract and strict unsupported-backend claim boundaries. |
| M3MBA-022 | merged | #6043 adds an M3 Air dense SLM accuracy comparison profile with prompt IDs, scoring policy, and comparable-evidence rules. |
| M3MBA-023 | merged | #57 adds a bounded M3 Air performance profile that uses completed-run timeout provenance and phase artifact retention. |
| M3MBA-024 | merged | #67 preflights secondary BitNet artifact authority, storage, and cleanup gates before any new large download. |
| M3MBA-025 | in progress | Align M3 accepted-artifact metadata with the separate M4 strict-proof checklist without manufacturing M4 proof. |
| M3MBA-026 | merged | #30 hardens server shared-engine receipts so all three configured M3 Air labels survive CUDA-active model metadata without implying live M3 Metal, MPSGraph, M4 proof, or BitNet proof. |

Current focus: `M3MBA-025` is in progress as the next M3 Air handoff-alignment
slice. It maps the accepted Microsoft 2B M3 reference-runner artifact metadata
to separate M4 receipt requirements without turning M3 evidence into M4 proof.
`M3MBA-024` is merged as a no-download secondary BitNet artifact unblock
preflight that refreshes official source availability, tokenizer-authority
needs, storage gates, conversion or third-party approval requirements, and
cleanup planning before any new M3 Air large download.
`M3MBA-022`, `M3MBA-023`, `M3MBA-024`, and `M3MBA-026` are merged. The
Microsoft 2B I2_S artifact is accepted only for the recorded M3 Air BitNet.cpp
reference-runner context and is ready to seed separate M4 strict-proof work with
fresh M4 receipts. `M3MBA-006` and
`M3MBA-007` remain blocked because the official 1bitLLM repositories do not
expose the GGUF artifacts required by their command shapes, and they are not
handoff targets until a future item resolves artifact authority and
storage-safe acceptance requirements. Accuracy, performance, and
handoff-alignment work now use
the structured M3 Air host/profile contract instead of broad Apple Silicon
claims.

## Phase Roadmap

| Phase | Work item(s) | Purpose | Committed output |
|---|---|---|---|
| Foundation | M3MBA-001, M3MBA-002, M3MBA-003, M3MBA-012, M3MBA-013 | Make the M3 Air a receipt-backed evidence source before model timing exists. | Campaign tracker, real machine profile, explicit MacBook backend label, dense harness contract, CI completion policy. |
| Dense control | M3MBA-004A, M3MBA-004B | Mirror the established dense Qwen SLM path on the exact MacBook host in smoke then operator steps. | Smoke/operator receipts, receipt-check output, thermal and power context. |
| BitNet artifact qualification | M3MBA-005A, M3MBA-005B, M3MBA-005C, M3MBA-006, M3MBA-007 | Use the MacBook storage budget to identify, authorize, then accept, reject, or block candidate artifacts. | Candidate reports with source, revision, SHA256, tokenizer authority, prompt output, and cleanup state. |
| Storage hygiene | M3MBA-010 | Keep the MacBook lane usable for large artifacts without hiding local cache state. | Artifact ledger audit with retained/deleted state and free-space floor. |
| Cross-lane synthesis | M3MBA-009 | Compare M3 dense SLM behavior against M4 and SLM CPU evidence without broad claims. | Synthesis report naming comparable receipts and non-comparable gaps. |
| Strict-proof handoff | M3MBA-008 | Convert accepted artifact evidence into separate M4 proof work. | Handoff report only; no manufactured M4 receipt. |
| Post-handoff execution | M3MBA-019, M3MBA-020, M3MBA-021, M3MBA-022, M3MBA-023, M3MBA-024, M3MBA-025, M3MBA-026 | Keep the M3 Air lane moving after the Microsoft 2B handoff with explicit device-model, accuracy, performance, artifact-unblock, and M4 handoff alignment tracks. | Updated roadmap, concrete follow-on item boundaries, and later receipt-backed implementation PRs; no manufactured runtime claim. |

## Post-Handoff Execution Queue

`M3MBA-019` named the tracks. `M3MBA-020` turns those tracks into reviewable
work items so the next PRs have concrete ownership and validation surfaces.

| Order | Work item | Lane slice | Exit evidence |
|---|---|---|---|
| 1 | `M3MBA-021` | Device-model hardening | Structured M3 Air host profile contract in shared device/config surfaces, with strict rejection for unsupported Metal, MPSGraph, Neural Engine, and hidden CPU fallback claims. |
| 2 | `M3MBA-022` | Accuracy comparison | Dense SLM comparison profile with corpus, prompt IDs, scoring policy, receipt fields, and explicit comparable vs non-comparable decisions. |
| 3 | `M3MBA-023` | Bounded performance | Selected M3 Air timing profile with cold/warm separation, power/thermal/storage fields, phase artifacts, and timeout caps derived from completed healthy runs plus cushion. |
| 4 | `M3MBA-024` | Secondary artifact unblock | Official artifact availability, tokenizer authority, storage footprint, conversion or third-party approval needs, and cleanup plan before any new large download. |
| 5 | `M3MBA-025` | M4 handoff alignment | Checklist mapping accepted M3 artifact metadata to separate M4 strict-proof requirements and unsupported claims. |
| 6 | `M3MBA-026` | Server receipt label hardening | CUDA-active model metadata cannot collapse configured M3 Air Metal, MPSGraph, or CPU/NEON labels into generic active-model backend wording or proof claims. |

`M3MBA-022`, `M3MBA-023`, `M3MBA-024`, and `M3MBA-026` are merged. `M3MBA-025`
is the active handoff-alignment slice. Performance follow-ups should remain scoped to the exact M3 Air host and
proof-lane labels now represented by the device/profile contract. Secondary
BitNet artifact work can proceed in parallel with dense SLM accuracy only if it
stays in preflight/reporting paths and does not start a new large local
download.

## Operating Tracks

The lane is managed as four tracks so SLM cross-check work, Mac BitNet artifact
qualification, and M4 strict-proof handoff do not collapse into one claim.

| Track | Items | Advances when | Stops when |
|---|---|---|---|
| Host proof | M3MBA-002, M3MBA-003, M3MBA-012 | Machine facts, cache root, storage budget, explicit M3 Air CPU/NEON receipt labeling, and dense harness contract are committed. | Host facts are missing, cache root is ambiguous, M3 harness contract is missing, or M4 receipt wording would need to be loosened. |
| Dense SLM control | M3MBA-004A, M3MBA-004B, M3MBA-009 | Dense Qwen smoke/operator receipts record model hash, tokenizer metadata, backend label, fallback status, power, thermal, and comparison-grade status. | The M3 Air receipt cannot be distinguished from M4 evidence, the harness contract is missing, or dense SLM evidence lacks model/tokenizer/backend context. |
| Mac BitNet artifacts | M3MBA-005A, M3MBA-005B, M3MBA-005C, M3MBA-006, M3MBA-007, M3MBA-010 | Candidate reports record source, revision, size, SHA256, tokenizer authority, prompt output, and cleanup/retention state. | Dense control has not passed or produced an accepted blocker, storage state is unclear, or tokenizer authority is missing. |
| Strict-proof handoff | M3MBA-008 | An accepted artifact has enough source/hash/tokenizer metadata to open separate M4 strict-proof work. | No artifact is accepted, or the handoff would imply M3 evidence is M4 proof. |

## Roadmap Horizons

The roadmap is planned as four horizons. Each horizon has a concrete exit
condition so the lane can stop, report, and re-plan without blurring evidence.

| Horizon | Scope | Exit condition |
|---|---|---|
| H0: lane readiness | `M3MBA-002`, `M3MBA-003`, `M3MBA-011`, `M3MBA-012`, `M3MBA-013` | The MacBook has a real machine/profile receipt, an explicit receipt label path, a dense harness contract, CI completion policy, and a roadmap that names gates, artifacts, owners, and local resource limits. |
| H1: dense SLM control | `M3MBA-004A`, `M3MBA-004B`, `M3MBA-009` | The known dense Qwen path has M3 Air smoke/operator evidence, or a blocker report explains why M3 dense SLM evidence is not comparison-grade. |
| H2: primary BitNet candidate | `M3MBA-005A`, `M3MBA-005B`, `M3MBA-005C`, `M3MBA-010` | The official Microsoft 2B I2_S artifact is accepted, rejected, or blocked with identity, tokenizer authority, prompt output, and cache-retention evidence. |
| H3: secondary sweep and handoff | `M3MBA-006`, `M3MBA-007`, `M3MBA-008` | Secondary candidates either add bounded evidence or are skipped by policy, then accepted artifacts are handed to separate M4 strict-proof work. |

H0 and H1 protect the SLM lane. H2 and H3 protect the Mac BitNet lane. Do not
start H2 before the dense control path proves that receipts, labels, cache
state, and local runner commands are trustworthy on this exact machine.

## Parallel Lane Policy

Some work can move in parallel, but only where the outputs do not depend on the
same local artifact cache or claim boundary.

| Parallel path | Allowed overlap | Shared dependency |
|---|---|---|
| Machine/profile and receipt-label work | `M3MBA-002` can collect host facts while `M3MBA-003` prepares or confirms synthetic label validation. | The final label must cite the committed machine-profile receipt before live timing is treated as M3 Air evidence. |
| Dense SLM reporting and BitNet planning | `M3MBA-009` can draft comparison tables while `M3MBA-005A` prepares artifact identity commands. | No BitNet download starts until `M3MBA-012` lands and `M3MBA-004A` passes or leaves a blocker accepted by the roadmap. |
| Tokenizer authority and storage audit prep | `M3MBA-005B` can define authority/rejection evidence while `M3MBA-010` defines audit tables. | The audit must use the actual artifact path and free-space data from `M3MBA-005A`. |
| M4 handoff planning and secondary candidates | `M3MBA-008` can draft the handoff template while `M3MBA-006`/`M3MBA-007` remain blocked. | Handoff is opened only after an accepted artifact exists, or closed as no-accepted-artifact evidence. |

The M3 Air is a single fanless local host, so live model execution is serialized.
Only docs, schema, validator, and report-template PRs should run in parallel
with active downloads or timing runs.

## Stop/Go Decisions

| Decision point | Go condition | Stop or re-plan condition |
|---|---|---|
| Start live M3 evidence | `M3MBA-002` records free disk, cache root, power, thermal context when available, and `inference_run=false`. | Host facts are missing, free-space floor is not recorded, or the cache root is ambiguous. |
| Accept M3 Air receipt label | `M3MBA-003` proves `apple-m3-air-cpu-neon` without weakening M4 labels. | The only path is to reuse M4 wording or loosen M4 validation. |
| Accept dense harness contract | `M3MBA-012` names the `mac validate`/`mac receipts-check` contract and synthetic no-model CI receipt expectation. | The live smoke command would rely on undocumented M4-centered behavior or generic CI would need live M3 timing. |
| Accept selected-job CI policy | `M3MBA-013` records routing, preflight, artifact upload, cap sizing, and actuals rules for M3 long jobs. | Cost control still depends on ending selected long jobs near completion instead of routing or shrinking before they start. |
| Proceed to BitNet downloads | Dense smoke passes with model hash, tokenizer metadata, backend label, fallback status, and receipt-check output. | Dense smoke fails without a named blocker, or the receipt cannot distinguish M3 from M4 evidence. |
| Keep a large artifact | SHA256, source revision, tokenizer authority, free-space before/after, and cleanup status are recorded. | The artifact lacks authority, exceeds the storage budget, or cannot be reproduced from source/revision. |
| Open M4 strict-proof work | A candidate is accepted by the M3 Air reference context and has handoff-ready source/hash/tokenizer metadata. | All candidates are rejected or blocked; close `M3MBA-008` with no-accepted-artifact evidence. |

## Secondary Candidate Depth

The smaller and diagnostic candidates are not a vague backlog. They have a
preflight shape even while upstream filenames and tokenizer authority are still
pending.

| Candidate | Owner | Preflight before download | Decision output |
|---|---|---|---|
| 0.7B `1bitLLM/bitnet_b1_58-large` | `M3MBA-006` | Resolve exact GGUF filename, revision, size estimate, tokenizer files, runner command, and expected ARM route. | Accept, reject, or block as a smaller control artifact, with source/hash/tokenizer evidence and cleanup status. |
| 3B `1bitLLM/bitnet_b1_58-3B` | `M3MBA-007` | Resolve only TL1/TL2 diagnostic files and confirm I2_S remains unsupported before any run. | Diagnostic report only; no I2_S or local-answer proof claim. |
| Falcon-E secondary family | deferred | Keep in the shared candidate matrix until Microsoft and 1bitLLM behavior is understood. | Future item only if the primary family leaves enough evidence and storage headroom. |

`M3MBA-008` waits for the secondary-candidate decision surface, not just the
Microsoft 2B path. If Microsoft 2B is accepted early, the handoff can cite that
artifact as the proof target; if it is rejected or blocked, the handoff must wait
for `M3MBA-006`/`M3MBA-007` or close with no accepted artifact.

## Local Resource Budget

The M3 Air lane can use local storage aggressively, but every large-artifact PR
must preserve a clear rollback path.

| Resource | Policy |
|---|---|
| Free-space floor | Record free space before and after each large artifact action. Re-plan before starting a secondary candidate if the post-cleanup floor is not explicitly acceptable in `M3MBA-010`. |
| Cache root | Use the machine-profile cache root unless a PR records a deliberate temporary override and cleanup plan. |
| Artifact retention | Retain only artifacts needed for the next accepted/rejected/blocker decision; otherwise record deletion plus enough source/revision/hash metadata to reproduce. |
| Power and thermal context | Dense operator and large artifact runs record charger/battery state and thermal state when macOS exposes it; missing thermal telemetry is reported as unavailable, not inferred. |
| CI boundary | CI validates docs, schemas, command wiring, synthetic receipts, and completion-policy rules. Live downloads and timing runs remain local M3 Air evidence or explicitly labeled/scheduled Apple-hardware lanes with healthy-run caps. |

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
| M3MBA-012 | Dense harness contract report section in `docs/apple-silicon/m3-macbook-air-roadmap.md`, with synthetic no-model CI receipt expectation. |
| M3MBA-013 | CI completion policy section in `docs/apple-silicon/m3-macbook-air-roadmap.md` and shared CI policy alignment in `docs/ci/cost-and-verification-policy.md`. |
| M3MBA-017 | CLI help/error surfaces and roadmap state proving all three M3 Air labels remain distinct while unsupported Metal/MPSGraph model inference is rejected. |
| M3MBA-018 | Roadmap, campaign, MacBook lane, and staged workflow refresh showing the post-device-model stack, the remaining M3MBA-007/M3MBA-008 sequence, comparison-profile inputs, and selected-long-job phase evidence uploads. |
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
3. `M3MBA-012` defines the dense harness contract before live smoke.
4. `M3MBA-013` encodes selected-job CI completion rules before any scheduled or labeled M3 long job is relied on for receipts.
5. `M3MBA-004A` runs the dense Qwen smoke control path before any BitNet artifact sweep.
6. `M3MBA-004B` runs the bounded dense Qwen operator profile only after smoke passes.
7. `M3MBA-005A` records official Microsoft 2B I2_S source, revision, size, hash, and storage context.
8. `M3MBA-005B` records tokenizer/pre-tokenizer authority and rejection evidence.
9. `M3MBA-005C` records prompt-suite reference output and accepts, rejects, or blocks the candidate.
10. `M3MBA-010` audits cache retention before secondary large downloads.
11. `M3MBA-006` evaluates the smaller 0.7B control candidate.
12. `M3MBA-009` summarizes M3 dense SLM behavior against comparable M4 and SLM CPU evidence.
13. `M3MBA-007` keeps 3B TL routes diagnostic-only.
14. `M3MBA-008` opens M4 proof handoff only for accepted artifacts.

Do not skip the dense control path and jump straight to BitNet downloads. The
dense run proves the MacBook runner, receipts, cache policy, backend labels, and
operator flow before larger artifacts consume the local storage budget.

`M3MBA-010` has recorded retained/deleted artifacts, free-space before/after,
and enough headroom for the next serialized large-candidate step. `M3MBA-006`
and `M3MBA-007` still need their own preflight, hash, cleanup/retention, and
claim-boundary evidence before they can merge.

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

M3 live and scheduled jobs should fail before expensive work when disk, cache,
runner, power, thermal, model, tokenizer, or receipt prerequisites are missing,
and should upload phase artifacts even when a later phase fails. Once a bounded
M3 profile is intentionally selected, its cap should allow a healthy run to
finish with cushion. If that is too costly, choose a smaller profile, manual
evidence, or a scheduled lane rather than cutting the job off near completion
and spending the same CI again.

`M3MBA-013` owns the first explicit CI-design pass for this rule. It should
leave reviewers with a concrete checklist for selected long M3 jobs: route
irrelevant PRs away, preflight before downloads or model runs, serialize one
large artifact at a time, upload partial phase receipts, derive caps from
successful completed runs plus cushion, and keep timeout/cancellation actuals
out of healthy-runtime percentiles.

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
