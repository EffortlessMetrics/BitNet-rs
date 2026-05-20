# M3 MacBook Air Lane Roadmap

The M3 MacBook Air lane is the live Apple Silicon MacBook lane for larger
artifact sweeps and dense SLM cross-checks. It is separate from the completed
M4 Mac mini product and performance campaigns.

Committed host facts from `M3MBA-002`:

```text
machine = MacBook Air
model_identifier = Mac15,13
chip = Apple M3
cpu_cores = 8
performance_cores = 4
efficiency_cores = 4
memory = 16 GB
available_repo_volume_space = about 59 GiB on 2026-05-13
receipt = ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/machine-profile.json
```

These facts make the machine suitable for storage-conscious BitNet artifact
qualification and dense SLM Apple CPU/NEON cross-checks. They do not create
M4 Mac mini performance evidence and do not prove BitNet local-answer quality.

The campaign-local tracker for this lane is:

```text
docs/tracking/campaigns/apple-m3-macbook-air/
```

Use that tracker as the source of truth for work-item state, allowed paths,
validation commands, and claim boundaries. This roadmap remains the operator
sequence and evidence rubric.

## Lane Roles

M3 MacBook Air:

```text
live mobile Apple Silicon cross-reference
large-artifact download and hash qualification when storage allows
dense Qwen SLM behavior comparison against established Mac receipts
BitNet candidate reference-runner screening before M4 strict proof handoff
```

M4 Mac mini:

```text
stable Apple Silicon product/performance proof lane
strict M4 CPU/NEON receipts
published dense SLM warm-session envelope
phase-scoped Metal evidence
```

BitNet artifact sweep:

```text
model and tokenizer authority qualification
reference-runner output sanity
candidate acceptance or rejection
handoff only after coherent reference output is recorded
```

Related control files:

```text
ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml
docs/apple-silicon/bitnet-candidate-matrix.md
docs/apple-silicon/apple-bitnet-artifact-sweep.md
docs/tracking/campaigns/apple-bitnet-artifact-sweep/
```

## Operating Tracks

The M3 Air work is not one roadmap item. It is four coordinated tracks with
different evidence standards:

| Track | Owner items | What moves here | What stops it |
|---|---|---|---|
| Host proof | `M3MBA-002`, `M3MBA-003`, `M3MBA-012` | Machine profile, storage budget, cache root, Apple CPU/NEON receipt label, dense harness contract, and no-inference proof. | Missing host facts, ambiguous cache root, missing M3 harness contract, or any need to weaken M4 receipt wording. |
| Dense SLM control | `M3MBA-004A`, `M3MBA-004B`, `M3MBA-009` | Qwen smoke/operator receipts and cross-lane SLM comparison against M4 and CPU evidence. | No explicit M3 backend label, missing harness contract, missing model/tokenizer hash, fallback ambiguity, or thermal/power context gaps. |
| Mac BitNet artifacts | `M3MBA-005A`, `M3MBA-005B`, `M3MBA-005C`, `M3MBA-006`, `M3MBA-007`, `M3MBA-010` | Larger local downloads, source revisions, SHA256, tokenizer authority, reference-runner output, and cleanup state. | Dense control has not passed or produced an accepted blocker, storage headroom is unclear, or tokenizer authority is missing. |
| Strict-proof handoff | `M3MBA-008` | Convert accepted artifact evidence into separate M4 proof work. | No accepted artifact exists, or the proposed handoff would imply M3 timing is M4 evidence. |

The first two tracks protect the SLM lane. The third track uses the MacBook's
available storage for BitNet qualification. The fourth track is deliberately
thin: proof still belongs to a fresh M4 item with fresh M4 receipts.

## Roadmap Summary

| Phase | Work item | Outcome | Evidence |
|---:|---|---|---|
| 0 | `M3MBA-001` | Campaign tracker, roadmap linkage, and claim boundaries | `docs/tracking/campaigns/apple-m3-macbook-air/` |
| 1 | `M3MBA-002` | Real M3 Air machine profile, no inference | `ci/hardware/apple-silicon-macbook/.../machine-profile.json` |
| 2 | `M3MBA-003` | Explicit M3 Air receipt label | Validator or label evidence for `apple-m3-air-cpu-neon` |
| 3 | `M3MBA-011` | Roadmap depth and stale-reference cleanup | Operating horizons, stop/go decisions, parallel lane policy, resource budget, and M3MBA authority alignment |
| 4 | `M3MBA-012` | Dense SLM harness contract | `mac validate`/`mac receipts-check` contract, synthetic CI receipt plan, and local-only timing boundary |
| 5 | `M3MBA-013` | CI completion policy for selected long jobs | Routing, preflight, artifact upload, cap sizing, and CI actuals policy |
| 6 | `M3MBA-014` | M3 Air Mac validate receipt bridge | Runtime path preserves M3 Air backend and machine identity before live smoke |
| 7 | `M3MBA-015` | M3 Air device-model label completion | Explicit M3 Air Metal/MPSGraph/CPU identities without runtime claims |
| 8 | `M3MBA-016` | M3 Air backend visibility and bounded preflight | Metal/MPSGraph visibility receipts without model loads, downloads, or performance claims |
| 8.5 | `M3MBA-017` | M3 Air device-model help and rejection alignment | All M3 Air labels are discoverable while unsupported Metal/MPSGraph model inference stays rejected |
| 9 | `M3MBA-004A` | Dense Qwen SLM smoke mirror on M3 Air | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json` plus report |
| 10 | `M3MBA-004B` | Dense Qwen SLM operator profile on M3 Air | Operator receipt with allocation-audit, thermal, power, and comparison-grade context |
| 11 | `M3MBA-005A` | Official Microsoft 2B I2_S artifact identity | Source revision, filename, size, SHA256, cache root, storage context |
| 12 | `M3MBA-005B` | Official Microsoft 2B I2_S tokenizer authority | Tokenizer/pre-tokenizer authority and bad/no-authority rejection evidence |
| 13 | `M3MBA-005C` | Official Microsoft 2B I2_S reference output decision | Reference-runner report, answer-gate result or failing prompt IDs, cleanup state |
| 14 | `M3MBA-010` | Storage and cache hygiene audit | Artifact ledger audit with retained/deleted model state and free-space floor |
| 15 | `M3MBA-006` | Smaller 0.7B BitNet control candidate | `docs/reports/apple-silicon-macbook-m3-air-1bitllm-07b.md` |
| 16 | `M3MBA-009` | M3 SLM lane synthesis | Cross-lane report comparing M3 Air dense SLM receipts against M4 and 8250U SLM evidence |
| 17 | `M3MBA-007` | 3B TL1/TL2 diagnostic only | `docs/reports/apple-silicon-macbook-m3-air-3b-tl-diagnostic.md` |
| 18 | `M3MBA-008` | M4 strict-proof handoff for accepted artifacts | `docs/reports/apple-silicon-macbook-m3-air-m4-proof-handoff.md` |

## Current PR Stack

`M3MBA-001`, `M3MBA-002`, `M3MBA-003`, `M3MBA-011`, `M3MBA-012`,
`M3MBA-013`, `M3MBA-014`, `M3MBA-004A`, `M3MBA-004B`, `M3MBA-005A`,
`M3MBA-005B`, `M3MBA-005C`, `M3MBA-010`, `M3MBA-009`, `M3MBA-015`,
`M3MBA-016`, and `M3MBA-017` are
merged. The lane now has a tracker, real machine profile, explicit M3 Air
CPU/NEON receipt label, roadmap depth, dense harness contract, selected-long-job
CI completion posture, M3 validate bridge, dense Qwen smoke/operator evidence,
Microsoft 2B I2_S identity, tokenizer authority, reference-output decision,
cache-retention audit, dense SLM cross-lane synthesis, and distinct M3 Air
Metal/MPSGraph device identities with operator-facing help and rejection
surfaces aligned.

The handoff stack is now closed. `M3MBA-018` kept the roadmap and campaign
state current after the device-model work, `M3MBA-007` closed as blocked for
the 3B TL diagnostic surface, and `M3MBA-008` handed the accepted Microsoft 2B
I2_S M3 Air BitNet.cpp reference-runner evidence to separate M4 strict-proof
work. `M3MBA-006` is blocked until the 0.7B control candidate has an official
GGUF, reproducible conversion path, or explicitly approved third-party artifact
path. `M3MBA-007` is blocked until the 3B candidate has an official TL1/TL2
GGUF, reproducible conversion path, or explicitly approved third-party
diagnostic artifact and enough local free space for safe large-candidate work.
The blocked secondary candidates are not handoff targets. Any new large
candidate remains serialized behind fresh disk preflight and cleanup/retention
evidence.

The next sequence should stay as small PRs, with each PR either merging evidence
or naming a blocker. The handoff stack is closed, so the next work is no longer
"run the old list"; it is a post-handoff execution plan:

| Stack position | Track | PR shape | Blocks |
|---:|---|---|---|
| 1 | `M3MBA-019` | Refresh the roadmap and tracker after the handoff stack closes. | Merged setup work from being treated as the next active task. |
| 2 | M3 device model | Tighten `apple-m3-air-*` capability surfaces, unsupported-backend rejection, and receipt wording without weakening M4 labels. | Hidden fallback, generic Metal/MPSGraph wording, or M4/M3 aliasing. |
| 3 | M3 accuracy comparison | Add a bounded comparison profile that reuses the accepted dense SLM receipt fields and records prompt IDs, generated IDs, decoded text, tokenizer authority, fallback status, and comparable/non-comparable gaps. | Accuracy claims without exact prompt/model/tokenizer/backend context. |
| 4 | M3 performance envelope | Record only completed-run actuals with power, thermal, storage, thread count, token budget, and cap sizing; use caps with cushion rather than near-completion termination. | Performance claims from cancelled, timed-out, or missing-context jobs. |
| 5 | Secondary artifact unblock | Re-check 0.7B and 3B candidates only through exact repo/file/revision/size/SHA/tokenizer/storage preflight before any download or run. | Official/approved artifact authority or storage-safe preflight is still missing. |
| 6 | M4 proof alignment | Keep accepted M3 artifact evidence linked to separate M4 strict-proof items and require fresh M4 receipts before any M4 claim. | A handoff would imply M3 reference-runner output is M4 proof. |

The follow-on implementation items should be created from these tracks as
single-purpose work items. M3 accuracy and performance work should prefer dense
SLM comparison receipts first because the lane already has model/tokenizer and
backend discipline there. BitNet artifact work should stay blocked until the
artifact authority surface changes or a separate approved third-party path is
recorded.

The just-closed stack is retained below as history:

| Stack position | Item | PR shape | Blocks |
|---:|---|---|---|
| 1 | `M3MBA-018` | Refresh the roadmap and campaign state after M3MBA-017. | Merged; prevents stale instructions from sending operators back through merged setup work. |
| 2 | `M3MBA-007` | Keep 3B work diagnostic-only on TL1/TL2 routes. | Blocked until an official/approved TL diagnostic artifact and safe storage state exist. |
| 3 | `M3MBA-008` | Hand accepted Microsoft 2B I2_S evidence to separate M4 strict-proof work. | Merged; fresh M4 receipts remain required before proof claims. |

## Post-Handoff Tracks

After `M3MBA-008`, the M3 Air lane has five active improvement tracks. These
tracks are intentionally narrower than the campaign objective so each PR can be
reviewed and merged independently.

| Track | Next evidence | Done when | Must not claim |
|---|---|---|---|
| Device model hardening | Tests, help text, and receipt wording proving `apple-m3-air-cpu-neon`, `apple-m3-air-metal`, and `apple-m3-air-mpsgraph` remain distinct. | Unsupported Metal/MPSGraph model requests fail closed and CPU fallback cannot masquerade as accelerator evidence. | Metal/MPSGraph model inference works. |
| Accuracy comparison | A bounded dense SLM profile with exact prompt/model/tokenizer/backend fields and comparable/non-comparable decisions against M4 and SLM CPU receipts. | Prompt IDs, generated IDs, decoded text, fallback status, and tokenizer authority are preserved for every comparison case. | Broad answer quality or BitNet behavior. |
| Performance envelope | Completed M3 Air run actuals with cap sizing, thermal/power context, storage state, thread count, token budget, and repeat policy. | The report separates healthy completed runtimes from timeouts/cancellations and sizes future caps from completed runs plus cushion. | Sustained broad Apple Silicon performance or M4 replacement timing. |
| Artifact unblock | Exact artifact preflights for blocked 0.7B/3B candidates or an explicit no-go refresh. | Repo, file, revision, size, SHA256, tokenizer authority, route, and free-space floor are known before any large download. | Candidate acceptance before local authority and cleanup evidence. |
| M4 proof alignment | Tracker linkage from accepted M3 artifact evidence to a separate M4 proof item. | The M4 item names the M3 evidence as input and requires fresh M4 backend receipts before proof. | M3 reference-runner output is M4 proof. |

For CI design, post-handoff M3 jobs follow the selected-long-job rule already
encoded by `M3MBA-013`: route irrelevant PRs away, preflight before expensive
work, upload phase evidence, and set caps from healthy completed runs with
cushion. Ending a selected M3 job just before receipt emission is treated as
wasted CI, not cost control.

## 2026-05-13 Tactical Plan

With H0 complete, the M3 Air should now move as dense control first, BitNet
artifact qualification second:

| Order | Item | Target | Merge decision |
|---:|---|---|---|
| 1 | `M3MBA-012` | Specify the dense SLM harness contract. | Merge when the docs name the CLI/receipt contract and synthetic CI receipt expectation. |
| 2 | `M3MBA-013` | Encode the selected-job CI completion rule. | Merge when the policy tells future M3 jobs how to route, preflight, cap, upload artifacts, and record actuals. |
| 3 | `M3MBA-014` | Implement the M3 Air Mac validate receipt bridge. | Merge when synthetic/CLI tests prove M3 backend and machine identity are preserved without live inference. |
| 4 | `M3MBA-004A` | Run the dense Qwen smoke mirror as the control path. | Merge pass receipts or a blocker report; do not skip to BitNet silently. |
| 5 | `M3MBA-004B` | Run the bounded dense Qwen operator profile. | Merge only after smoke passes, with allocation and thermal/power context. |
| 6 | `M3MBA-005A` | Identify and hash Microsoft 2B I2_S. | Record source, revision, size, SHA256, cache root, and storage context. |
| 7 | `M3MBA-005B` | Prove tokenizer/pre-tokenizer authority. | Record authority and bad/no-authority rejection evidence before output decisions. |
| 8 | `M3MBA-005C` | Screen Microsoft 2B I2_S reference outputs. | Accept, reject, or block with answer-gate result or failing prompt IDs. |
| 9 | `M3MBA-010` | Audit local cache retention after the first large download. | Merge before additional large candidates if free space falls below policy. |
| 10 | `M3MBA-006` | Try the smaller 0.7B control candidate. | Proceed only after Microsoft 2B has a decision. |
| 11 | `M3MBA-009` | Summarize dense SLM cross-lane behavior. | Merge after M3 dense receipts exist; feeds SLM CPU/M4 comparison. |
| 11 | `M3MBA-007` | Run 3B TL diagnostics only. | Keep diagnostic-only unless the compatibility matrix changes. |
| 12 | `M3MBA-008` | Write M4 strict-proof handoff. | Only for accepted artifacts, with fresh M4 proof still required. |

This order keeps the MacBook useful immediately while preserving the existing
proof hierarchy: first prove the host, then prove the dense control route, then
spend disk on BitNet candidates.

## Dense Harness Contract

`M3MBA-012` defines the contract that live M3 dense SLM smoke must satisfy
before any timing or answer receipt is treated as comparison-grade evidence.
It is a contract item only: it does not run a model, enable live M3 CI, or
publish performance results.

The live smoke command for `M3MBA-004A` must use the Mac CLI path and an
explicit MacBook backend label:

```bash
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  mac validate \
  --profile-set smoke \
  --device apple-m3-air-cpu-neon \
  --json-out ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json
```

The corresponding receipt check must run before the receipt is accepted:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  mac receipts-check \
  ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json \
  --json
```

The accepted smoke receipt must record these fields without aliasing the
MacBook to the M4 Mac mini lane:

| Field | Required value or rule |
|---|---|
| Requested backend | `apple-m3-air-cpu-neon` |
| Selected backend | `apple-m3-air-cpu-neon` |
| Fallback status | `fallback_used=false` for the accepted dense smoke path |
| Host identity | M3 Air machine profile path and machine identifier |
| Model identity | Exact dense Qwen model ID, local file name, size, and SHA256 |
| Tokenizer authority | Tokenizer source, revision, prompt template, and tokenizer metadata |
| Power and thermal context | Charger or battery state, thermal state when macOS exposes it, and unavailable fields marked unavailable |
| Storage context | Cache root plus free space before and after the run |
| Claim boundary | Dense SLM smoke only; no BitNet, M4, Metal, Neural Engine, MPSGraph, QK256, or broad Apple Silicon claim |

CI for the harness stays synthetic until a labeled, scheduled, manual, release,
or campaign hardware lane is explicitly selected. Required PR CI may validate:

```text
receipt schema shape
backend-label parsing for apple-m3-air-cpu-neon
mac receipts-check behavior on a no-model synthetic fixture
campaign tracker status and generated dashboards
documentation links and command spelling
```

Required PR CI must not download models, start live M3 timing, or require the
local MacBook to be online. A synthetic no-model receipt is acceptable only when
it says `inference_run=false` or otherwise marks itself as fixture evidence; it
must never be promoted into dense smoke, answer quality, or performance proof.

The first live dense receipt can be merged as either:

| Outcome | Required evidence |
|---|---|
| Pass | Receipt-check output, backend identity, fallback status, model/tokenizer identity, power/thermal/storage context, and dense-only claim boundary. |
| Blocked | A blocker report naming the missing prerequisite, command failure, storage or thermal issue, receipt validation failure, or model/tokenizer authority gap. |
| Reject | Receipt-check output and report explaining why the dense M3 smoke path is not acceptable for comparison. |

Do not proceed to large BitNet artifact downloads merely because the harness
contract exists. `M3MBA-004A` must either pass or leave an accepted blocker
before the lane spends disk on the Microsoft 2B I2_S path.

## Planning Horizons

The lane should be managed as three review horizons instead of one long
hardware push:

| Horizon | Work items | Purpose | Exit condition |
|---|---|---|---|
| H1: Make the host trustworthy | `M3MBA-002`, `M3MBA-003`, `M3MBA-012` | Turn this exact M3 Air into a receipt-backed Apple CPU/NEON evidence source with a named dense harness contract. | Machine profile, backend label, and harness contract are committed without inference claims. |
| H2: Prove the operator path | `M3MBA-004A`, `M3MBA-004B`, `M3MBA-009` | Show the MacBook can run the known dense SLM control path with comparable receipt fields. | Smoke/operator receipts are pass/fail reviewed and synthesis names comparable gaps. |
| H3: Spend disk on BitNet candidates | `M3MBA-005A`, `M3MBA-005B`, `M3MBA-005C`, `M3MBA-010`, `M3MBA-006`, `M3MBA-007`, `M3MBA-008` | Use the available storage for artifact identity, tokenizer authority, answer screening, cleanup, and strict-proof handoff. | Each candidate is accepted, rejected, blocked, or handed to a separate M4 proof item. |

H1 is complete through `M3MBA-012`, and `M3MBA-013` has added the selected
long-job CI posture for the next live dense runs. H2 should run live dense smoke
through `M3MBA-004A` only after `M3MBA-014` proves the Mac validate path
preserves M3 Air identity. H3 should not download the official 2B artifact until
the dense control path leaves either passing receipts or a committed
MacBook-specific blocker.

## CI Completion Policy

The M3 Air lane has two CI surfaces:

| Surface | Runs where | Timeout posture |
|---|---|---|
| Docs, tracker, schema, synthetic receipt validation | Required PR CI when relevant to the diff | Keep cheap and deterministic; fail fast before expensive work. |
| Live model downloads, dense SLM timing, large artifact sweeps | Local M3 Air evidence, campaign receipts, manual dispatch, or labeled/scheduled lanes | Size caps from healthy successful p95 plus cushion; do not use near-completion timeouts as cost control. |

Ending a selected long job just before it emits receipts is pure waste: the
runner minutes and cache churn are already spent, and the lane still has no
evidence. For M3 live lanes, control cost before the run starts by routing,
preflight checks, smaller profile selection, and one-active-large-artifact
serialization. Once selected, the job cap must be long enough for a healthy run
to finish with cushion.

Timeout and cancellation records remain evidence, but they are cap-failure
evidence. They should be retained in CI actuals and excluded from healthy
runtime percentiles so future caps are based on completed runs, not incomplete
ones.

`M3MBA-013` owns the concrete implementation pass for this rule. It adds the
manual-only `Apple M3 Air Dense SLM Evidence (staged)` workflow as the first
selected M3 hardware lane. A disabled dispatch writes only the staged status;
an enabled dispatch requires the provisioned self-hosted M3 Air runner, preserves
started runs with `cancel-in-progress: false`, performs disk preflight before
model work, writes phase artifacts under `target/apple-m3-air-dense-slm/`, and
uploads the artifact directory even if a late validation step fails.

Reviewers should use this selected-job checklist:

```text
1. classify the diff before selecting any live model, artifact, or timing job
2. reject missing disk, cache, power, thermal, model, tokenizer, or receipt
   prerequisites before downloads or model execution begin
3. serialize live M3 model jobs unless the storage audit explicitly allows
   multiple active artifacts
4. upload phase artifacts after profile, download, hash, validation, and receipt
   phases so a late failure still leaves evidence
5. size timeout-minutes from successful completed runs, not timed-out attempts
6. give the selected profile at least the larger configured percentage or minute
   cushion
7. keep aggregator and PR-gate polling deadlines longer than the upstream job
   they wait on
8. record timeout and cancellation as cap-failure actuals, excluded from healthy
   runtime percentiles
```

The staged workflow is intentionally not an ordinary PR trigger. It is a manual
or campaign lane for `M3MBA-004A` and `M3MBA-004B` evidence after `M3MBA-012`
lands. If a future labeled or scheduled variant is added, it must keep the same
completion posture: preflight before expensive work, selected runs allowed to
finish, phase evidence retained, and timeout caps based on successful completed
runs plus cushion.

`M3MBA-023` extends that staged workflow with an explicit `performance` profile
shape. The profile records cold-load sample count, warm prompt sample count,
token budget, storage state, power/thermal host context, and timeout-cap
provenance in the retained artifact bundle before any timing value can be used.
The cap provenance names completed healthy run evidence plus cushion; timed-out
or cancelled attempts remain cap-failure actuals and are excluded from healthy
runtime samples. A performance run may report timing only for the selected M3
MacBook Air receipt it produced, not for M4 Mac mini, broad Apple Silicon,
Metal, MPSGraph, Neural Engine, QK256, or BitNet answer-quality claims.

If these rules make a live M3 job too expensive for ordinary PR CI, the correct
design is to route it to a manual, labeled, scheduled, release, or campaign lane.
It is not correct to start the job and cap it just short of receipt emission.

## Dependency Map

The roadmap has one hard path and three side paths:

```text
M3MBA-001
  -> M3MBA-002
    -> M3MBA-003
      -> M3MBA-012
        -> M3MBA-013
          -> M3MBA-014
            -> M3MBA-004A
              -> M3MBA-004B
                -> M3MBA-005A
                  -> M3MBA-005B
                    -> M3MBA-005C
                      -> M3MBA-010
                      -> M3MBA-006
                      -> M3MBA-007
                        -> M3MBA-008

M3MBA-005A -> M3MBA-010
M3MBA-004B -> M3MBA-009
M3MBA-005C -> M3MBA-008 only after accepted-artifact or no-accepted-artifact state is explicit
```

`M3MBA-010` is allowed to interrupt the main artifact path after the first large
download and is a required gate before secondary large candidates. If free space
drops below the lane floor, storage audit and cleanup take priority over
additional candidate screening.

`M3MBA-009` is a synthesis side path, not a prerequisite for Microsoft 2B
identity work. It should wait for dense smoke/operator receipts, then compare
only fields that are present across M3, M4, and SLM CPU evidence.

`M3MBA-008` is the handoff side path. It should not invent M4 evidence; it
should create the next strict-proof item with the exact artifact and authority
requirements that passed on M3. If no artifact is accepted, close the handoff
with a no-accepted-artifact report instead of opening proof work.

## Parallelization Policy

The lane can use agents and parallel review, but the evidence sequence should
stay narrow:

| Work type | Can run in parallel? | Boundary |
|---|---|---|
| Review of machine-profile schema, report templates, and tracker docs | Yes | Must not change live receipt state without the owning work item. |
| Dense SLM command planning and artifact-cache inspection | Yes | Must not run model inference before `M3MBA-014` lands. |
| M3 CI completion-policy work | Yes | Must not enable live M3 timing in ordinary required CI; policy can land before live smoke. |
| Tokenizer authority research for Microsoft 2B | Yes | Must not accept or reject the artifact before `M3MBA-005A` records identity and hash. |
| Large artifact downloads | No, unless storage audit says there is headroom | One active large BitNet candidate at a time until `M3MBA-010` records retention policy. |
| M4 proof planning | Yes | Planning only; proof claims require fresh M4 receipts outside this lane. |

This lets the lane move quickly on docs, schemas, and review while keeping model
execution, artifact acceptance, and proof claims serialized behind committed
evidence.

## Success Metrics

The M3 Air lane is successful when it produces one of these reviewable outcomes:

| Area | Success metric | Failure still worth merging |
|---|---|---|
| Host readiness | Machine profile has exact model, chip, memory, macOS, power, thermal, cache, and free-space fields. | A blocker report names the missing host field and command that failed. |
| Dense SLM control | Qwen mirror receipts pass with deterministic settings and explicit M3 backend labels. | Receipt-check or quality failure is committed with model hash, tokenizer metadata, and fallback state. |
| BitNet Microsoft 2B | Official I2_S artifact has source revision, SHA256, tokenizer authority, and reference output decision. | Reference runner or tokenizer authority failure is committed as a rejection/blocker. |
| Storage discipline | Every large artifact has retention, cleanup status, and free-space before/after. | Additional model work pauses until cleanup is recorded. |
| Handoff quality | Accepted artifacts name the exact M4 proof item and receipt requirements. | No handoff if the artifact is only diagnostic or reference-bad. |

Timing numbers are secondary until the lane has comparison-grade receipts. A
fast diagnostic run without power, thermal, fallback, model hash, tokenizer, and
prompt context should not be used to steer product claims.

## SLM Lane Integration

The M3 Air is a useful dense SLM cross-check because it is a mobile Apple
Silicon host with enough storage to keep the known-good dense control model near
the BitNet candidate cache. It should feed the SLM lanes in three ways:

```text
M4 dense SLM lane:
  compare behavior and timing context, but keep M4 as the published Mac product
  and performance envelope

SLM CPU lane:
  compare dense Qwen prompt behavior and failure signatures against the i5-8250U
  CPU lane when both receipts name the same model, tokenizer, prompt, and greedy
  settings

Apple BitNet artifact sweep:
  use dense SLM receipts as a control that proves the MacBook runner, cache,
  tokenizer handling, receipt fields, and operator workflow before spending disk
  on larger BitNet artifacts
```

`M3MBA-009` owns the first cross-lane synthesis after `M3MBA-004A` smoke and
`M3MBA-004B` operator evidence exist. That report should not add a new model
claim; it should tell reviewers whether M3 behavior looks aligned enough to keep
using the MacBook as an SLM/BitNet screening host.

## Milestone Gates

The lane should advance only when the previous milestone leaves durable evidence
that a reviewer can inspect without access to the local model cache.

| Gate | Owner item | Required committed evidence | Local-only state | Exit decision |
|---|---|---|---|---|
| Machine readiness | `M3MBA-002` | Machine-profile receipt, schema-valid profile JSON, campaign event | None | Ready for receipt-label work |
| Receipt label readiness | `M3MBA-003` | Validator or documented label support for `apple-m3-air-cpu-neon` | None | Ready to record M3 timing without M4 wording |
| Device model completion | `M3MBA-015` | Shared backend/config labels for `apple-m3-air-metal`, `apple-m3-air-mpsgraph`, and `apple-m3-air-cpu-neon` | None | Ready to plan future M3 Air Metal or graph receipts without generic or M4 aliasing |
| Dense harness readiness | `M3MBA-012` | `mac validate`/`mac receipts-check` contract, synthetic no-model CI receipt expectation, and local-only timing boundary | None | Ready for live dense smoke or implementation PR |
| CI completion readiness | `M3MBA-013` | Routing, preflight, artifact upload, cap sizing, and CI actuals policy for selected long M3 jobs | None | Ready for scheduled/labeled live M3 evidence without near-completion caps |
| M3 validate bridge | `M3MBA-014` | Mac validate and receipts-check preserve `apple-m3-air-cpu-neon` backend, M3 machine ID, and M3-specific artifact kinds without live inference | None | Ready for live dense smoke |
| Dense control | `M3MBA-004A` | Smoke receipt, receipts-check output, model hash, tokenizer metadata, fallback status | Downloaded dense Qwen artifact | Ready for bounded operator run or blocker report |
| Dense operator | `M3MBA-004B` | Operator receipt, allocation-audit summary, thermal/power context | Warm model cache | Ready for BitNet artifact screening |
| Microsoft 2B identity | `M3MBA-005A` | Candidate report with source revision, filename, size, SHA256, cache root, free space before/after | Official 2B GGUF while active | Ready for tokenizer authority work |
| Microsoft 2B authority | `M3MBA-005B` | Tokenizer/pre-tokenizer authority and bad/no-authority rejection evidence | Official 2B GGUF while active | Ready for reference output decision |
| Microsoft 2B output | `M3MBA-005C` | Reference output, answer-gate result or failing prompt IDs, cleanup status | Official 2B GGUF while active | Accept, reject, or block before secondary candidates |
| Small candidate screening | `M3MBA-006` | Candidate report with route evidence and cleanup status | 0.7B GGUF while active | Keep for fast iteration or reject |
| Diagnostic candidate | `M3MBA-007` | TL1/TL2 diagnostic report and I2_S non-claim | 3B GGUF while active | Diagnostic only, no proof claim |
| Strict proof handoff | `M3MBA-008` | New M4 work item naming artifact, backend, and receipt requirements | No dependency on M3 cache | Ready for separate M4 proof |

If a gate cannot pass, the PR should still leave a blocker report instead of
silently skipping forward. A blocker report is acceptable evidence when it names
the failed command, host context, artifact identity when relevant, and the next
smallest fix.

Use numbered acceptance criteria in each PR:

```text
AC1 schema-valid evidence exists at the expected path
AC2 source, model, tokenizer, backend, and fallback fields are recorded when relevant
AC3 receipts-check or the applicable schema validator passes
AC4 no model binaries are committed
AC5 storage cleanup or retention status is recorded for every large artifact
AC6 claim boundaries remain unchanged or the PR explicitly explains the change
```

## Roadmap Shape

The M3 Air lane should move in four narrow lanes, each with a concrete stop
condition:

```text
foundation lane:
  prove the local machine facts, storage budget, cache root, and receipt label
  stop when M3MBA-002 records inference_run=false profile evidence

dense SLM lane:
  mirror the known-good Qwen route on the M3 Air with deterministic receipts
  stop when smoke/operator receipts either pass or name the MacBook blocker

BitNet artifact lane:
  qualify candidate artifacts with source, hash, tokenizer authority, reference
  output, rejection evidence, and cleanup state
  stop at candidate acceptance/rejection, not backend proof

handoff lane:
  create M4 strict-proof work only for accepted artifacts
  stop unless a fresh M4 receipt is produced on the target backend
```

This means the near-term roadmap is more than "run models on the MacBook". The
lane first proves the M3 Air as an evidence source, then proves the dense SLM
control path, then uses the available storage for BitNet artifact screening.

## Receipt Label

M3 Air receipts should use an explicit label instead of reusing M4 wording:

```text
requested_backend = apple-m3-air-cpu-neon
selected_backend = apple-m3-air-cpu-neon
machine_profile = mac15_13_m3_air_local
```

If the current validator cannot accept `apple-m3-air-cpu-neon`, the next PR
should add the smallest alias or receipt label needed for MacBook evidence. Do
not weaken the existing `apple-m4-cpu-neon` checks to make M3 receipts fit.

## First-Run Checklist

Run the first M3 Air session in this order:

```text
1. Record host profile:
   - model identifier
   - chip and core split
   - memory
   - macOS version
   - free disk
   - cache root
   - power source
   - thermal state when available
   - CPU/NEON, Metal, and MPSGraph visibility
   - inference_run=false

2. Confirm storage policy:
   - free disk before download
   - expected model sizes
   - minimum free-space floor
   - cleanup path for rejected artifacts

3. Run dense SLM smoke:
   - known model hash
   - tokenizer metadata
   - deterministic greedy settings
   - backend label and fallback status
   - receipts-check output

4. Run dense SLM operator profile only if smoke passes.

5. Download and qualify the official Microsoft 2B I2_S candidate only after the
   dense SLM control path is recorded.
```

## Thermal And Power Policy

The M3 Air is fanless, so receipts need mobile context. Every run that records
timing or throughput should include:

```text
power_source = ac | battery | unknown
low_power_mode = true | false | unknown
thermal_state_before = nominal | fair | serious | critical | unknown
thermal_state_after = nominal | fair | serious | critical | unknown
cooldown_seconds_before_run
repeat_count
```

Performance language is allowed only when AC/battery and thermal context are
recorded. If the run starts or ends in `serious` or `critical` thermal state,
record the receipt as diagnostic and do not compare it against M4 Mac mini
performance.

## Measurement Plan

Each model run should record enough context to separate behavior evidence from
mobile performance noise:

```text
run_mode = cold | warm | operator | diagnostic
power_source
low_power_mode
thermal_state_before
thermal_state_after
cooldown_seconds_before_run
repeat_count
prompt_count
ttft_ms when available
max_new_tokens
decode_tokens
wall_time_ms
tokens_per_second when supported by the receipt
peak_rss_bytes when available
swap_used_bytes when available
memory_pressure = normal | warning | critical | unknown
disk_free_before_bytes
disk_free_after_bytes
fallback_used
requested_backend
selected_backend
grade = comparison_grade | diagnostic_only
```

Use this ordering:

1. Cold smoke run after a clean process start.
2. Warm smoke rerun with the same artifact and prompt set.
3. Operator profile only after smoke passes.
4. Artifact diagnostic runs only after dense control receipts exist.

Do not compare cold and warm runs as regressions. Do not compare battery and AC
runs unless the report says the comparison is mobile-context-only. Do not compare
M3 Air timing to M4 Mac mini timing unless both receipts name the same model,
tokenizer, backend label, fallback status, prompt set, token budget, and thermal
context.

A run is comparison-grade only when power source, Low Power Mode, thermal
before/after, fallback status, model hash, tokenizer metadata, prompt set,
repeat count, and token budget are all recorded. Otherwise the run is
diagnostic-only.

## Accuracy Comparison Profile

`M3MBA-022` adds a distinct `mac validate --profile-set accuracy` route for the
M3 Air dense SLM lane. It reuses the established
`ci/quality/apple-m4-slm-quality-corpus.yaml` corpus and `mac receipts-check`
contract, but it records an explicit `accuracy_comparison_profile` block before
any cross-lane comparison claim is allowed.

The profile must record:

```text
work_item = M3MBA-022
device = apple-m3-air-cpu-neon
corpus path, name, sha256, case_count, repeat_runs, max_new_tokens
prompt_ids with case_id, prompt_index, repeat_index
generated_token_ids_recorded = true for every prompt
decoded_text_recorded = true for every prompt
scoring_policy.mechanical_scoring_only = true
comparison_grade_claim_made = false
M4 and SLM-CPU evidence marked non-comparable until a fresh matching receipt is selected
```

The staged hardware workflow exposes `profile_set=accuracy` so a selected M3
run can emit the same preflight/model-fetch phase evidence as smoke/operator
runs while preserving the comparison boundary. This is still dense SLM evidence:
it does not claim BitNet answer quality, M4 Mac mini performance, full Metal
inference, MPSGraph inference, Neural Engine execution, QK256 support, broad
quality, or broad Apple Silicon performance.

## Artifact Ledger

Large model downloads should have a small committed ledger entry in the relevant
report or receipt even when the binary remains local-only:

```text
artifact_id
source_url_or_repo
source_revision
filename
size_bytes
sha256
local_cache_root
download_started_at
download_completed_at
free_space_before_bytes
free_space_after_bytes
retention = keep | delete_after_report | delete_after_handoff
cleanup_status
```

Accepted candidates may stay in the local cache until M4 handoff is created.
Rejected candidates should be deleted unless their failure evidence cannot be
reproduced cheaply. The committed report should state what happened either way.

## Historical PR Stack

This table records the closed tactical order that brought the lane to the
Microsoft 2B handoff. It is retained for auditability; it is not the next active
execution queue.

| Order | Branch / item | Scope | Stop condition |
|---:|---|---|---|
| 1 | `M3MBA-012` | Dense SLM harness contract | CLI/receipt contract and synthetic CI receipt expectation recorded |
| 2 | `M3MBA-013` | CI completion policy | Selected long jobs have routing, preflight, artifact upload, cap, and actuals rules |
| 3 | `M3MBA-014` | M3 Air Mac validate receipt bridge | M3 backend and machine identity are preserved without live model inference |
| 4 | `M3MBA-004A` | Dense Qwen smoke mirror | Smoke receipt passes or blocker is recorded |
| 5 | `M3MBA-004B` | Dense Qwen operator mirror | Operator receipt passes or remains diagnostic-only |
| 6 | `M3MBA-005A` | Microsoft 2B I2_S artifact identity | Source, revision, size, hash, cache root, and storage state recorded |
| 7 | `M3MBA-005B` | Microsoft 2B I2_S tokenizer authority | Authority and bad/no-authority rejection evidence recorded |
| 8 | `M3MBA-005C` | Microsoft 2B I2_S reference output decision | Accept/reject/block report with answer-gate result |
| 9 | `M3MBA-010` | Storage and cache hygiene audit | Headroom decision before secondary large candidates |
| 10 | `M3MBA-006` | 0.7B 1bitLLM control candidate | Accept/reject report and cleanup state |
| 11 | `M3MBA-009` | Dense SLM cross-lane synthesis | Comparable receipt fields and gaps named |
| 12 | `M3MBA-007` | 3B TL1/TL2 diagnostic | Diagnostic report only |
| 13 | `M3MBA-008` | M4 strict-proof handoff | New M4 proof item, or no-accepted-artifact closure |

## Execution Roadmap

1. M3 Air lane bootstrap

   Record the real machine profile, cache root, free disk, power/thermal context
   when available, CPU/NEON visibility, Metal visibility, and MPSGraph visibility.
   This step should not run model inference.

   Planned receipt:

   ```text
   ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/machine-profile.json
   ```

   Minimum fields:

   ```text
   machine_id = mac15_13_m3_air_local
   model_identifier = Mac15,13
   chip = Apple M3
   memory_bytes
   macos_version
   available_disk_bytes
   model_cache_root
   power_source
   thermal_state when available
   cpu_neon_available
   metal_visible
   mpsgraph_visible when available
   inference_run = false
   requested_backend = none
   selected_backend = none
   ```

2. Dense SLM mirror

   Rerun the known-good dense Qwen2.5 Mac path on the M3 Air with the same model
   hash, tokenizer metadata, deterministic greedy settings, quality corpus, and
   receipt schema used by the established M4 SLM lane. The result is a mobile
   Apple Silicon cross-check, not a replacement for the M4 performance envelope.

   Pass/fail criteria:

   ```text
   corpus = ci/quality/apple-m4-slm-quality-corpus.yaml
   profile_set = smoke before operator
   corpus_repeat_runs >= 2
   max_new_tokens = 32 for smoke/operator parity unless the PR explains a change
   requested_backend = apple-m3-air-cpu-neon or documented successor
   selected_backend = apple-m3-air-cpu-neon or documented successor
   fallback_used = false for pass; true requires blocker or diagnostic-only grade
   receipts-check = pass
   generated output must satisfy the existing corpus checks
   thermal/power context must be present for comparison-grade timing
   ```

   Candidate command shape:

   ```text
   cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- model fetch \
     qwen2.5-0.5b-instruct-q8_0 \
     --json

   cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- model verify \
     qwen2.5-0.5b-instruct-q8_0 \
     --json

   cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac validate \
     --profile-set smoke \
     --corpus ci/quality/apple-m4-slm-quality-corpus.yaml \
     --corpus-repeat-runs 2 \
     --max-new-tokens 32 \
     --device apple-m3-air-cpu-neon \
     --json-out target/apple-silicon-macbook/m3-air/M3MBA-004A/qwen-mirror-smoke.json \
     --quiet

   cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac receipts-check \
     target/apple-silicon-macbook/m3-air/M3MBA-004A/qwen-mirror-smoke.json \
     --json
   ```

   If smoke passes, run the bounded operator profile:

   ```text
   cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac validate \
     --profile-set operator \
     --corpus ci/quality/apple-m4-slm-quality-corpus.yaml \
     --corpus-repeat-runs 2 \
     --max-new-tokens 32 \
     --allocation-audit \
     --device apple-m3-air-cpu-neon \
     --json-out target/apple-silicon-macbook/m3-air/M3MBA-004B/qwen-mirror-operator.json \
     --quiet
   ```

   The exact command may change if the CLI adds a MacBook-specific profile. If the existing M4
   profile label rejects the M3 host, add the smallest MacBook-specific receipt
   label instead of weakening M4 validation.

3. Official BitNet artifact qualification

   Start with `microsoft/bitnet-b1.58-2B-4T-gguf`
   `ggml-model-i2_s.gguf`. Record source revision, file size, SHA256, tokenizer
   authority, external Microsoft pre-tokenizer authority, reference-runner
   command, prompt outputs, bad/no-authority rejection evidence, and cleanup
   status.

   Cross-check candidate priority and route expectations against:

   ```text
   ci/hardware/apple-silicon-macbook/bitnet-candidate-matrix.toml
   docs/apple-silicon/bitnet-candidate-matrix.md
   docs/apple-silicon/apple-bitnet-artifact-sweep.md
   ```

   Required report:

   ```text
   docs/reports/apple-silicon-macbook-m3-air-microsoft-2b-i2s.md
   ```

   Required evidence:

   ```text
   source repository and revision
   exact GGUF filename
   size_bytes
   sha256
   tokenizer file source and revision
   tokenizer.ggml.pre authority
   reference runner and commit
   prompt suite and generated outputs
   no-authority rejection or bad-tokenizer rejection evidence
   cleanup status
   ```

   Reference-output rubric:

   ```text
   prompt suite = ci/quality/bitnet-answer-corpus.yaml unless the report names a narrower suite
   deterministic settings and prompt template are recorded
   every required prompt has non-empty generated text
   answers do not collapse into repeated special tokens or tokenizer garbage
   shared answer gate passes or the report lists failing prompt IDs
   no-authority tokenizer attempts are rejected or explicitly marked diagnostic
   ```

4. Smaller and diagnostic BitNet candidates

   Evaluate `1bitLLM/bitnet_b1_58-large` as the smaller control candidate, then
   use `1bitLLM/bitnet_b1_58-3B` only for supported TL1/TL2 diagnostic routes.
   Before either download, resolve the exact GGUF filename, revision, size
   estimate, tokenizer authority, and runner route in the report. Falcon-E
   candidates remain secondary in the shared matrix and should wait until
   Microsoft and 1bitLLM behavior is understood; the M3 Air campaign should add
   a future work item only if those earlier candidates leave a useful gap.

   Required reports:

   ```text
   docs/reports/apple-silicon-macbook-m3-air-1bitllm-07b.md
   docs/reports/apple-silicon-macbook-m3-air-3b-tl-diagnostic.md
   ```

   The 0.7B candidate should answer whether the smaller artifact is useful for
   fast local iteration on this M3 Air. The 3B diagnostic should answer only
   whether supported TL routes provide useful compatibility evidence; it must
   not be treated as I2_S support.

5. Strict proof handoff

   Promote only accepted artifacts to a separate M4 strict Apple CPU/NEON proof
   item. The M3 Air can qualify artifacts and compare Apple Silicon behavior; it
   must not be used to manufacture M4 receipts.

   Required handoff report:

   ```text
   docs/reports/apple-silicon-macbook-m3-air-m4-proof-handoff.md
   ```

## Post-Handoff Near-Term Order

1. `M3MBA-019` is merged; keep the roadmap, campaign prose, and generated
   tracker aligned with the post-handoff state.
2. `M3MBA-020` and `M3MBA-021` are merged, and `M3MBA-026` is the active
   device-model hardening follow-up. It is limited to server shared-engine
   receipt backend labeling so CPU/NEON, Metal, and MPSGraph M3 Air identities
   remain distinct from generic active-model devices and M4 proof labels even
   when the active model reports CUDA metadata.
3. Use `M3MBA-022` as the merged accuracy comparison baseline around bounded
   dense SLM receipts
   with exact prompt IDs, generated IDs, tokenizer authority, backend identity,
   fallback state, and comparable/non-comparable decisions.
4. Create the next M3 performance item only from completed-run actuals with
   thermal, power, storage, thread, token-budget, and cap-sizing evidence.
5. Keep `M3MBA-006` blocked unless a concrete 0.7B GGUF, reproducible conversion
   path, or explicitly approved third-party artifact path is named.
6. Keep `M3MBA-007` blocked until an official or explicitly approved 3B TL1/TL2
   diagnostic artifact and safe local storage state exist.
7. Keep M4 proof handoff separate from M3 evidence. `M3MBA-008` closed the first
   handoff report; any follow-on M4 proof item must run fresh M4 receipts before
   claiming proof.
8. Preserve the `M3MBA-013` selected-long-job rule for any future live M3 lane:
   route irrelevant work before it starts, preflight before expensive phases,
   upload partial phase artifacts, and size caps from completed runs plus
   cushion instead of ending selected jobs near completion.

## Review Checklist

Every PR in this lane should answer these questions in its description or
committed report:

```text
What evidence was produced?
Which receipts or reports were committed?
Which artifacts remain only in local cache?
What was deleted?
Which claim boundary is unchanged?
Which next work item is unblocked?
Which validation commands passed?
```

Do not merge a lane PR that only updates prose when a machine-readable campaign
state or generated tracker page also needs to change.

## Decision Gates

Proceed from machine profile to dense SLM mirror only when:

```text
available_disk_bytes is recorded
cache root is recorded
CPU/NEON visibility is recorded
the receipt explicitly says inference_run=false
```

Proceed from dense SLM mirror to BitNet artifact download only when:

```text
the known-good dense model hash and tokenizer metadata are recorded
requested_backend and selected_backend use apple-m3-air-cpu-neon or a documented successor label
power and thermal context are recorded for any timing comparison
receipts-check passes
fallback status is recorded
the report states dense SLM evidence is not BitNet evidence
```

Accept a BitNet artifact candidate only when:

```text
source, revision, file, size, and SHA256 are recorded
tokenizer authority and pre-tokenizer authority are recorded
reference output is coherent for the prompt suite
bad/no-authority tokenizer evidence is recorded when required
cleanup status is recorded
```

Promote to M4 strict proof only when:

```text
the artifact is accepted by reference output
the target backend route is named
the handoff does not claim M4 success
the next item requires a fresh M4 strict receipt
```

## Open Engineering Questions

1. `M3MBA-006`: which exact 0.7B GGUF filename, source revision, tokenizer
   authority, and runner route should be used for the smaller control candidate?
2. `M3MBA-007`: which TL1/TL2 3B artifact is small and authoritative enough to
   justify a diagnostic-only M3 run after the 0.7B decision?
3. Which dense SLM comparison profile is the smallest useful next M3 accuracy
   surface: the existing smoke corpus, a warm-session subset, or a new
   comparison-only prompt set?
4. Which completed M3 run actuals should seed future cap sizing, and which
   cancelled or timed-out runs must be excluded from healthy-runtime percentiles?
5. Which M4 strict-proof item should consume the Microsoft 2B I2_S handoff, and
   which fresh M4 receipt fields are required before it can claim local proof?

The default answer is conservative: keep one MacBook lane until actual receipts
show enough work to justify a separate M3 performance campaign; keep artifacts
under cache or `target/`; and do not add a new backend label unless the existing
receipt validator cannot represent M3 Air evidence without M4 wording.

## Storage Policy

Use cache or `target/` for all downloads and generated receipts. Never commit
model binaries.

With about 99 GiB free at lane bootstrap, the M3 Air can attempt the official 2B
I2_S candidate and one smaller control candidate without treating local storage
as unlimited. Use 8 GiB as the hard floor for avoiding an unsafe local checkout,
prefer at least 25 GiB free after active downloads, delete rejected candidates
unless a later work item explicitly retains them, and record cleanup status in
every artifact report.

## Claim Boundaries

The M3 Air lane may claim:

```text
M3 MacBook Air machine/profile facts are recorded.
Dense SLM behavior was cross-checked on this exact M3 Air when receipts exist.
BitNet artifacts are accepted or rejected as candidates when reference evidence exists.
```

The M3 Air lane must not claim:

```text
M4 Mac mini performance from M3 Air timing.
BitNet local-answer quality from dense Qwen evidence.
Rust Apple BitNet local answers before strict backend receipts.
full Apple Metal inference.
Neural Engine execution.
MPSGraph model inference.
QK256 support on Apple Silicon.
broad Apple Silicon performance from one mobile machine.
```
