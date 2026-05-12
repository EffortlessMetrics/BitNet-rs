# M3 MacBook Air Lane Roadmap

The M3 MacBook Air lane is the live Apple Silicon MacBook lane for larger
artifact sweeps and dense SLM cross-checks. It is separate from the completed
M4 Mac mini product and performance campaigns.

Current host facts from the active runner:

```text
machine = MacBook Air
model_identifier = Mac15,13
chip = Apple M3
cpu_cores = 8
performance_cores = 4
efficiency_cores = 4
memory = 16 GB
available_repo_volume_space = about 99 GiB on 2026-05-12
```

These facts make the machine suitable for storage-conscious BitNet artifact
qualification and dense SLM Apple CPU/NEON cross-checks. They do not create M4
Mac mini performance evidence and do not prove BitNet local-answer quality.

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

## Roadmap Summary

| Phase | Work item | Outcome | Evidence |
|---:|---|---|---|
| 0 | `M3MBA-001` | Campaign tracker, roadmap linkage, and claim boundaries | `docs/tracking/campaigns/apple-m3-macbook-air/` |
| 1 | `M3MBA-002` | Real M3 Air machine profile, no inference | `ci/hardware/apple-silicon-macbook/.../machine-profile.json` |
| 2 | `M3MBA-003` | Explicit M3 Air receipt label | Validator or label evidence for `apple-m3-air-cpu-neon` |
| 3 | `M3MBA-004` | Dense Qwen SLM mirror on M3 Air | `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-smoke.json` plus report |
| 4 | `M3MBA-005` | Official Microsoft 2B I2_S artifact qualification | Reference-runner report, SHA256, tokenizer authority |
| 5 | `M3MBA-006` | Smaller 0.7B BitNet control candidate | `docs/reports/apple-silicon-macbook-m3-air-1bitllm-07b.md` |
| 6 | `M3MBA-007` | 3B TL1/TL2 diagnostic only | `docs/reports/apple-silicon-macbook-m3-air-3b-tl-diagnostic.md` |
| 7 | `M3MBA-008` | M4 strict-proof handoff for accepted artifacts | `docs/reports/apple-silicon-macbook-m3-air-m4-proof-handoff.md` |
| 8 | `M3MBA-009` | M3 SLM lane synthesis | Cross-lane report comparing M3 Air dense SLM receipts against M4 and 8250U SLM evidence |
| 9 | `M3MBA-010` | Storage and cache hygiene audit | Artifact ledger audit with retained/deleted model state and free-space floor |

## 2026-05-12 Tactical Plan

The M3 Air should now move as a short stack, not a single exploratory blob:

| Order | Item | Target | Merge decision |
|---:|---|---|---|
| 1 | `M3MBA-002` | Commit the live machine profile and storage budget. | Merge when schema-valid and `inference_run=false`. |
| 2 | `M3MBA-003` | Make `apple-m3-air-cpu-neon` a valid receipt label. | Merge only if M4 labels remain strict. |
| 3 | `M3MBA-004` | Run the dense Qwen mirror as the control path. | Merge pass receipts or a blocker report; do not skip to BitNet silently. |
| 4 | `M3MBA-005` | Screen Microsoft 2B I2_S with tokenizer authority. | Accept, reject, or block with hash and reference output evidence. |
| 5 | `M3MBA-010` | Audit local cache retention after the first large download. | Merge before additional large candidates if free space falls below policy. |
| 6 | `M3MBA-006` | Try the smaller 0.7B control candidate. | Proceed only after Microsoft 2B has a decision. |
| 7 | `M3MBA-009` | Summarize dense SLM cross-lane behavior. | Merge after M3 dense receipts exist; feeds SLM CPU/M4 comparison. |
| 8 | `M3MBA-007` | Run 3B TL diagnostics only. | Keep diagnostic-only unless the compatibility matrix changes. |
| 9 | `M3MBA-008` | Write M4 strict-proof handoff. | Only for accepted artifacts, with fresh M4 proof still required. |

This order keeps the MacBook useful immediately while preserving the existing
proof hierarchy: first prove the host, then prove the dense control route, then
spend disk on BitNet candidates.

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

`M3MBA-009` owns the first cross-lane synthesis after `M3MBA-004` exists. That
report should not add a new model claim; it should tell reviewers whether M3
behavior looks aligned enough to keep using the MacBook as an SLM/BitNet
screening host.

## Milestone Gates

The lane should advance only when the previous milestone leaves durable evidence
that a reviewer can inspect without access to the local model cache.

| Gate | Owner item | Required committed evidence | Local-only state | Exit decision |
|---|---|---|---|---|
| Machine readiness | `M3MBA-002` | Machine-profile receipt, schema-valid profile JSON, campaign event | None | Ready for receipt-label work |
| Receipt label readiness | `M3MBA-003` | Validator or documented label support for `apple-m3-air-cpu-neon` | None | Ready to record M3 timing without M4 wording |
| Dense control | `M3MBA-004` | Smoke receipt, receipts-check output, model hash, tokenizer metadata, fallback status | Downloaded dense Qwen artifact | Ready for bounded operator run or blocker report |
| Dense operator | `M3MBA-004` | Operator receipt, allocation-audit summary, thermal/power context | Warm model cache | Ready for BitNet artifact screening |
| Microsoft 2B screening | `M3MBA-005` | Candidate report with source revision, SHA256, tokenizer authority, reference output, cleanup status | Official 2B GGUF while active | Accept, reject, or block before secondary candidates |
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

## PR Stack

| Order | Branch / item | Scope | Stop condition |
|---:|---|---|---|
| 1 | `M3MBA-002` | Real M3 Air profile receipt | Machine facts and `inference_run=false` committed |
| 2 | `M3MBA-003` | Explicit M3 Air receipt label | `apple-m3-air-cpu-neon` validation does not weaken M4 checks |
| 3 | `M3MBA-004` | Dense Qwen smoke/operator mirror | Receipts pass or blocker is recorded |
| 4 | `M3MBA-005` | Microsoft 2B I2_S artifact qualification | Accept/reject report with tokenizer authority |
| 5 | `M3MBA-006` | 0.7B 1bitLLM control candidate | Accept/reject report and cleanup state |
| 6 | `M3MBA-007` | 3B TL1/TL2 diagnostic | Diagnostic report only |
| 7 | `M3MBA-008` | M4 strict-proof handoff | New M4 proof item, not an M3 claim |

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
     --backend-label apple-m3-air-cpu-neon \
     --json-out target/apple-silicon-macbook/m3-air/M3MBA-004/qwen-mirror-smoke.json \
     --quiet

   cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac receipts-check \
     target/apple-silicon-macbook/m3-air/M3MBA-004/qwen-mirror-smoke.json \
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
     --backend-label apple-m3-air-cpu-neon \
     --json-out target/apple-silicon-macbook/m3-air/M3MBA-004/qwen-mirror-operator.json \
     --quiet
   ```

   The exact command may change if the CLI already has a MacBook-specific profile
   or if `--backend-label` is not the final CLI spelling. If the existing M4
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
   Falcon-E candidates remain secondary and should wait until Microsoft and
   1bitLLM behavior is understood.

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

## Near-Term Order

1. Land `M3MBA-001` so the campaign tracker, generated status, and roadmap link
   become the lane control plane.
2. Run `M3MBA-002` to generate a real M3 Air machine-profile receipt under
   `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/`.
3. Run `M3MBA-003` to add or confirm the `apple-m3-air-cpu-neon` receipt label
   before model timing is recorded.
4. Run `M3MBA-004` as an M3 Air dense Qwen mirror now that real MacBook hardware
   is available.
5. Run `M3MBA-005` for the official Microsoft 2B I2_S reference qualification
   before any secondary BitNet candidate.
6. Use the 0.7B 1bitLLM candidate in `M3MBA-006` only after the Microsoft path
   records either acceptance or a clear blocker.
7. Keep M4 proof handoff separate in `M3MBA-008` until a candidate passes
   reference output with tokenizer authority.

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

1. Should the CLI keep using `mac validate` for all Apple Silicon hosts, or should
   the receipt schema add an explicit `apple-macbook-cpu-neon` backend label for
   M3 Air cross-reference receipts?
2. Should M3 Air dense SLM timing be tracked in the MacBook lane only, or should a
   separate `apple-m3-slm-performance` campaign exist after the first receipts?
3. Should accepted BitNet artifacts remain in the local cache after reference
   qualification, or should the default be delete-and-redownload for strict M4
   proof to avoid stale local state?
4. Which cache root should be canonical for large MacBook artifacts on this
   machine: the default user cache, `target/model-cache`, or an external volume?

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
