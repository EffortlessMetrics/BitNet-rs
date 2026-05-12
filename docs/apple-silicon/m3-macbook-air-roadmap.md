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

## Roadmap Summary

| Phase | Work item | Outcome | Evidence |
|---:|---|---|---|
| 0 | `MB-AS-007` | M3 Air roadmap, storage policy, sequencing, and claim boundaries | This document and campaign tracking |
| 1 | `MB-AS-008` | Real M3 Air machine profile, no inference | `ci/hardware/apple-silicon-macbook/.../machine-profile.json` |
| 2 | `MB-AS-002` | Dense Qwen SLM mirror on M3 Air | Mac validate receipt plus receipts-check output |
| 3 | `ABAS-001` / `MB-AS-004` | Official Microsoft 2B I2_S artifact qualification | Reference-runner report, SHA256, tokenizer authority |
| 4 | `ABAS-002` / `MB-AS-005` | Smaller 0.7B BitNet control candidate | Reference-runner report and cleanup record |
| 5 | `ABAS-003` / `MB-AS-006` | 3B TL1/TL2 diagnostic only | Diagnostic report, explicit I2_S non-support note |
| 6 | `ABAS-005` | M4 strict-proof handoff for accepted artifacts | Handoff plan only; no manufactured M4 receipt |

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
   ```

2. Dense SLM mirror

   Rerun the known-good dense Qwen2.5 Mac path on the M3 Air with the same model
   hash, tokenizer metadata, deterministic greedy settings, quality corpus, and
   receipt schema used by the established M4 SLM lane. The result is a mobile
   Apple Silicon cross-check, not a replacement for the M4 performance envelope.

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
     --json-out target/apple-silicon-macbook/m3-air/MB-AS-002/qwen-mirror-smoke.json \
     --quiet

   cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac receipts-check \
     target/apple-silicon-macbook/m3-air/MB-AS-002/qwen-mirror-smoke.json \
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
     --json-out target/apple-silicon-macbook/m3-air/MB-AS-002/qwen-mirror-operator.json \
     --quiet
   ```

   The exact command may change if the CLI already has a MacBook-specific profile
   or if the existing M4 profile label rejects the M3 host. If that happens, add
   the smallest MacBook-specific receipt label instead of weakening M4 validation.

3. Official BitNet artifact qualification

   Start with `microsoft/bitnet-b1.58-2B-4T-gguf`
   `ggml-model-i2_s.gguf`. Record source revision, file size, SHA256, tokenizer
   authority, external Microsoft pre-tokenizer authority, reference-runner
   command, prompt outputs, bad/no-authority rejection evidence, and cleanup
   status.

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

4. Smaller and diagnostic BitNet candidates

   Evaluate `1bitLLM/bitnet_b1_58-large` as the smaller control candidate, then
   use `1bitLLM/bitnet_b1_58-3B` only for supported TL1/TL2 diagnostic routes.
   Falcon-E candidates remain secondary and should wait until Microsoft and
   1bitLLM behavior is understood.

   The 0.7B candidate should answer whether the smaller artifact is useful for
   fast local iteration on this M3 Air. The 3B diagnostic should answer only
   whether supported TL routes provide useful compatibility evidence; it must
   not be treated as I2_S support.

5. Strict proof handoff

   Promote only accepted artifacts to a separate M4 strict Apple CPU/NEON proof
   item. The M3 Air can qualify artifacts and compare Apple Silicon behavior; it
   must not be used to manufacture M4 receipts.

## Near-Term Order

1. Finish `MB-AS-007` and merge the M3 Air roadmap.
2. Run `MB-AS-008` to generate a real M3 Air machine-profile receipt under `ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/`.
3. Run `MB-AS-002` as an M3 Air dense Qwen mirror now that real MacBook hardware is available.
4. Run the official Microsoft 2B I2_S reference qualification before any secondary BitNet candidate.
5. Use the 0.7B 1bitLLM candidate only after the Microsoft path records either acceptance or a clear blocker.
6. Keep M4 proof handoff separate until a candidate passes reference output with tokenizer authority.

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
as unlimited. Keep at least 25 GiB free after active downloads when practical,
delete rejected candidates unless a later work item explicitly retains them, and
record cleanup status in every artifact report.

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
