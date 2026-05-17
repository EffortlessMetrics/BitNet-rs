# Kaby Lake SLM Q4 Expansion Plan

`SLM-CPU-028` defines the gate for Q4_K_M and Q4_K_S work on the
i5-8250U dense SLM lane. It is a planning artifact only: it does not add Q4
runtime support and it must not be used as a Q4 support claim.

The Qwen3 Q8_0 appliance profile remains the behavior oracle. Any Q4 work must
prove the same receipt discipline before claiming support:

```text
real dense GGUF
pinned model SHA256
verified GGUF metadata
strict GGUF tokenizer authority
selected_backend = cpu-rust
fallback_used = false
prompt IDs preserved where a Q8 oracle exists
generated IDs and decoded text preserved for regression prompts where an oracle exists
constrained corpus evidence
multi-token determinism
warm-session receipt
operator profile with timing, memory, storage, thermal, and power fields
```

## Candidate Status

The current candidate manifest lives at
`ci/slm-cpu/q4-expansion-candidates.toml`.

| Candidate | Status | Role |
| --- | --- | --- |
| `qwen2_5_0_5b_instruct_q4_k_m` | candidate pinned | First Q4_K_M sanity target because the official Qwen GGUF repo publishes a Q4_K_M file with a pinned SHA256. |
| `qwen3_0_6b_q4_k_m_lmstudio` | candidate pinned, community artifact | Optional Qwen3 Q4_K_M comparison candidate. It is not the Qwen3 proof anchor unless artifact provenance, metadata, tokenizer authority, and generated-ID behavior pass the same gates. |
| Q4_K_S | no accepted candidate | No Q4_K_S artifact is accepted for this lane until an exact repo, file, SHA256, architecture metadata, tokenizer source, and tensor policy are pinned. |

The official `Qwen/Qwen3-0.6B-GGUF` repo remains the Qwen3 Q8_0 proof anchor.
At the time this gate was authored, that official repo exposes the verified
Q8_0 artifact used by the Kaby appliance profile, not an accepted Q4_K_M or
Q4_K_S artifact for this lane.

## Required Gate Sequence

Q4 work must proceed in this order. A later gate cannot compensate for a weak
earlier one.

1. Artifact identity
   - repo, file, and source URL recorded
   - SHA256 recorded from an artifact-linked digest
   - local `Get-FileHash` or equivalent matches before runtime use
   - file size recorded for storage planning

2. Metadata preflight
   - `general.architecture` recorded
   - quant format recorded as Q4_K_M or Q4_K_S
   - context length capped for the 8250U
   - tokenizer source recorded
   - tokenizer strictness true
   - dense adapter selected by metadata, not model name
   - unsupported metadata fails closed

3. Strict CPU load
   - real GGUF loader path
   - selected backend `cpu-rust`
   - fallback false
   - no compatibility dequant fallback hidden as proof
   - no BitNet QK256/I2_S path selected

4. Correctness evidence
   - constrained corpus receipt
   - prompt IDs recorded
   - generated IDs recorded
   - decoded text recorded
   - quality status and failed rules recorded
   - generated-ID comparison against the Q8 oracle where the prompt and model
     family make that comparison meaningful

5. Determinism evidence
   - multi-token run at 4, 8, and 16 generated-token bounds where feasible
   - repeated greedy runs produce the same generated IDs
   - stop reason and EOS behavior recorded
   - KV/cache position metadata recorded where available

6. Warm-session evidence
   - model loaded once
   - tokenizer loaded once
   - prompt/session buffers reused where available
   - per-case receipts emitted
   - aggregate receipt emitted
   - generated IDs and decoded text checked against the cold/corpus oracle

7. Operator profile
   - timing fields: model load, tokenizer load, prefill, first token, steady decode
   - memory fields: resident and virtual memory where available
   - storage/free-space fields populated on Windows
   - thread count recorded
   - thermal and power fields recorded as measured or explicitly unavailable
   - speedup claim false unless a later item adds a dedicated before/after
     performance receipt with unchanged generated IDs

## Failure Policy

Q4 is fail-closed. If any gate cannot be satisfied, the artifact remains a
candidate and the receipt must say why. In particular:

```text
missing SHA256 -> no support claim
metadata mismatch -> no support claim
tokenizer source guessed -> no support claim
fallback_used = true -> no support claim
generated IDs drift without explanation -> no support claim
thermal/power unavailable but not recorded -> no operator-profile claim
```

SmolLM2 remains governed by its exact metadata-scoped normalization policy.
Q4 work must not use SmolLM2 as a shortcut around that fail-closed rule.

## Claim Boundary

This plan may be used to claim:

```text
The Kaby SLM lane has a Q4_K_M/Q4_K_S expansion gate.
Qwen2.5-0.5B-Instruct Q4_K_M is a pinned candidate, not supported runtime.
Qwen3 community Q4_K_M is a pinned comparison candidate, not the proof anchor.
Q4_K_S has no accepted candidate until exact metadata and SHA are pinned.
```

This plan must not be used to claim:

```text
Q4_K_M or Q4_K_S runtime support exists
sustained 8250U throughput
broad answer quality
server inference
GPU, NPU, OpenVINO, or UHD 620 execution
Qwen3.5 or hybrid architecture support
BitNet QK256 changes
```
