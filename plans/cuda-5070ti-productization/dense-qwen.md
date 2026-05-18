# Dense Qwen2.5 0.5B Q8_0 CUDA Product Path

This file owns the first dense regular-LLM CUDA product lane. It is useful
local inference evidence, but it is not BitNet, I2_S, QK256, or 1-bit proof.

## Current Claim

The model coverage matrix records `dense_qwen25_05b_q8_cuda` as a dense SLM row
with:

```text
route = dense_regular_llm_cuda
dense_regular_llm_cuda_proof = true
bitnet_packed_i2s_qk256_proof = false
speedup_claim = false
server_ready = false
```

The productization audit, one-token proof, short-decode proof, warm-session
proof, benchmark qualification review, and bounded server-smoke receipt have
landed. The remaining boundary is still conservative: speedup and broad server
readiness stay false until later profile-specific readiness gates promote them.

## Work item: CUDA-DENSE-050

Status: merged (#4589)
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-DENSE-050`

### Goal

Produce `docs/reports/CUDA_DENSE_QWEN25_Q8_PRODUCT_AUDIT.md`.

### Acceptance

The audit answers:

- Is one-token strict CUDA real hardware execution or validator-only?
- Is short decode real?
- Is warm session real?
- Is chat path real?
- Are CPU and CUDA generated token IDs compared?
- Which receipts are committed?
- Which receipts are only synthetic validators?
- Which benchmark profiles exist?
- Which gaps block product CLI readiness?

### Proof commands

```bash
python -m json.tool <each committed dense qwen receipt>
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
git diff --check
```

### Receipt paths

```text
ci/hardware/windows-9950x3d-rtx5070ti/**/dense-*.json
```

### Claim boundary

Audit only. No new dense CUDA, answer, speed, server, or BitNet claim.

### Rollback

Revert the audit report and any docs-only references.

## Work item: CUDA-DENSE-051

Status: merged
Campaign item: `CUDA-DENSE-051`
Blocked by: CUDA-DENSE-050

### Goal

Implement or refresh the dense Qwen one-token strict CUDA receipt.

### Acceptance

The receipt records:

```text
artifact_kind = dense_gguf_qwen_one_token_strict_cuda
route = dense_regular_llm_cuda
selected_backend = nvidia-rtx-5070-ti-cuda
fallback_used = false
cpu selected token id
cuda selected token id
first divergence if any
kernel stats
transfer stats
quality gate result
speedup_claim = false
bitnet_packed_i2s_qk256_proof = false
```

### Proof command

```powershell
cargo run --locked --release -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-one-token-strict-cuda --model <qwen2.5-0.5b-q8_0.gguf> --prompt "What is 2+2? Answer with only the number." --json-out ci\hardware\windows-9950x3d-rtx5070ti\<date>\dense-qwen25-q8-one-token-cuda.json
```

On Windows, run this from a Visual Studio x64 developer shell with the CUDA
toolkit `bin` and `lib\x64` paths present so cudarc and Candle resolve the same
CUDA dynamic-linking mode. Non-Windows advisory lanes use fallback dynamic
loading so tracker and ripr checks do not require CUDA driver libraries.

### Receipt path

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/dense-qwen25-q8-one-token-cuda.json
```

### Rollback

Remove the new receipt and demote any matrix/status row that depended on it.

## Work item: CUDA-DENSE-052

Status: merged (#4695)
Campaign item: `CUDA-DENSE-052`
Blocked by: none

### Goal

Add deterministic 8 to 32 token strict CUDA short-decode proof.

### Current Proof

The 2026-05-14 current-source rerun records fallback-free RTX 5070 Ti CUDA
execution, CPU/CUDA generated-token equality, kernel stats, transfer stats,
`speedup_claim=false`, and bounded decoded text `The answer is 4. What is`.
It supersedes the earlier same-day stale-binary diagnostic blocker.

### Acceptance

- no CPU fallback;
- stable greedy token sequence;
- valid UTF-8 answer;
- no raw special-token garbage;
- prefill, KV, logits, sampler, kernel, and transfer fields recorded;
- no broad chat or speed claim.

### Proof command

```bash
cargo run --locked --release -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-short-decode-strict-cuda --model <qwen2.5-0.5b-q8_0.gguf> --one-token-proof ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/dense-qwen25-q8-one-token-cuda.json --json-out ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json
python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-diagnostic.json
python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json
git diff --check
```

### Rollback

Remove the short-decode receipt and keep one-token proof scoped.

## Work item: CUDA-DENSE-053

Status: merged (#4713)
Campaign item: `CUDA-DENSE-053`
Blocked by:

### Goal

Prove dense Qwen warm-session behavior.

Current-source receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json
```

### Acceptance

- model loaded once;
- tokenizer loaded once;
- CUDA context initialized once;
- intended persistent buffers reused;
- per-turn receipts and a session summary exist;
- `full_cuda_residency_claimed=false` unless all phases prove residency;
- `speedup_claim=false` unless benchmark-qualified.

Observed current-source proof:

```text
turns_count = 3
generated_tokens_total = 24
model_loaded_once = true
tokenizer_loaded_once = true
cuda_context_initialized_once = true
runtime_buffers_reused = true
weights_uploaded_once = true
per_turn_weight_upload = false
generated_token_ids_match = true
top_k_all_match = true
fallback_used = false
speedup_claim = false
full_cuda_residency_claimed = false
bitnet_packed_i2s_qk256_proof = false
```

### Receipt path

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/dense-qwen25-q8-warm-session.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json
```

### Rollback

Remove the warm-session receipt and keep dense Qwen product status limited to
the last proven stage.

## Work item: CUDA-DENSE-054

Status: merged (#4720)
Campaign item: `CUDA-DENSE-054`
Blocked by:

### Goal

Add dense Qwen benchmark qualification.

Current-source receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-benchmark-qualification-current-source.json
```

### Acceptance

The review consumes one-token, short-decode, and warm-session receipts,
explicitly accepts or rejects speedup by profile, records blockers, and keeps
BitNet proof false.

Observed current-source review:

```text
qualification_decision.status = not_accepted
accepted_profiles = []
blocked_profiles = [one_token, short_decode_8, warm_session_3_turns]
speedup_claim = false
benchmark_qualified_speedup = false
bitnet_packed_i2s_qk256_proof = false
```

### Rollback

Remove or demote only the benchmark qualification. Do not edit historical
execution receipts by hand.
