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

The first productization task is to audit which Qwen receipts are real hardware
user-path receipts versus validators, fixtures, or contracts.

## Work item: CUDA-DENSE-050

Status: pr open
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

Status: planned
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
cargo run --locked --release -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-ask --device nvidia-rtx-5070-ti-cuda --model <qwen2.5-0.5b-q8_0.gguf> --question "What is 2+2? Answer with only the number." --max-new-tokens 1 --temperature 0 --strict-cuda --json-out ci\hardware\windows-9950x3d-rtx5070ti\<date>\dense-qwen25-q8-one-token-cuda.json
```

### Rollback

Remove the new receipt and demote any matrix/status row that depended on it.

## Work item: CUDA-DENSE-052

Status: planned
Campaign item: `CUDA-DENSE-052`
Blocked by: CUDA-DENSE-051

### Goal

Add deterministic 8 to 32 token strict CUDA short-decode proof.

### Acceptance

- no CPU fallback;
- stable greedy token sequence;
- valid UTF-8 answer;
- no raw special-token garbage;
- prefill, KV, logits, sampler, kernel, and transfer fields recorded;
- no broad chat or speed claim.

### Proof command

```bash
cargo run --locked --release -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-answer-corpus <args>
python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/<date>/dense-qwen25-q8-short-decode.json
git diff --check
```

### Rollback

Remove the short-decode receipt and keep one-token proof scoped.

## Work item: CUDA-DENSE-053

Status: planned
Campaign item: `CUDA-DENSE-053`
Blocked by: CUDA-DENSE-052

### Goal

Prove dense Qwen warm-session behavior.

### Acceptance

- model loaded once;
- tokenizer loaded once;
- CUDA context initialized once;
- intended persistent buffers reused;
- per-turn receipts and a session summary exist;
- `full_cuda_residency_claimed=false` unless all phases prove residency;
- `speedup_claim=false` unless benchmark-qualified.

### Receipt path

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/dense-qwen25-q8-warm-session.json
```

### Rollback

Remove the warm-session receipt and keep dense Qwen product status limited to
the last proven stage.

## Work item: CUDA-DENSE-054

Status: planned
Campaign item: `CUDA-DENSE-054`
Blocked by: CUDA-DENSE-053

### Goal

Add dense Qwen benchmark qualification.

### Acceptance

The review consumes one-token, short-decode, and warm-session receipts,
explicitly accepts or rejects speedup by profile, records blockers, and keeps
BitNet proof false.

### Rollback

Remove or demote only the benchmark qualification. Do not edit historical
execution receipts by hand.
