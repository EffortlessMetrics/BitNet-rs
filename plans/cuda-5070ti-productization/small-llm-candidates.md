# Dense SLM And Small Dense LLM Candidate Onboarding

Candidate rows do not inherit official BitNet or Qwen2.5 receipts. Each model
must pass its own artifact, tokenizer, prompt, CPU, CUDA, benchmark, status,
and user-surface gates.

## Priority Order

1. Qwen3 0.6B Q8/Q4.
2. SmolLM2 360M.
3. Llama 3.2 1B.
4. SmolLM2 1.7B.
5. Llama 3.2 3B.
6. Gemma or Phi small.

## Per-Model Ladder

Each model uses the same sequence:

| Step | Output | Claim boundary |
| --- | --- | --- |
| A. Artifact contract | `ci/model-artifacts/<model-id>.toml`, `docs/reports/<MODEL>_ARTIFACT_CONTRACT.md` | registered or structurally valid only |
| B. CPU answer sanity | CPU answer receipt | no CUDA claim |
| C. Dense all-layer plan | route counts, unsupported ops, gaps | plan only |
| D. Model-boundary fixtures | embedding, norm, LM head, KV, sampling fixtures | fixture proof only |
| E. One-token CUDA | strict fallback-free one-token receipt | exact model, one-token only |
| F. Short decode / warm session | decode and session receipts | scoped answer proof |
| G. Benchmark qualification | profile-specific decision | exact accepted profiles only |

## Work items: CUDA-MODEL-001 through CUDA-MODEL-005

Qwen3 0.6B was the first candidate because it is closest to the existing
Qwen2.5 dense infrastructure. It has now advanced through artifact contract,
CPU sanity, all-layer planning, one-token strict CUDA, short-decode strict CUDA,
warm-session strict CUDA, benchmark review, and earned status sync as an
accelerator-ready candidate. Product CLI readiness, speedup, server readiness,
full CUDA residency, broad dense GGUF readiness, and BitNet QK256 proof remain
false.

### CUDA-MODEL-001: Artifact Contract

Acceptance fields:

```text
repo/source
file
SHA256
bytes
GGUF type
architecture
quantization
tokenizer
chat template
context length
license
storage envelope
VRAM estimate
```

Proof:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

Rollback: remove the candidate artifact contract and report.

### CUDA-MODEL-002: CPU Answer Sanity

Acceptance prompts:

```text
math
capital
short summary
basic code
yes/no
```

Receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/qwen3-06b-cpu-answer-corpus.json
```

Rollback: remove the CPU receipt and leave the model registered only.

### CUDA-MODEL-003: CUDA All-Layer Plan

Acceptance:

- every transformer layer inspected;
- routed op counts recorded;
- unsupported ops recorded;
- model-boundary gaps explicit;
- strict CUDA readiness false unless all required routes are proven.

Rollback: revert the plan receipt and any matrix status change.

### CUDA-MODEL-004: One-Token CUDA Proof

Acceptance:

- strict selected backend `nvidia-rtx-5070-ti-cuda`;
- fallback false;
- CPU and CUDA selected tokens recorded;
- kernel and transfer stats present;
- BitNet proof false.

Rollback: remove the one-token receipt and keep CUDA answer readiness false.

### CUDA-MODEL-005: Short Decode And Warm Session

Acceptance:

- deterministic short-decode receipt;
- warm-session receipt when model/session reuse is claimed;
- quality gate result present;
- speedup false unless benchmark-qualified.

Rollback: remove the decode/session receipt and demote status rows.

## Next Candidate

The next candidate is SmolLM2 360M. Step A has landed as an exact artifact
contract. Step B has advanced through strict CPU preflight, governed
normalization policy, exact metadata-scoped normalization validation, and a
strict CPU retry that reaches tokenizer loading, prompt rendering, and
one-token generation with `fallback_used=false`.

Do not start SmolLM2 CUDA planning yet. The strict CPU retry selected `The`
for the math prompt, and SLM-CPU-021 records this as a wrong-first-token
blocker rather than CPU answer readiness. The next SmolLM2 proof must be a
reference-compatible first-token/top-k or checkpoint comparator. The current
SmolLM2 state can only claim structurally valid artifact metadata plus the
committed strict CPU blocker/diagnosis chain; it cannot claim CPU answer
readiness, CUDA route, product CLI, benchmark, speed, server, full-residency,
broad dense GGUF, or BitNet QK256 proof.

After SmolLM2 360M clears step B, repeat the same A through G sequence for
Llama 3.2 1B, SmolLM2 1.7B, Llama 3.2 3B, Gemma, and Phi. Do not combine
candidate families in one PR.
