# CUDA-MODEL-002 Qwen3 0.6B CPU Sanity

## Scope

`CUDA-MODEL-002` records a bounded CPU answer sanity receipt for the exact
Qwen3 0.6B Q8_0 artifact from `CUDA-MODEL-001` on the 9950X3D + RTX 5070 Ti
product bench. This is the CPU rung for model onboarding before any CUDA route
plan or strict CUDA proof.

## Receipt

| Field | Value |
|---|---|
| Receipt | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cpu-answer-corpus.json` |
| Per-case receipts | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cpu-answer-corpus-runs/` |
| Artifact kind | `slm_cpu_answer_corpus` |
| Backend lane | `dense_slm_cpu` |
| Requested backend | `cpu` |
| Selected backend | `cpu-rust` |
| Requested CPU kernel | `avx512` |
| Selected kernel | `dense-qwen-cpu-reference` |
| Fallback used | `false` |
| Speedup claim | `false` |
| Model SHA256 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |

## Corpus Result

The run used `ci/quality/slm-answer-corpus.yaml` with the Qwen no-thinking
prompt policy and strict GGUF tokenizer provenance. All five selected cases
passed:

| Case | Gate |
|---|---|
| `say_four` | exact trimmed |
| `capital_france` | contains `Paris` |
| `repeat_colors` | contains the requested color sequence |
| `say_ok` | exact trimmed |
| `yes_no_water` | starts with yes/no |

Quality summary:

```text
total: 5
passed: 5
failed: 0
fallback_used: false
speedup_claim: false
```

## Claim Boundary

This PR may claim that Qwen3 0.6B Q8_0 has a pinned artifact contract and a
bounded 9950X3D AVX-512 CPU answer sanity receipt.

It must not claim:

- Qwen3 CUDA execution;
- Qwen3 product CLI readiness;
- Qwen3 speedup;
- Qwen3 server readiness;
- full CUDA or CPU residency;
- dense Qwen2.5 proof inheritance;
- BitNet packed I2_S/QK256 proof.

The next proof is the Qwen3 CUDA all-layer plan and model-boundary fixture
classification. One-token CUDA proof remains blocked behind that plan.

## Validation

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- answer-corpus --corpus ci/quality/slm-answer-corpus.yaml --model models/slm/Qwen3-0.6B-Q8_0.gguf --device cpu --cpu-kernel avx512 --json-out ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cpu-answer-corpus.json --fail-on-quality
python -m json.tool ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-cpu-answer-corpus.json
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard_lists_qwen3_as_candidate_not_cuda_ready
```
