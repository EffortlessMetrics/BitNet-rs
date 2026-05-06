# BitNet Parity Tolerances

## Purpose

Implementation PRs must not invent tolerances ad hoc. Update this file deliberately when a tolerance is proven or intentionally changed.

## Initial Policy

```yaml
scalar_vs_scalar:
  token_ids_exact: true
  logits_exact: true

scalar_vs_avx2:
  logits_max_abs_error: TBD
  logits_mean_abs_error: TBD
  greedy_next_token_must_match: true

scalar_vs_avx512:
  logits_max_abs_error: TBD
  logits_mean_abs_error: TBD
  greedy_next_token_must_match: true
  note: do not invent before 9950X3D or another AVX-512 lane proves it

scalar_vs_neon:
  logits_max_abs_error: TBD
  logits_mean_abs_error: TBD
  note: do not invent before ARM/M4 CPU proof

cpu_vs_gpu_native:
  logits_max_abs_error: TBD
  logits_mean_abs_error: TBD
  greedy_next_token_must_match: true

cpu_vs_openvino_graph:
  logits_max_abs_error: TBD
  logits_mean_abs_error: TBD
  note: graph conversion may change math path

sampling:
  deterministic_tests_use_temperature: 0.0
  require_seed_when_sampling: true
```

## Rules

- Exact scalar comparisons should remain exact unless the scalar path changes semantics.
- Deterministic greedy tests use `temperature=0.0`.
- Sampling tests require a seed and sampling policy.
- GPU and OpenVINO tolerances remain TBD until proven.
- Token agreement for greedy output is required when comparing user-visible output.
- Unknown tolerances must stay `TBD`.

## Receipt Fields

Parity receipts should include:

```json
{
  "parity": {
    "class": "scalar_vs_avx2",
    "logits_max_abs_error": 0.0,
    "logits_mean_abs_error": 0.0,
    "token_agreement_for_greedy": true,
    "tolerance_source": "docs/bitnet/BITNET_PARITY_TOLERANCES.md"
  }
}
```
