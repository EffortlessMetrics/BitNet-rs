# OpenVINO Lunar Lake Corpus V2 Failure Report

Status: diagnostic report
Created: 2026-05-18
Machine: intel-258v

## Scope

This report summarizes Lunar Lake OpenVINO dense SLM corpus-v2 candidate-route
failures after rerunning the corpus under the accepted one-token
`yes_no_clear_sky` fixture policy and the cross-runtime exact-text
`stop_token_one_word_done` fixture, plus the cross-runtime prefill-heavy and
decode-heavy fixture updates. It does not promote any route, claim speedup or
power advantage, claim native Arc/NPU acceleration, or change BitNet QK256/I2_S
behavior.

## Source Evidence

```text
ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-cpu-corpus-v2-diagnosis.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-gpu-corpus-v2-diagnosis.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-corpus-v2-diagnosis.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-generation-budget-sensitivity.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-route-profile-comparison.json
ci/quality/lunar-lake-answer-corpus-v2.yaml
```

All OpenVINO rows remain `promotion_status=candidate_only_not_promoted` with
`fallback_used=false`. Generated token IDs are marked as retokenized from
decoded text, not direct OpenVINO GenAI pipeline-internal token IDs.

## Route Summary

| Route | Corpus v2 result | Failed profiles | Promotion result |
| --- | ---: | --- | --- |
| OpenVINO CPU | 12/12 pass, 0 fail | none | Candidate remains blocked |
| OpenVINO GPU.0 / Arc 140V | 12/12 pass, 0 fail | none | Candidate remains blocked |
| OpenVINO NPU | 12/12 pass, 0 fail | none | Candidate remains blocked |

Candidate routes also remain blocked by missing benchmark-qualified speed or
power advantage, incomplete direct generated-token visibility, and
profile-regression evidence requirements in the route-profile comparison.

## Failure Classification

The current OpenVINO CPU/GPU/NPU corpus-v2 rerun has no failed cases. The prior
prefill-heavy and decode-heavy answer-content blockers were fixture/prompt
stability issues: the canonical corpus now asks the prefill-heavy summary to
include the exact words `Lunar`, `CPU`, and `route`, and asks the decode-heavy
case to use a stable route-check phrase bank containing `fallback` and `model`.

## Budget Sensitivity

The generation-budget sensitivity receipt isolates the normalized-match cases:

| Case | CPU | GPU.0 | NPU | Interpretation |
| --- | --- | --- | --- | --- |
| yes_no_clear_sky | passes at max_new_tokens=1 only | passes at max_new_tokens=1 only | passes at max_new_tokens=1 only | Accepted one-token fixture policy; rerun now passes |
| stop_token_one_word_done | passes at max_new_tokens=1/2/4 | passes at max_new_tokens=1/2/4 | passes at max_new_tokens=1/2/4 | Cross-runtime exact-text fixture now passes |

This means the yes/no failure was a stop/max-token fixture-budget issue rather
than a standing route-quality blocker after rerun. The one-word `done` case is
no longer an OpenVINO candidate blocker under the cross-runtime fixture wording.

## Profile Blockers

OpenVINO GPU.0 remains blocked for:

- All profiles: generated token IDs are retokenized, benchmark-qualified
  advantage is missing, and candidate-route promotion evidence is incomplete.
- `prefill_heavy` and `decode_heavy`: quality now passes, but profile-specific
  timing evidence is still insufficient for promotion.

OpenVINO NPU remains blocked for:

- All profiles: generated token IDs are retokenized, benchmark-qualified
  advantage is missing, and candidate-route promotion evidence is incomplete.
- NPU-specific: cache or resident warm-route proof is missing, and cold start is
  still classified as OpenVINO pipeline load or device compile dominated.
- `prefill_heavy` and `decode_heavy`: quality now passes, but profile-specific
  timing evidence is still insufficient for promotion.

OpenVINO CPU remains blocked for:

- All profiles: generated token IDs are retokenized, benchmark-qualified
  advantage is missing, and candidate-route promotion evidence is incomplete.
- `prefill_heavy` and `decode_heavy`: quality now passes, but profile-specific
  timing evidence is still insufficient for promotion.

## Next Actions

1. Keep OpenVINO GPU/NPU routes unpromoted until exact-profile timing, direct
   token visibility, and promotion evidence gaps are closed.
2. Preserve direct versus retokenized generated-token visibility in every
   OpenVINO candidate receipt.
3. Run route promotion only after quality gates pass and exact-profile timing
   or power evidence proves an advantage over the current promoted CPU route.

## Claim Boundary

This report supports only the following claim:

```text
Existing Lunar Lake OpenVINO CPU/GPU/NPU corpus-v2 candidate routes pass the
bounded quality fixture while remaining unpromoted because route-promotion
evidence is incomplete.
```

It does not prove OpenVINO GPU/NPU route promotion, speedup, power advantage,
native OpenCL execution, native NPU execution, full BitNet accelerator
inference, packed QK256 accelerator decode, or BitNet QK256/I2_S behavior.
