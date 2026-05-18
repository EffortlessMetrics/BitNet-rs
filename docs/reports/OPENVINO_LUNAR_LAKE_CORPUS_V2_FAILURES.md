# OpenVINO Lunar Lake Corpus V2 Failure Report

Status: diagnostic report
Created: 2026-05-18
Machine: intel-258v

## Scope

This report summarizes Lunar Lake OpenVINO dense SLM corpus-v2 candidate-route
failures after rerunning the corpus under the accepted one-token
`yes_no_clear_sky` fixture policy and the refreshed exact-lowercase
`stop_token_one_word_done` fixture. It does not promote any route, claim
speedup or power advantage, claim native Arc/NPU acceleration, or change BitNet
QK256/I2_S behavior.

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
| OpenVINO CPU | 9/12 pass, 3 fail | regression_tiny, prefill_heavy, decode_heavy | Candidate remains blocked |
| OpenVINO GPU.0 / Arc 140V | 9/12 pass, 3 fail | regression_tiny, prefill_heavy, decode_heavy | Candidate remains blocked |
| OpenVINO NPU | 10/12 pass, 2 fail | regression_tiny, prefill_heavy | Candidate remains blocked |

Both routes also remain blocked by missing benchmark-qualified speed or power
advantage, incomplete direct generated-token visibility, and profile-regression
evidence requirements in the route-profile comparison.

## Failure Classification

| Route | Case | Profile | Category | Classification | Evidence |
| --- | --- | --- | --- | --- | --- |
| CPU | stop_token_one_word_done | regression_tiny | stop_and_eos | exact_answer_instruction_not_followed | Expected `done`; observed `ai` |
| CPU | long_prompt_summary_route_policy | prefill_heavy | long_prompt_summarization | answer_content_missing_required_terms | Missing required term `Lunar` |
| CPU | decode_heavy_short_list | decode_heavy | decode_heavy | readable_output_missing_required_terms | Missing required term `model` |
| GPU.0 | stop_token_one_word_done | regression_tiny | stop_and_eos | exact_answer_instruction_not_followed | Expected `done`; observed `ai` |
| GPU.0 | long_prompt_summary_route_policy | prefill_heavy | long_prompt_summarization | answer_content_missing_required_terms | Missing required term `Lunar` |
| GPU.0 | decode_heavy_short_list | decode_heavy | decode_heavy | readable_output_missing_required_terms | Missing required term `model` |
| NPU | stop_token_one_word_done | regression_tiny | stop_and_eos | exact_answer_instruction_not_followed | Expected `done`; observed `ai` |
| NPU | long_prompt_summary_route_policy | prefill_heavy | long_prompt_summarization | answer_content_missing_required_terms | Missing required term `CPU` |

## Budget Sensitivity

The generation-budget sensitivity receipt isolates the normalized-match cases:

| Case | CPU | GPU.0 | NPU | Interpretation |
| --- | --- | --- | --- | --- |
| yes_no_clear_sky | passes at max_new_tokens=1 only | passes at max_new_tokens=1 only | passes at max_new_tokens=1 only | Accepted one-token fixture policy; rerun now passes |
| stop_token_one_word_done | no tested budget passes | no tested budget passes | no tested budget passes | True exact-answer instruction miss for tested budgets |

This means the yes/no failure was a stop/max-token fixture-budget issue rather
than a standing route-quality blocker after rerun. The one-word `done` case is
not explained by the tested smaller budgets.

## Profile Blockers

OpenVINO GPU.0 remains blocked for:

- `regression_tiny`: one-word stop/EOS instruction miss.
- `prefill_heavy`: required content term missing.
- `decode_heavy`: readable output missing required term.
- All profiles: generated token IDs are retokenized, benchmark-qualified
  advantage is missing, and candidate-route promotion evidence is incomplete.

OpenVINO NPU remains blocked for:

- `regression_tiny`: one-word stop/EOS instruction miss.
- `prefill_heavy`: required content term missing.
- All profiles: generated token IDs are retokenized, benchmark-qualified
  advantage is missing, and candidate-route promotion evidence is incomplete.
- NPU-specific: cache or resident warm-route proof is missing, and cold start is
  still classified as OpenVINO pipeline load or device compile dominated.

OpenVINO CPU remains blocked for:

- `regression_tiny`: one-word stop/EOS instruction miss.
- `prefill_heavy`: required content term missing.
- `decode_heavy`: readable output missing required term.
- All profiles: generated token IDs are retokenized, benchmark-qualified
  advantage is missing, and candidate-route promotion evidence is incomplete.

## Next Actions

1. Keep OpenVINO GPU/NPU routes unpromoted until profile failures pass or are
   intentionally re-gated by spec.
2. Fix or document exact-answer generation policy for
   `stop_token_one_word_done`.
3. Revisit prefill-heavy and decode-heavy expected terms only if the answer
   contracts are too narrow for the intended user profile.
4. Preserve direct versus retokenized generated-token visibility in every
   OpenVINO candidate receipt.
5. Run route promotion only after quality gates pass and exact-profile timing
   or power evidence proves an advantage over the current promoted CPU route.

## Claim Boundary

This report supports only the following claim:

```text
Existing Lunar Lake OpenVINO GPU/NPU corpus-v2 candidate failures are classified
by route, profile, case, and failure class.
```

It does not prove OpenVINO GPU/NPU route promotion, speedup, power advantage,
native OpenCL execution, native NPU execution, full BitNet accelerator
inference, packed QK256 accelerator decode, or BitNet QK256/I2_S behavior.
