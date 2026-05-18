# BITNET-SPEC-INTEL-GPU-QUALITY

## Purpose

Ensure Intel GPU "works" means intelligible, route-specific inference rather
than only device visibility or isolated speed.

## BitNet A770/OpenCL gates

- Official answer corpus.
- Prompt conditioning.
- Paired context changes answer.
- Copy/repeat.
- Yes/no.
- Format following.
- Stop-token behavior.
- Long decode.
- CPU/A770 generated-token parity or first-divergence classification.

## Dense SLM OpenVINO GPU gates

- Lunar Lake answer corpus v2.
- Profile summaries.
- Category summaries.
- Failure taxonomy.
- Generation-budget sensitivity.
- Stop/EOS diagnosis.
- Retokenized-vs-direct-token-ID boundary.

## Failure taxonomy

Failures must use one or more of: `exact_answer_instruction_not_followed`,
`exact_answer_overgenerated`, `missing_required_keyword`,
`forbidden_token_observed`, `raw_special_token_seen`, `empty_answer`,
`repetition`, `stop_policy_failed`, `context_sensitivity_failed`,
`structured_output_failed`, `timeout`, or `runtime_error`.
