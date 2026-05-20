# BITNET-SPEC-NPU-DENSE-SLM

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Define the dense SLM NPU route. This route is allowed to become useful earlier
than full BitNet NPU inference because OpenVINO GenAI can target small dense
models on NPU with explicit export, generation, quality, and cache contracts.

## Initial target

```text
Qwen2.5 0.5B Instruct OpenVINO INT4/NF4 symmetric export on Lunar Lake NPU
```

## Model/export contract

```json
{
  "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "export_tool": "optimum-cli export openvino",
  "format": "openvino_ir",
  "weight_format": "int4|nf4",
  "symmetric": true,
  "group_size": 128,
  "ratio": 1.0,
  "tokenizer_source": "hf_tokenizer_export",
  "prompt_template": "qwen2.5",
  "model_binary_committed": false
}
```

## Required proof ladder

1. Export manifest.
2. OpenVINO CPU control.
3. OpenVINO GPU comparison.
4. OpenVINO NPU bounded ask.
5. Corpus v2.
6. Generation-budget sensitivity.
7. Cold/cache/warm benchmark.
8. Resident-session benchmark.
9. Route promotion review.
10. Model status surface.
11. Optional exact-profile server smoke.

## Receipt fields

Dense SLM NPU receipts must record source model, exported artifact manifest,
tokenizer source, prompt template, generation config, OpenVINO/GenAI version,
`MAX_PROMPT_LEN`, `MIN_RESPONSE_LEN`, `PREFILL_HINT`, `GENERATE_HINT`, cache/blob
settings, selected device, fallback status, and quality result.

## Must not claim

- BitNet QK256.
- Native NPU kernels.
- Full residency.
- Broad SLM quality.
- Cold one-off usability.
- Generic NPU support across non-Intel families.
