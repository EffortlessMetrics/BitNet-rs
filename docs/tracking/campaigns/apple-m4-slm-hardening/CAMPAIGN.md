# Apple M4 SLM Hardening Campaign

Campaign ID: `apple-m4-slm-hardening`

Status: active

## Objective

Make the completed Apple M4 SLM path boring for local users: simple Mac commands, default verified model-cache behavior, clear first-run guidance, stable receipts, and conservative hardware claim boundaries.

## Why This Exists

The `apple-m4-slm-answer`, `apple-m4-productization`, and `apple-m4-slm-performance` campaigns proved the practical dense-SLM path: the M4 can run Qwen2.5 0.5B Instruct, a regular dense instruct GGUF, through Rust-native `apple-m4-cpu-neon`, keep the model resident for warm prompts, produce measured receipts, and exercise a named Metal phase without claiming full Metal inference.

Qwen2.5 dense-SLM success validates the Mac user experience, model-cache flow, receipts, warm sessions, and Apple CPU/NEON routing. It does not validate BitNet, 1-bit / 1.58-bit kernels, I2_S/TL1/TL2 layouts, QK256, or Apple BitNet local-answer quality.

This campaign owns the next user-facing polish layer. It should remove avoidable command friction and make failure modes clearer without reopening proof, performance, BitNet, QK256, Metal-kernel, or server-inference work.

## End State

- `bitnet mac ask "question"` works as the shortest supported Mac local-answer command.
- Existing `bitnet mac ask --question "question"` scripts continue to work.
- First-run and missing-cache failures point directly to `bitnet model fetch qwen2.5-0.5b-instruct-q8_0`.
- Device-boundary failures still reject full `apple-m4-metal`, MPSGraph inference, and hidden fallback before cache or model work.
- Receipts and docs preserve model, tokenizer, backend, fallback, generated text, token IDs, and timing expectations.

## Hard Constraints

- Do not reopen completed Apple M4 proof, operational, SLM answer, productization, or performance campaigns.
- Do not weaken blocked BitNet local-answer gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full `apple-m4-metal` inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not touch QK256, `bitnet-qk256-dispatch`, server inference, or Metal kernels.
- Never commit model binaries.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SLM-HARDEN-001 | merged | Add positional `bitnet mac ask "question"` UX while preserving `--question`, default cache, and backend boundaries. |
| M4-SLM-HARDEN-002 | merged | Improve first-run cache repair and low-disk guidance for Mac SLM operators. |
| M4-SLM-HARDEN-003 | ready | Expand the small operator quality corpus without turning it into a broad eval. |
| M4-SLM-HARDEN-004 | proposed | Seed regression guardrails from the measured performance envelope. |

## Review Policy

Each PR owns one hardening item. User-facing polish must include CLI regression coverage and must not broaden backend claims. Runtime or performance changes should move to a follow-up campaign item instead of being folded into command UX.
