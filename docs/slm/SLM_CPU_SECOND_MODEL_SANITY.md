# Kaby Lake Second Dense Model Sanity

`SLM-CPU-017` checks that the i5-8250U SLM proof-appliance path is not only
Qwen3-shaped. The second candidate is intentionally still small and Qwen-class:
`qwen2.5-0.5b-instruct-q8_0`, selected from the repo's supported model cache by
exact pinned artifact metadata rather than by model-name recognition.

## Candidate

- Repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- Revision: `9217f5db79a29953eb74d5343926648285ec7e67`
- File: `qwen2.5-0.5b-instruct-q8_0.gguf`
- SHA256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`
- GGUF architecture: `qwen2`
- Quantization: `Q8_0`
- Tokenizer authority: GGUF metadata / `qwen2`
- Prompt template: `qwen2.5`

## Evidence

The 2026-05-16 Kaby Lake sanity bundle is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-16/qwen25-model-verify.json
ci/slm-cpu/intel-i5-8250u/2026-05-16/qwen25-second-model-sanity.json
ci/slm-cpu/intel-i5-8250u/2026-05-16/qwen25-second-model-sanity-runs/math_2_plus_2_brief.json
```

The run uses the strict CPU answer-corpus surface with `selected_backend =
cpu-rust`, `runtime_api = cpu`, `fallback_used = false`, `loader.mode =
real_gguf`, and `tokenizer.source = gguf_metadata`. It records prompt IDs,
generated IDs, decoded text, dense Qwen kernel identity, and first-step top-k
logits for the bounded `math_2_plus_2_brief` case.

This is a second-model sanity receipt, not a broad model-quality or performance
claim.

## Reproduction

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  model fetch qwen2.5-0.5b-instruct-q8_0 `
  --cache-dir target\slm-cpu-017\cache `
  --json

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  model verify qwen2.5-0.5b-instruct-q8_0 `
  --cache-dir target\slm-cpu-017\cache `
  --json

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  answer-corpus `
  --corpus ci\quality\slm-second-model-sanity-corpus.yaml `
  --model target\slm-cpu-017\cache\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf `
  --model-id qwen2.5-0.5b-instruct-q8_0 `
  --device cpu `
  --threads 4 `
  --case-id math_2_plus_2_brief `
  --dump-logit-steps 1 `
  --logits-topk 5 `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-16\qwen25-second-model-sanity.json
```

## Claim Boundary

This slice does not claim broad answer quality, sustained 8250U throughput,
Q4/Q5 quant expansion, server inference, GPU/NPU/OpenVINO/UHD 620 execution,
Qwen3.5 support, or BitNet QK256 behavior. The established Kaby Lake baseline
remains the Qwen3-0.6B Q8_0 operator appliance profile.
