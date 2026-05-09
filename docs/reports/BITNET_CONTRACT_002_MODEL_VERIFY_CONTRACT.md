# BITNET-CONTRACT-002 Model Verify Contract Surface

## Summary

`bitnet model verify` now knows the official Microsoft
`BitNet-b1.58-2B-4T` GGUF `I2_S` artifact:

```powershell
bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s --json
```

The verifier checks the expected byte count and SHA256 for
`ggml-model-i2_s.gguf` and includes the
`microsoft_bitnet_b158_2b_4t_i2s` model-contract summary in JSON output and
cache metadata.

## Contract Fields

The emitted contract summary records:

- model family: `bitnet_b1_58`
- artifact format: `gguf`
- kernel family: `i2_s_qk256`
- contract status: `reference_ready`
- tokenizer authority: `external_llama_bpe`
- prompt authority: `bitnetcpp-answer`
- CPU oracle: `x86_cpu_scalar_then_avx512_parity`
- accelerator route: `bitnet_qk256_cuda`
- permitted claims and required receipt names
- claim boundary for speedup and full CUDA residency

## Claim Boundary

This PR does not change model loading, tokenizer behavior, prompt rendering,
transformer math, QK256 kernels, CUDA behavior, dense GGUF inference, server
runtime, or speed claims. Verifying the model cache proves artifact identity and
contract metadata only; backend answer, parity, benchmark, speedup, and
full-residency claims still require their strict receipts.
