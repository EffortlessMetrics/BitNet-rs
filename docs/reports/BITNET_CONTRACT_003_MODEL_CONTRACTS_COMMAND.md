# BITNET-CONTRACT-003 Model Contracts Command

## Summary

`BITNET-CONTRACT-003` exposes the BitNet-family model contract matrix through
the user-facing model command surface without changing runtime inference. The
goal is to let operators inspect official I2_S, official TL1/TL2, unsupported
3B I2_S, listed-but-unverified 3B TL routes, and alternate-control contracts
before any artifact is used as proof authority.

## Behavior

- `bitnet model contracts` lists every BitNet-family contract.
- `bitnet model contracts <id-or-alias> --json` emits one contract summary.
- `bitnet model verify <known-contract-without-supported-artifact> --json`
  fails closed with `passed=false`, `supported_artifact=false`, and the
  contract summary.
- `bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s --json` continues to
  verify the registered official I2_S artifact bytes and emit the reference
  contract summary.

## Claim Boundary

This PR may claim that the CLI exposes contract metadata and unsupported-path
boundaries for every registered BitNet-family contract.

This PR must not claim:

- TL1 or TL2 answer readiness.
- 3B x86 I2_S support.
- new CPU, CUDA, dense GGUF, tokenizer, prompt-template, loader, transformer,
  QK256, server, speedup, or full-residency behavior.

