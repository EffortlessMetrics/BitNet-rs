# BITNET-CONTRACT-004 Architecture Support Summary

## Summary

`BITNET-CONTRACT-004` adds architecture support rows to the contract summaries
emitted by `bitnet model contracts` and by contract-aware `bitnet model verify`
output. The registry already stored architecture support; this work makes that
state visible to operators and downstream receipt tooling.

## Behavior

Contract summaries now include entries like:

```json
{
  "architecture_support": [
    {
      "arch": "x86",
      "kernel": "i2_s",
      "status": "supported_reference"
    }
  ]
}
```

The unsupported 3B x86 `I2_S` contract therefore reports
`status=unsupported_upstream` in the same JSON surface that also reports
`supported_artifact=false` for byte verification.

## Claim Boundary

This PR exposes contract metadata only. It does not change runtime inference,
tokenizer behavior, loader behavior, transformer math, QK256, CUDA routing,
dense GGUF inference, server behavior, speed claims, or residency claims.

