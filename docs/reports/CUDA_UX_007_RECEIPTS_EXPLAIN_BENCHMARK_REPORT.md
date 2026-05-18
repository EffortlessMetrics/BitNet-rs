# CUDA-UX-007 Receipts Explain Benchmark Report

`CUDA-UX-007` extends `bitnet receipts explain` for governed CUDA benchmark
qualification receipts.

The command now surfaces:

```text
qualification status
benchmark_qualified_speedup
accepted and blocked profiles
per-profile CPU/CUDA mean total timing
per-profile H2D/D2H timing evidence
H2D model-load envelope source and scope
pure H2D timing blockers
speedup blockers
```

This keeps benchmark interpretation visible in the same proof surface users
already use for CUDA ask/chat receipts:

```powershell
bitnet receipts explain <benchmark-qualification-receipt.json>
bitnet receipts explain <benchmark-qualification-receipt.json> --json
bitnet receipts explain <benchmark-qualification-receipt.json> --format json
```

Claim boundary:

```text
fresh_cuda_benchmark_executed=false
speedup_claim=false unless present and benchmark-qualified in the input receipt
full_cuda_residency_claimed=false unless present in the input receipt
server_ready_claimed=false
bitnet_packed_i2s_qk256_proof=false for dense receipts
```

This is receipt explanation only. It does not run benchmarks, change kernel
math, change inference behavior, or upgrade speed claims.
