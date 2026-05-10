# CUDA-UX-006 Benchmark Receipt Report

`CUDA-UX-006` adds a governed receipt-reporting mode to the legacy benchmark
command:

```powershell
bitnet bench --device cuda --model <path> --cuda-benchmark-receipt <receipt.json>
```

The command validates that the input is a recognized CUDA benchmark
qualification receipt, requires `fallback_used=false`, requires CUDA backend
identity, and renders the evidence as text, JSON, or CSV. Without
`--cuda-benchmark-receipt`, CUDA benchmark requests still fail closed instead
of falling through to the legacy simulated CPU benchmark path.

Claim boundary:

```text
fresh_cuda_benchmark_executed=false
speedup_claim=false unless present and benchmark-qualified in the input receipt
full_cuda_residency_claimed=false unless present in the input receipt
server_ready_claimed=false
bitnet_packed_i2s_qk256_proof=false for dense receipts
```

This is a UX bridge for existing governed benchmark receipts, not a new
benchmark runner and not a speedup qualification.
