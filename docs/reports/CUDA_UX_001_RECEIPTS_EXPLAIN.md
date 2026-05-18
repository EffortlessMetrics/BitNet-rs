# CUDA-UX-001: Receipt Explanation

## Summary

`CUDA-UX-001` adds a user-facing receipt explanation surface:

```powershell
bitnet receipts explain <receipt.json>
bitnet receipts explain --latest
bitnet receipts explain <receipt.json> --json
bitnet receipts explain --latest --format json
```

The command reads existing BitNet-rs JSON receipts and prints a compact proof
summary for operators. It does not validate one narrow schema. Instead, it
extracts common proof fields across strict BitNet CUDA, answer-corpus,
warm-session, benchmark, and dense regular-LLM CUDA receipts:

- artifact kind and claim;
- model identity;
- requested and selected backend;
- runtime API and fallback status;
- model-aware execution-plan route;
- kernel IDs;
- answer, benchmark, and parity quality signals;
- timing, transfer, and kernel-time fields where present;
- QK256 and dense residency claim fields where present;
- claim limits such as `speedup_claim`, benchmark qualification, dense GGUF
  inference, full CUDA residency, and BitNet packed proof boundaries.

## Claim Boundary

May claim:

- existing BitNet CUDA and dense CUDA receipts can be summarized for operator
  UX;
- `--latest` can select the newest local JSON receipt under
  `target/bitnet/receipts`;
- `--json` and `--format json` emit the normalized explanation object for
  tooling.

Must not claim:

- new inference behavior;
- new tokenizer, loader, transformer, kernel, benchmark, or server behavior;
- dense GGUF inference;
- Qwen one-token, decode, or chat;
- BitNet packed proof from dense CUDA receipts;
- speedup or full CUDA residency;
- full schema-specific validation for every receipt kind.

Schema-specific validation remains owned by the existing receipt validators and
campaign gates. This command is a readable proof cockpit, not a replacement for
strict receipt acceptance.

## Examples

Strict BitNet CUDA warm-session benchmark:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  receipts explain `
  ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json
```

Dense regular-LLM CUDA fixture receipt:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  receipts explain `
  ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json `
  --format json
```

## Next Step

`CUDA-UX-002` should use the same explanation object to improve the strict
`ask` and warm-session proof summaries after generation, while keeping the
receipt file as the authority.
