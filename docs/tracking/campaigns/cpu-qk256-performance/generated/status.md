<!-- GENERATED: do not edit by hand. Run cargo run --no-default-features -p xtask --no-default-features -- campaign generate. -->
# CPU QK256 performance Campaign Status

- Campaign: `cpu-qk256-performance`
- State: `active`
- Objective: Move CPU BitNet work from strict proof surfaces into scalar, packed-layout, AVX2, and sustained benchmark evidence without hiding fallback behavior.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| KBL8250U-003 | merged | #3785 | `codex/cpu-qk256-performance/KBL8250U-003-avx2-proof-artifact` | Prove i5-8250U scalar and AVX2 dispatch with receipt-backed selected CPU kernel identity and no GPU/NPU fallback. |
| KBL8250U-004 | ready | TBD | `codex/cpu-qk256-performance/KBL8250U-004-strict-proof-run` | Add an i5-8250U strict CPU proof run with selected backend, no fallback, model hash, timing, and thermal context. |

## Hard Constraints

- Do not claim performance before strict proof receipts exist.
- Do not mix GPU, NPU, CUDA, OpenCL, or Metal execution into CPU proof.
- Do not treat helper-only AVX2 code as real inference.
- Do not report short turbo behavior as sustained performance.
