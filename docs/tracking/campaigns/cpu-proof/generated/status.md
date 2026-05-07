<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# BitNet CPU proof Campaign Status

- Campaign: `cpu-proof`
- State: `active`
- Objective: Make real BitNet CPU inference strict, receipt-backed, and measurable without routing around model, tokenizer, layout, or fallback truth.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| CPU-BITNET-000 | merged | #3642 | `codex/cpu-proof/CPU-BITNET-000-path-plan` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Document the real BitNet CPU path implementation plan and sequence strict loader, tokenizer, layout, scalar, AVX2, receipts, and benchmarks. |
| CPU-BITNET-001 | merged | #3651 | `codex/cpu-proof/CPU-BITNET-001-loader-authority` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Strict CPU inference has one authoritative real GGUF loader path for BitNet models, and minimal fallback is impossible in strict proof mode. |
| CPU-BITNET-002 | merged | #3680 | `codex/cpu-proof/CPU-BITNET-002-tokenizer-authority` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Strict tokenizer resolution uses explicit override, GGUF metadata, sibling tokenizer assets, then strict failure. |
| CPU-BITNET-003 | merged | #3690 | `codex/cpu-bitnet-003-canonical-packed-layout` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Canonical block geometry, alignment, stride, and row/block iteration API are defined. |
| CPU-BITNET-004 | merged | #3696 | `codex/cpu-bitnet-004-scalar-packed-truth` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Canonical scalar packed QK256 GEMV/GEMM kernels are deterministic correctness oracles for decode and prefill. |
| CPU-BITNET-005a | merged | #3735 | `codex/cpu-bitnet-005a-avx2-fma-gating` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | AVX2/FMA feature plumbing is explicit, FMA-using QK256 helpers are target-feature gated, and scalar fallback remains unchanged. |
| CPU-BITNET-005b | merged | #3748 | `codex/cpu-bitnet-005b-kernel-selection` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | QK256 decode GEMV selection reports requested kernel, selected kernel, fallback status, fallback reason, and CPU features with strict AVX2 fallback failure. |
| CPU-BITNET-005c | merged | #3753 | `codex/cpu-bitnet-005c-avx2-parity-hardening` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | AVX2 decode GEMV parity is hardened against scalar across rows, tail-column shapes, deterministic patterns, and repeated-run equality. |
| CPU-BITNET-006 | merged | #3793 | `codex/cpu-bitnet-006-cpu-decode-step` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Embedding gather, RMSNorm/subln, RoPE, attention scores, softmax, A/V, FFN, KV-cache, output head, logits handoff, and sampling handoff are authoritative for CPU decode; prefill and decode scheduling are separated; one real-model decode step can run with real tensors; missing ops fail explicitly; KV-cache append/read is deterministic. |
| CPU-BITNET-007 | merged | #3800 | `codex/cpu-bitnet-007-strict-receipts` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Strict CPU proof receipts tie loader mode, tokenizer source, requested/selected backend, requested/selected kernel, CPU features, quant format, decode/prefill phase, prompt/generated tokens, and fallback status into one machine-checkable record; validation fails on hidden fallback, non-real GGUF loader mode, guessed tokenizer source, mock/diagnostic kernels, or requested AVX2 silently selecting scalar. |
| CPU-BITNET-008 | pr_open | #3864 | `codex/cpu-bitnet-008-phase-timing-accuracy` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | CPU proof benchmark profiles emit receipt-backed micro, layer, prefill, first-token, and decode measurements with selected backend/kernel, fallback status, workload shape, model identity, quant format, prompt/generated token counts, and hardware context. |

## Hard Constraints

- No GPU or NPU claims.
- No silent GGUF fallback.
- No performance claim without receipt artifacts.
- No helper-only SIMD work unless it is wired to real inference or explicitly scoped as preparation.
