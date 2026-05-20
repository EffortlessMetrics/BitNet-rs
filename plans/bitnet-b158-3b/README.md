# BitNet b1.58 3B TL candidate plan

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../../docs/proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [artifact](../../docs/specs/BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md), [conversion](../../docs/specs/BITNET-SPEC-B158-3B-CONVERSION.md), [TL layout](../../docs/specs/BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [tokenizer/prompt](../../docs/specs/BITNET-SPEC-B158-3B-TOKENIZER-PROMPT.md), [quality](../../docs/specs/BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md), [CPU](../../docs/specs/BITNET-SPEC-B158-3B-CPU.md), [CUDA](../../docs/specs/BITNET-SPEC-B158-3B-CUDA.md), [Apple](../../docs/specs/BITNET-SPEC-B158-3B-APPLE.md), [performance](../../docs/specs/BITNET-SPEC-B158-3B-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [implementation-plan.md](implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; registers a guarded candidate lane
Policy impact: no policy exception

This directory sequences the `1bitLLM/bitnet_b1_58-3B` TL candidate lane. The
lane is artifact-safe first, TL-correct second, answer-good third, and fast
fourth.

Start with [`implementation-plan.md`](implementation-plan.md) for PR order,
proof commands, allowed paths, and claim boundaries.
