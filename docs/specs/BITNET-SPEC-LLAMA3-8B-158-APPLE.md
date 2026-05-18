# BITNET-SPEC-LLAMA3-8B-158-APPLE

Status: proposed
Owner: apple-silicon
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no Apple claim before exact Apple receipts
Policy impact: no policy exception

## Apple paths

Candidate Apple work includes MacBook artifact inventory, MacBook reference
runner proof, M4 CPU/NEON `I2_S` answer proof, M4 CPU/NEON `TL1` answer proof,
and Metal phase candidates only after CPU/NEON proof.

## Required receipt distinctions

Apple receipts must record machine identity, chip identity, OS/toolchain,
artifact identity, tokenizer/prompt profile, selected backend, selected kernel,
route family, fallback status, generated IDs, decoded text, and whether the
receipt counts for MacBook, M4 Mac mini, Metal phase, or full Metal inference.

## Hard rules

MacBook proof does not prove M4 Mac mini. M4 proof does not prove MacBook. Apple
CPU/NEON proof does not prove Metal. Metal phase proof does not prove full Metal
inference.
