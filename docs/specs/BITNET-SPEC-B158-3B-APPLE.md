# BITNET-SPEC-B158-3B-APPLE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B CPU](BITNET-SPEC-B158-3B-CPU.md), [3B TL layout](BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [3B performance](BITNET-SPEC-B158-3B-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no Apple support promotion until receipts pass
Policy impact: no policy exception

## Purpose

Define the Apple proof path for the 3B ARM TL1 lane. Apple is a natural first
hardware candidate because upstream lists ARM TL1 for this model, but BitNet-rs
must still prove artifact inventory, conversion/runner authority, tokenizer /
prompt authority, reference-good output, and TL1 CPU/NEON receipts.

## Required path

```text
MacBook artifact inventory
→ storage/free-space receipt
→ TL1 conversion or runner proof
→ reference-good output
→ M4 or MacBook CPU/NEON TL1 structural loader
→ TL1 scalar/NEON fixtures
→ strict Apple answer corpus
→ warm-session
→ benchmark
```

## Required Apple receipts

Apple receipts must record:

- machine identity and whether it is MacBook, M4 Mac Mini, or another exact
  profile;
- storage/free-space context for large safetensors or converted artifacts;
- source revision, artifact hash, tokenizer hashes, and prompt policy;
- route `tl1`;
- selected backend, for example `apple-macbook-cpu-neon`;
- selected kernel, for example `tl1-scalar-reference-gemv` or
  `tl1-neon-reference-gemv`;
- fallback status with `fallback_used = false` for strict proof;
- generated token IDs and decoded text;
- `speedup_claim = false` until benchmark review.

## Hard rules

- MacBook proof does not prove M4 Mac Mini.
- M4 proof does not prove MacBook.
- Apple CPU/NEON proof does not prove Metal.
- Metal phase proof does not prove full Metal inference.
- ARM TL2 is unsupported for this model unless the compatibility ledger changes
  through a separate authority update.
