# Small LLM Candidates

Start with one small LLM at a time. Do not batch-promote candidate families.

## Recommended Order

1. Llama 3.2 1B
2. SmolLM2 1.7B
3. Llama 3.2 3B
4. Gemma small
5. Phi small

## Work item: SMALL-LLM-001

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: per-model artifact contracts
Blocked by: native inference plan

### Goal

Select the first small dense LLM candidate and open a per-model artifact
contract PR.

### Production delta

One model gets a registered-to-structural path without promoting other
candidates.

### Non-goals

No CPU, CUDA, speed, server, or product CLI claim.

### Acceptance

Chosen model records artifact contract, tokenizer/prompt authority plan, CPU
sanity command, all-layer plan placeholder, and forbidden claims.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Remove the candidate row or demote it back to registered.

## Shared Ladder

Each selected model follows:

```text
artifact contract
CPU sanity
all-layer plan
model-boundary fixtures
KV/sampling policy
one-token CUDA
short decode
warm session
benchmark review
product UX review
```

Dense small-LLM CUDA proof remains separate from dense SLM CUDA and BitNet
QK256 proof.
