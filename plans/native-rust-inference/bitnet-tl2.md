# BitNet TL/TL2 And GPU-int2 Diagnostic Lanes

BitNet TL1/TL2 and GPU-int2 paths are registered candidates or diagnostic
lanes. They do not inherit official I2_S/QK256 proof.

## Work item: BITNET-TL2-001

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: TL/TL2 artifact contracts
Blocked by: native inference plan

### Goal

Record TL/TL2 and GPU-int2 as separate proof families with explicit unsupported
routes where proof is absent.

### Production delta

Status and model coverage surfaces reject inherited I2_S/QK256 proof for TL/TL2
and GPU-int2.

### Non-goals

No kernel work and no model promotion.

### Acceptance

Rows or docs say which proof is missing and which claims remain forbidden.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Remove diagnostic lane additions and keep existing registered state.


## Update 2026-05-19

TL2 now has dedicated source-of-truth docs under `docs/bitnet/tl2/` and `plans/tl2/`.
