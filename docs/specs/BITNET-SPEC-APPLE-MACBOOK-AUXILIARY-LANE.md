# BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md), [Apple reproducible run identity](BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; MacBook lane is auxiliary
Policy impact: no policy exception

## Purpose

Use MacBook hardware for larger-artifact and longer-soak Apple work without
corrupting M4 Mac Mini proof.

## MacBook lane can do

- larger BitNet artifact storage;
- larger dense/SLM artifact storage;
- external reference comparisons;
- longer soaks;
- BitNet model exploration;
- Metal/CPU parity experiments.

## MacBook lane cannot do

- prove M4 Mac Mini runtime behavior;
- replace M4 evidence;
- promote broad Apple Silicon support;
- silently change M4 defaults.

## Receipt distinction

MacBook receipts must distinguish themselves from M4 Mac Mini receipts:

```json
{
  "machine_id": "apple-silicon-macbook",
  "proof_family": "apple_macbook_cpu_neon_bitnet",
  "counts_as_m4_mac_mini_proof": false
}
```

MacBook evidence may inform future work, but promotion of M4 Mac Mini support
requires M4 Mac Mini receipts under the relevant M4 proof family.
