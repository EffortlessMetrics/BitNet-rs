# bitnet-rs Alignment Burndown

This tracker coordinates the pre-publish stabilization of bitnet-rs.

## Objective

Reduce public crate surface, preserve SRP through modules, make CPU inference proof-backed, remove fake runtime paths, and keep CI green PR by PR.

## Current rules

- Default library features are empty.
- Always test with explicit features.
- Public claims require receipts.
- Server inference must not simulate output.
- GGUF minimal fallback must be explicit.
- GPU backends are scaffolded unless receipt-backed.
- Hardware validation is lane-based; generic accelerator claims are not allowed.
- BitNet proof must record model, tokenizer, quantization, kernel family, runtime phase, reference path, and fallback status.
- Crate collapse must preserve modularity inside destination crates.

## Workstreams

1. Truth boundary
2. Crate consolidation inventory
3. Leaf crate collapse
4. Domain crate collapse
5. CPU runtime proof
6. Legacy mobile CPU validation
7. Desktop CPU validation
8. Server real inference
9. GPU validation
10. Apple Silicon validation
11. Intel Arc GPU validation
12. Lunar Lake 258V platform validation
13. Intel NPU validation
14. NVIDIA CUDA validation
15. Publish alignment

## Source of truth

- Work items: `workstream-ledger.yaml`
- Crate destinations: `crate-consolidation-map.yaml`
- Feature cleanup: `feature-lattice-map.yaml`
- Backend status: `backend-status.yaml`
- Verification commands: `verification-gates.md`
- PR rules: `pr-playbook.md`
- Review rules: `codex-review-guide.md`

Shared hardware contracts:

- `../../hardware/HARDWARE_MATRIX.md`
- `../../hardware/PROOF_STAGES.md`
- `../../hardware/LANE_OWNERSHIP.md`
- `../../hardware/BENCHMARK_PROTOCOL.md`
- `../../hardware/machine-profile.schema.yaml`

BitNet proof contracts:

- `../../bitnet/BITNET_MODEL_CONTRACT.md`
- `../../bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `../../bitnet/BITNET_KERNEL_MATRIX.md`
- `../../bitnet/BITNET_RUNTIME_PHASES.md`
- `../../bitnet/BITNET_REFERENCE_RUNS.md`
- `../../bitnet/BITNET_RECEIPT_FIELDS.md`
- `../../bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `../../bitnet/BITNET_PARITY_TOLERANCES.md`
