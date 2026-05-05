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
- Crate collapse must preserve modularity inside destination crates.

## Workstreams

1. Truth boundary
2. Crate consolidation inventory
3. Leaf crate collapse
4. Domain crate collapse
5. CPU runtime proof
6. Server real inference
7. GPU validation
8. Publish alignment

## Source of truth

- Work items: `workstream-ledger.yaml`
- Crate destinations: `crate-consolidation-map.yaml`
- Feature cleanup: `feature-lattice-map.yaml`
- Backend status: `backend-status.yaml`
- Verification commands: `verification-gates.md`
- PR rules: `pr-playbook.md`
