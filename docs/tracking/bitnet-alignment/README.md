# bitnet-rs Alignment Burndown

This tracker coordinates the pre-publish stabilization of bitnet-rs.

Use `docs/tracking/bitnet-alignment/workstream-ledger.yaml` as the control plane:
pick the next ready item with no unmet dependencies, stay inside its allowed paths,
avoid forbidden paths, update the tracker, and report only verification commands
actually run. The goal is to make bitnet-rs smaller, stricter, greener, and more
honest by collapsing excess public crate seams into SRP modules, preserving
explicit feature gates, removing fake or ambiguous runtime paths, and requiring
receipt-backed proof for working claims. Keep the sequence disciplined: truth
boundary first, then crate inventory, then consolidation, then CPU runtime proof,
then server inference and GPU validation; when work grows beyond the item, add a
follow-up ledger entry instead of broadening the PR.

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

## Sequence gates

- No crate movement before `TRUTH-001`, `TRUTH-002`, `TRUTH-003`, and `INV-001`
  are merged.
- No server real-inference work before fake/simulated paths are fenced and GGUF
  fallback is explicit.
- No GPU validation claims before backend identity and fallback semantics are clean
  and a CPU proof path exists.

## Source of truth

- Work items: `workstream-ledger.yaml`
- Crate destinations: `crate-consolidation-map.yaml`
- Feature cleanup: `feature-lattice-map.yaml`
- Backend status: `backend-status.yaml`
- Verification commands: `verification-gates.md`
- PR rules: `pr-playbook.md`
- Review rules: `codex-review-guide.md`
