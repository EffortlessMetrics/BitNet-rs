# CI Guardrails (Stub)

**Note**: This is a placeholder stub. The complete CI guardrails documentation will be added in PR #510.

## Quick Reference

Until the full documentation lands, see:

* **CONTRIBUTING.md** - Contribution guidelines and workflow
* **GitHub Actions workflows** in `.github/workflows/` for CI implementation details

## Key Guardrails (Preview)

The complete guide will cover:

1. **SHA Pin Enforcement** - All GitHub Actions pinned to full commit SHAs
2. **MSRV Single-Sourcing** - Rust version consistency across workspace
3. **`--locked` Policy** - Deterministic dependency resolution
4. **Runner Pinning** - Explicit runner version specification (e.g., `ubuntu-22.04`)
5. **Receipt Workflow Hygiene** - Honest compute validation and receipt verification
6. **Validation Gates** - Quality gates for merge readiness

---

For the complete, comprehensive CI guardrails documentation, see PR #510.
