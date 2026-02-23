# CI Migration to Nix

## Overview

This document describes the future GitHub Actions CI migration to Nix-based validation.
**Status**: GitHub Actions is currently offline; this is the planned approach for when it returns.

## Current State (Offline CI)

- GitHub Actions workflows disabled
- Local validation via `./scripts/ci-local.sh`
- Nix flake checks provide hermetic validation (`nix flake check`)

## Future CI Design (Nix-First)

When GitHub Actions returns, the **only** CI job needed is:

```yaml
name: Nix CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  nix-checks:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        check:
          - workspace            # Full workspace validation
          - bitnet-server-receipts  # Receipts validation
          - nextest              # Fast test runner

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install Nix
        uses: cachix/install-nix-action@v27
        with:
          nix_path: nixpkgs=channel:nixos-24.05

      - name: Run ${{ matrix.check }} validation
        run: nix build .#checks.${{ matrix.check }}

      - name: Upload check logs
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: check-logs-${{ matrix.check }}
          path: result/ci-*.log
          retention-days: 7
```

## Why This Design?

### Single Source of Truth

All build configuration lives in `flake.nix`:
- Rust versions (stable + MSRV)
- Native dependencies (pkg-config, openssl, cmake, etc.)
- Environment variables (RUSTC_WRAPPER, CARGO_INCREMENTAL, etc.)
- Feature flags and build commands

No duplication between CI YAML and local validation.

### Hermetic Builds

Nix guarantees:
- ✅ Same toolchain across all machines
- ✅ No hidden dependencies
- ✅ Reproducible artifacts
- ✅ Local/CI parity (if it works locally, it works in CI)

### Minimal CI Maintenance

When you change:
- Rust version → Update `flake.nix` only
- Dependencies → Update `nativeDeps` in `flake.nix` only
- Build commands → Update `scripts/ci-local.sh` only

CI workflow stays unchanged.

### Failure Debugging

When a check fails:
1. CI uploads `result/ci-*.log` artifacts
2. Developer downloads log and inspects
3. Developer can reproduce **exactly** with `nix build .#checks.<name>`

No "works on my machine" divergence.

## Migration Checklist (Future Work)

When GitHub Actions returns:

- [ ] Create `.github/workflows/nix-ci.yml` with the job above
- [ ] Add Cachix setup for binary caching (optional, speeds up builds)
- [ ] Remove old workflow files (e.g., `ci-core.yml`, `model-gates.yml`)
- [ ] Update README CI badge to point to new workflow
- [ ] Test on a feature branch first
- [ ] Merge to main and verify all checks pass

## Cachix Setup (Optional)

For faster CI builds, add a binary cache:

```yaml
- name: Setup Cachix
  uses: cachix/cachix-action@v14
  with:
    name: bitnet-rs
    authToken: '${{ secrets.CACHIX_AUTH_TOKEN }}'
```

This caches Nix build artifacts so CI doesn't rebuild everything on every push.

**Prerequisites:**
1. Create Cachix account and cache at https://app.cachix.org
2. Generate auth token
3. Add `CACHIX_AUTH_TOKEN` to GitHub repository secrets

## Local Development Parity

The key principle: **Nix is the source of truth.**

| Command | Purpose | Where |
|---------|---------|-------|
| `nix develop` | Enter dev environment | Local only |
| `nix flake check` | Run all checks (hermetic) | Local + CI |
| `nix build .#checks.workspace` | Run workspace check | Local + CI |
| `nix build .#checks.bitnet-server-receipts` | Run receipts check | Local + CI |
| `nix build .#checks.nextest` | Run nextest validation | Local + CI |

CI simply calls `nix build .#checks.*` — same as local.

## Troubleshooting

### "Check failed on CI but passes locally"

This should never happen with Nix-based CI. If it does:

1. Verify you're using `nix build .#checks.*`, not `cargo` directly
2. Check Nix version matches CI (`nix --version`)
3. Ensure `flake.lock` is committed (reproducibility guarantee)

### "CI is slow"

Options:
1. Add Cachix binary cache (recommended)
2. Use GitHub Actions cache for Nix store
3. Split checks into parallel matrix jobs (already done in example above)

### "Need to add a new check"

1. Add check derivation to `flake.nix` under `checks = { ... }`
2. Add corresponding mode to `scripts/ci-local.sh` (if needed)
3. Add check name to CI matrix in `.github/workflows/nix-ci.yml`

## See Also

- `docs/kv-pool/NIX_FLAKE_USAGE.md` - Complete Nix flake documentation
- `docs/kv-pool/LOCAL_VALIDATION_WORKFLOW.md` - Local validation workflow
- `flake.nix` - Source of truth for all build configuration
- `scripts/ci-local.sh` - Local validation script (Nix-independent interface)

---

**Last Updated**: 2025-11-20 (GitHub Actions currently offline)
**Status**: Design complete, awaiting GitHub Actions restoration
