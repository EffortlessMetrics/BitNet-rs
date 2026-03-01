# Repository Maintenance Instruments

This directory stores generated, reviewable maintenance artifacts.

## Artifacts

- `deps.dot` / `deps.svg`: workspace crate dependency graph.
- `public_api_snapshot.md`: public API item inventory for selected crates.
- `sbom.json`: dependency inventory (machine-readable SBOM-like snapshot).
- `churn.md`: 30/90-day git churn hotspots.
- `status.md`: repository invariants summary (MSRV, workflow inventory, contracts hint).

## Local generation

```bash
python3 scripts/maintenance/generate_repo_instruments.py
```

## CI generation

The workflow `.github/workflows/repo-maintenance-instruments.yml` runs this generator on schedule and on-demand, then uploads artifacts and (for scheduled runs) commits updates with `[skip ci]`.
