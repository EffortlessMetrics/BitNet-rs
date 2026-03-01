# Repo Maintenance Instruments

This directory contains generated, diffable maintenance artifacts.

## Artifacts

- `deps.mmd`: workspace crate dependency graph (Mermaid).
- `public_api.md`: snapshot of top-level public items for selected public crates.
- `sbom.json`: machine-readable workspace software bill of materials.
- `churn.md`: 90-day churn hotspots.
- `status.md`: repository invariant/status summary (MSRV, workflow gates, counts).

## Generation

Run locally:

```bash
python3 scripts/maintenance/generate_instruments.py
```

In CI, `.github/workflows/repo-maintenance-instruments.yml` regenerates these artifacts and:

- uploads them on pull requests,
- commits updates to the `generated` branch on schedule and manual runs,
- runs dependency policy checks (`cargo deny`).
