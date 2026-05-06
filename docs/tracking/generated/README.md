# Generated Tracking Dashboards

This directory is reserved for generated alignment dashboards.

Generated outputs:

- `global-dashboard.md`
- `active-prs.md`
- `lane-dashboard.md`
- `blocked-items.md`

Do not hand-edit generated dashboard files. Source data comes from campaign-local `active.toml` files and append-only event files.

Refresh dashboards with:

```bash
cargo run -p xtask --no-default-features -- campaign generate
```

Check for stale dashboards with:

```bash
cargo run -p xtask --no-default-features -- campaign generate --check
```
