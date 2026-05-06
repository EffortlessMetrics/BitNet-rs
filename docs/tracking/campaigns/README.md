# Campaign Trackers

Campaign-local trackers are the control cards for parallel BitNet work.

Each campaign directory contains:

- `CAMPAIGN.md`: human-readable objective, end state, constraints, work items, and review policy.
- `active.toml`: machine-readable current work items, commands, allowed paths, forbidden paths, and claim boundaries.
- `events/`: append-only lifecycle events such as `in_progress`, `pr_open`, `merged`, and `closeout`.
- `generated/`: generated dashboards. Do not edit these files by hand.

Use `cargo run -p xtask --no-default-features -- campaign check <campaign>` before opening a tracker PR, and `cargo run -p xtask --no-default-features -- campaign generate` to refresh dashboards.

Normal item PRs should not hand-edit legacy global tracker files under `docs/tracking/bitnet-alignment/`. Those files remain transition surfaces until they are frozen or generated.
