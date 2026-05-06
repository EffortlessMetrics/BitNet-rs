# Codex Campaign Index

Use campaign-local goals and trackers for BitNet alignment work.

Campaign goals:

- `apple-m4`: `.codex/campaigns/apple-m4/goal.md`
- `cpu-proof`: `.codex/campaigns/cpu-proof/goal.md`
- `intel-a770`: `.codex/campaigns/intel-a770/goal.md`
- `intel-npu`: `.codex/campaigns/intel-npu/goal.md`
- `crate-collapse`: `.codex/campaigns/crate-collapse/goal.md`

Campaign manifests:

- `apple-m4`: `docs/tracking/campaigns/apple-m4/active.toml`
- `cpu-proof`: `docs/tracking/campaigns/cpu-proof/active.toml`
- `intel-a770`: `docs/tracking/campaigns/intel-a770/active.toml`
- `intel-npu`: `docs/tracking/campaigns/intel-npu/active.toml`
- `crate-collapse`: `docs/tracking/campaigns/crate-collapse/active.toml`

Rules:

- Pick only from the selected campaign.
- Check GitHub for an existing PR with the item ID before starting.
- One work item, one PR.
- Respect `stackable`, `requires_human_merge`, and `blocked_by`.
- Do not edit global dashboards by hand.
- Do not delete hardware lane visibility.
- Add follow-up items instead of broadening a PR.

Transition note:

`docs/tracking/bitnet-alignment/workstream-ledger.yaml` and `docs/tracking/bitnet-alignment/status.md` remain transition trackers until generator/checker tooling makes campaign manifests authoritative. New planning work should prefer `docs/tracking/campaigns/**` and generated dashboards.
