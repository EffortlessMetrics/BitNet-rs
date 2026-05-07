# Codex Campaign Index

Use campaign-local goals and trackers for BitNet alignment work.

Campaign goals:

- `apple-m4`: `.codex/campaigns/apple-m4/goal.md`
- `apple-m4-local-answer`: `.codex/campaigns/apple-m4-local-answer/goal.md`
- `apple-m4-operational`: `.codex/campaigns/apple-m4-operational/goal.md`
- `cpu-proof`: `.codex/campaigns/cpu-proof/goal.md`
- `cpu-qk256-performance`: `.codex/campaigns/cpu-qk256-performance/goal.md`
- `intel-a770`: `.codex/campaigns/intel-a770/goal.md`
- `intel-npu`: `.codex/campaigns/intel-npu/goal.md`
- `intel-258v-platform`: `.codex/campaigns/intel-258v-platform/goal.md`
- `nvidia-5070ti`: `.codex/campaigns/nvidia-5070ti/goal.md`
- `amd-cpu-baselines`: `.codex/campaigns/amd-cpu-baselines/goal.md`
- `crate-collapse`: `.codex/campaigns/crate-collapse/goal.md`
- `server-real-inference`: `.codex/campaigns/server-real-inference/goal.md`
- `ci-coverage`: `.codex/campaigns/ci-coverage/goal.md`
- `tracker-infra`: `.codex/campaigns/tracker-infra/goal.md`

Campaign manifests:

- `apple-m4`: `docs/tracking/campaigns/apple-m4/active.toml`
- `apple-m4-local-answer`: `docs/tracking/campaigns/apple-m4-local-answer/active.toml`
- `apple-m4-operational`: `docs/tracking/campaigns/apple-m4-operational/active.toml`
- `cpu-proof`: `docs/tracking/campaigns/cpu-proof/active.toml`
- `cpu-qk256-performance`: `docs/tracking/campaigns/cpu-qk256-performance/active.toml`
- `intel-a770`: `docs/tracking/campaigns/intel-a770/active.toml`
- `intel-npu`: `docs/tracking/campaigns/intel-npu/active.toml`
- `intel-258v-platform`: `docs/tracking/campaigns/intel-258v-platform/active.toml`
- `nvidia-5070ti`: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- `amd-cpu-baselines`: `docs/tracking/campaigns/amd-cpu-baselines/active.toml`
- `crate-collapse`: `docs/tracking/campaigns/crate-collapse/active.toml`
- `server-real-inference`: `docs/tracking/campaigns/server-real-inference/active.toml`
- `ci-coverage`: `docs/tracking/campaigns/ci-coverage/active.toml`
- `tracker-infra`: `docs/tracking/campaigns/tracker-infra/active.toml`

Rules:

- Pick only from the selected campaign.
- Check GitHub for an existing PR with the item ID before starting.
- One work item, one PR.
- Respect `stackable`, `review_mode`, `merge_policy`, `human_gate`, and `blocked_by`.
- Do not edit global dashboards by hand.
- Do not delete hardware lane visibility.
- Add follow-up items instead of broadening a PR.
- Use `cargo run -p xtask --no-default-features -- campaign check <campaign>` before opening a tracker PR.
- Use `cargo run -p xtask --no-default-features -- campaign generate` to refresh dashboards; do not hand-edit generated files.

Lunar Lake platform priority:

- Core Ultra 7 258V CPU is the BitNet CPU lead for strict CPU proof validation, scalar-vs-AVX2 answer parity, phase receipts, and same-machine CPU reference artifacts.
- Intel NPU and Arc 140V are significant secondary lanes, but they must compare against 258V CPU reference receipts before BitNet-adjacent parity claims.
- NPU work must not claim full BitNet inference, QK256 decode, acceleration, or CPU fallback as NPU proof.
- Arc 140V work must prove native OpenCL execution before any BitNet kernel claim.
- i5-8250U owns the SLM CPU lane and remains a legacy/low-power BitNet comparison lane; it must not block new BitNet CPU sequencing.
- Ryzen 9 9950X3D may own AVX-512 BitNet CPU validation when needed, but its primary machine focus remains RTX 5070 Ti.
- Ryzen 7 5700X may support AVX2 desktop validation when needed, but its primary machine focus remains A770.

Transition note:

`docs/tracking/bitnet-alignment/workstream-ledger.yaml` and `docs/tracking/bitnet-alignment/status.md` are transition surfaces. Normal item PRs should use campaign-local manifests and events once generated dashboards are available.
