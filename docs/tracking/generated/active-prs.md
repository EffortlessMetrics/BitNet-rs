<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m3-macbook-air | M3MBA-026 | #30 | `codex/apple-m3-macbook-air/M3MBA-020-server-device-labels` | Harden the shared server device-model receipt path so apple-m3-air-metal, apple-m3-air-mpsgraph, and apple-m3-air-cpu-neon preserve their configured backend identity in server shared-engine receipts even when the active model reports a CUDA device, without collapsing into generic active-model device labels or M4 proof wording. |
| intel-258v-platform | SWARM-LNL258V-POWER-006-PREFLIGHT-001 | #37 | `codex/swarm-lnl258v-power006-blocked-telemetry-refresh` | Refresh the committed Lunar Lake low_power battery telemetry blocked receipt in swarm by rerunning telemetry-context --require-battery on the current machine, record that the strict command still exits blocked because AC power is inferred, and keep POWER-006, low_power route status, route promotion, inference, speedup, power-advantage, measured-temperature, native accelerator, broad quality, and BitNet QK256/I2_S behavior claims unchanged. |
