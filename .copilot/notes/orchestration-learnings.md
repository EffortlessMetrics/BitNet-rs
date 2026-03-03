# Orchestration Learnings

## Cherry-pick rebase: always restore main-owned files after
Files: perf_tracker.rs, shape_validator.rs, format_detector.rs, model_fingerprint.rs, xtask/main.rs

## CI timing: BT=25-35min, CI Core Success adds 1-5min after BT
Dispatch agents during wait periods to maximize throughput.

## Metal PRs are independent (merge parallel), NEON PRs touch mod.rs (merge sequential)

## Other teams break formatting/clippy constantly - always restore main files on rebase
