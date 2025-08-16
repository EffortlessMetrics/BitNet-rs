# Writing stable tests

## Env mutations
- Take `let _g = env_guard();` at the top of any test that touches `std::env`.
- Save & restore original env values if you temporarily override them.

```rust
#[test]
fn my_env_test() {
    let _g = env_guard(); // serialize env access across tests
    std::env::set_var("BITNET_NO_NETWORK", "true");
    // ...
}
```

## Double‑clamp detectors
We keep two micro tests to prevent regression:
- `test_fast_feedback_is_applied_once`
- `test_no_double_clamp_on_resources`

These fail if someone accidentally re‑introduces clamps inside the manager.

## MB→bytes
Use `BYTES_PER_MB` in assertions (e.g., `500 * BYTES_PER_MB`)—no raw `1024*1024` literals.

## CI smoke
Run minimal and fixtures paths locally:

```bash
cargo build -p bitnet-tests --lib
cargo test  -p bitnet --test test_configuration_scenarios

cargo build -p bitnet-tests --lib --features full-framework
cargo test  -p bitnet --features fixtures --test test_configuration_scenarios
```

## Quick regressions (local)

```bash
# Ensure env mutations are guarded
rg -n "env::(set_var|remove_var)" tests | rg -v "env_guard"

# Ensure no raw 1024*1024 conversions
rg -n "1024 \* 1024" tests

# Ensure clamps still only live in the wrapper
rg -n "target_feedback_time|include_artifacts|max_disk_cache_mb|generate_coverage|network_access|max_parallel" tests/common/config_scenarios.rs
```