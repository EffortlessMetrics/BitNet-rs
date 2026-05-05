# Verification Gates

## Gate 0 — Formatting only

```bash
cargo fmt --all -- --check
```

## Gate 1 — Focused crate

```bash
cargo fmt --all -- --check
cargo clippy --locked -p <crate> --all-targets --no-default-features --features cpu -- -D warnings
cargo test --locked -p <crate> --no-default-features --features cpu
```
