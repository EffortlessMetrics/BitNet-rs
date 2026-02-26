# BitNet-rs Quick Start (INVALID Pattern)

## ANTI-PATTERN: Bare --features without --no-default-features

### Installation

```bash
git clone https://github.com/YOUR_ORG/BitNet-rs.git
cd BitNet-rs
cargo build --features cpu  # WRONG: Missing --no-default-features
```

### Running Tests

```bash
cargo test --features cpu  # WRONG: Bare features flag
```

### GPU Build

```bash
cargo build --features gpu  # WRONG: No default features specification
```

**Problems with these examples:**

1. **Inconsistent behavior**: Users may not understand why builds behave differently
2. **No explicit defaults**: Doesn't make clear that default features are empty
3. **Hidden assumptions**: Assumes readers know about Cargo default feature behavior

### Correct Pattern

Always use:

```bash
cargo build --no-default-features --features cpu
cargo test --no-default-features --features cpu
cargo build --no-default-features --features gpu
```

## Summary

This documentation will **FAIL AC7 validation** because:
- Uses `--features` without `--no-default-features`
- No explanation of feature flag policy
- Inconsistent with BitNet-rs standardized patterns
