# Building BitNet-rs (INVALID Documentation Pattern)

## ANTI-PATTERN: Standalone cuda examples without context

### Build Commands (WRONG)

```bash
# GPU build (deprecated pattern - DO NOT USE)
cargo build --features cuda
```

**Problems with this pattern:**

1. **No `--no-default-features`**: Relies on defaults (which are empty)
2. **Uses `cuda` without context**: Doesn't explain it's an alias for `gpu`
3. **No migration path**: Doesn't guide users to the preferred `--features gpu` syntax

### Expected Documentation Pattern

Instead, use:

```bash
# GPU build (unified flag)
cargo build --no-default-features --features gpu
```

With a note: "Note: `cuda` is a temporary alias for `gpu` (will be removed in future versions)."

### Test Commands (WRONG)

```bash
# GPU tests
cargo test --features cuda
```

**Problems:**
- Missing `--no-default-features`
- Using deprecated `cuda` flag
- No workspace specification

### Correct Pattern

```bash
# GPU tests
cargo test --workspace --no-default-features --features gpu
```

## Summary

This documentation pattern will **FAIL AC7 validation** because:
- Uses `--features cuda` without alias context
- Missing `--no-default-features` flag
- No migration guidance for users
