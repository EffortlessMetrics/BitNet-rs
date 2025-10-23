# Documentation Code Example Fixes

## Missing Feature Flags in Code Examples

This document specifies all code examples that need feature flag updates for consistency.

### File 1: docs/troubleshooting/troubleshooting.md

#### Issue: compat-check/compat-fix Examples Missing Features

**Current Examples** (❌ Missing feature flags):

```bash
cargo run -p bitnet-cli -- compat-check model.gguf --verbose
cargo run -p bitnet-cli -- compat-check model.gguf --json > model_validation.json
cargo run -p bitnet-cli -- compat-fix model.gguf fixed_model.gguf
cargo run -p bitnet-cli -- compat-check fixed_model.gguf
RUST_LOG=debug cargo run -p bitnet-cli -- compat-check model.gguf 2>&1 | grep -i align
```

**Fixed Examples** (✅ With correct feature flags):

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check model.gguf --verbose
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check model.gguf --json > model_validation.json
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-fix model.gguf fixed_model.gguf
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check fixed_model.gguf
RUST_LOG=debug cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check model.gguf 2>&1 | grep -i align
```

**Changes Required**:
- Append `--no-default-features --features cpu,full-cli` after `-p bitnet-cli` on 5 lines
- No other changes needed

**Search Pattern** for exact locations:
```bash
grep -n "cargo run -p bitnet-cli -- compat-" docs/troubleshooting/troubleshooting.md
grep -n "RUST_LOG=debug cargo run -p bitnet-cli -- compat-check" docs/troubleshooting/troubleshooting.md
```

### File 2: docs/development/build-commands.md

#### Issue: Various CLI Examples Missing Features

**Current Examples** (❌ Missing feature flags):

```bash
cargo run -p bitnet-cli -- inspect --ln-stats --json models/your-model.gguf
cargo run -p bitnet-cli -- inspect --ln-stats --json models/your-model.gguf | jq '.status'
cargo run -p bitnet-cli -- inspect --ln-stats your-model.gguf
cargo run -p bitnet-st2gguf -- --input model.safetensors --output model.gguf --strict
```

**Fixed Examples** (✅ With correct feature flags):

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats --json models/your-model.gguf
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats --json models/your-model.gguf | jq '.status'
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats your-model.gguf
cargo run -p bitnet-st2gguf --no-default-features --features cpu -- --input model.safetensors --output model.gguf --strict
```

**Note**: For bitnet-st2gguf, the correct features may be `cpu,full-cli` or just `cpu` - verify in Cargo.toml

**Search Pattern** for exact locations:
```bash
grep -n "cargo run -p bitnet-cli -- inspect" docs/development/build-commands.md
grep -n "cargo run -p bitnet-st2gguf" docs/development/build-commands.md
```

### File 3: docs/development/validation-ci.md

#### Issue: st2gguf and inspect Examples Missing Features

**Current Examples** (❌ Missing feature flags):

```bash
cargo run -p bitnet-st2gguf -- --input model.safetensors --output model.gguf --strict
cargo run -p bitnet-cli -- inspect --ln-stats --json models/your-model.gguf
```

**Fixed Examples** (✅ With correct feature flags):

```bash
cargo run -p bitnet-st2gguf --no-default-features --features cpu -- --input model.safetensors --output model.gguf --strict
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- inspect --ln-stats --json models/your-model.gguf
```

**Search Pattern** for exact locations:
```bash
grep -n "cargo run -p bitnet-st2gguf\|cargo run -p bitnet-cli -- inspect" docs/development/validation-ci.md
```

### File 4: docs/troubleshooting/troubleshooting.md (Additional)

#### Issue: Model Validation Chain Missing Features

**Current Examples** (❌ Missing feature flags):

```bash
cargo run -p bitnet-cli -- compat-check model.gguf
- Always verify model integrity after download: `cargo run -p bitnet-cli -- compat-check model.gguf`
```

**Fixed Examples** (✅ With correct feature flags):

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check model.gguf
- Always verify model integrity after download: `cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- compat-check model.gguf`
```

## Implementation Checklist

### Update docs/troubleshooting/troubleshooting.md

- [ ] Find all occurrences of `cargo run -p bitnet-cli -- compat-`
- [ ] Update 5 main examples with `--no-default-features --features cpu,full-cli`
- [ ] Update inline documentation references (backtick blocks)
- [ ] Verify RUST_LOG=debug example updated
- [ ] Test: `grep -c "no-default-features" docs/troubleshooting/troubleshooting.md` should increase by 6

### Update docs/development/build-commands.md

- [ ] Find all occurrences of `cargo run -p bitnet-cli -- inspect` and `cargo run -p bitnet-st2gguf`
- [ ] Add `--no-default-features --features cpu,full-cli` after `-p bitnet-cli`
- [ ] Add `--no-default-features --features cpu` after `-p bitnet-st2gguf`
- [ ] Test: `grep -c "no-default-features" docs/development/build-commands.md` should increase

### Update docs/development/validation-ci.md

- [ ] Find `cargo run -p bitnet-st2gguf` example
- [ ] Find `cargo run -p bitnet-cli -- inspect` examples
- [ ] Update with appropriate feature flags
- [ ] Test: `grep -c "no-default-features" docs/development/validation-ci.md` should increase

## Verification Script

```bash
#!/bin/bash
# Script to verify all code examples have proper feature flags

echo "=== Checking for cargo run examples missing feature flags ==="
echo ""

FILES=(
  "docs/troubleshooting/troubleshooting.md"
  "docs/development/build-commands.md"
  "docs/development/validation-ci.md"
)

for file in "${FILES[@]}"; do
  echo "File: $file"
  echo "Without features:"
  grep -n "cargo run -p bitnet-" "$file" | grep -v "no-default-features" | head -10
  echo ""
done

echo "=== Summary ==="
echo "Total lines needing fixes:"
for file in "${FILES[@]}"; do
  count=$(grep "cargo run -p bitnet-" "$file" | grep -v "no-default-features" | wc -l)
  echo "$file: $count lines"
done
```

## Manual Testing After Updates

```bash
# After fixing docs/troubleshooting/troubleshooting.md:
cd /home/steven/code/Rust/BitNet-rs
grep "cargo run -p bitnet-cli.*compat-check" docs/troubleshooting/troubleshooting.md | \
  head -1 | \
  sed 's/^[^c]*cargo/cargo/' > /tmp/test_cmd.sh
# Manually review the command extracted

# Count examples fixed:
echo "Examples with no-default-features in troubleshooting.md:"
grep -c "no-default-features" docs/troubleshooting/troubleshooting.md

echo "Examples with no-default-features in build-commands.md:"
grep -c "no-default-features" docs/development/build-commands.md

echo "Examples with no-default-features in validation-ci.md:"
grep -c "no-default-features" docs/development/validation-ci.md
```

## Commit Message Template

```
docs: add missing feature flags to code examples

Add --no-default-features --features cpu,full-cli to cargo run 
examples in troubleshooting and build-commands documentation to 
ensure consistency with primary documentation and prevent user 
confusion from missing default features.

Fixes:
- 5 examples in docs/troubleshooting/troubleshooting.md
- 3 examples in docs/development/build-commands.md  
- 2 examples in docs/development/validation-ci.md

Verification:
- All examples now include explicit feature flags
- grep "no-default-features" docs/{troubleshooting,development}/*.md works
- Affected files tested for markdown syntax validity
```

---

**Total Fixes Required**: ~10-12 lines across 3 files
**Estimated Time**: 10-15 minutes
**Risk Level**: None (documentation only)
**Testing**: Grep-based verification + manual link check
