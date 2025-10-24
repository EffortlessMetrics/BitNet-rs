# Phase 1 Execution Summary - Quick Reference

**Date**: October 23, 2025  
**Status**: Ready for Execution  
**Estimated Time**: 2 hours  
**Automation Rate**: 95%

---

## Quick Start (Copy-Paste Ready)

### 1. Pre-Flight Check (5 minutes)

```bash
cd /home/steven/code/Rust/BitNet-rs

# Check current state
MODE=full bash scripts/check-ignore-hygiene.sh | head -20

# Expected: 236 total, 134 bare (56.8%)
```

### 2. Dry-Run Validation (30 minutes)

```bash
# Create target file list
cat > phase1-files.txt <<'FILELIST'
crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs
crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs
crates/bitnet-inference/tests/neural_network_test_scaffolding.rs
crates/bitnet-inference/tests/ac3_autoregressive_generation.rs
crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs
crates/bitnet-inference/tests/issue_254_layer_norm_invariants.rs
crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs
crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs
crates/bitnet-inference/tests/issue_254_ac1_quantized_linear_no_fp32_staging.rs
crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs
crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs
FILELIST

# Dry-run all files
while read -r file; do
  TARGET_FILE="$file" DRY_RUN=true bash scripts/auto-annotate-ignores.sh
done < phase1-files.txt > phase1-dryrun-output.txt

# Review output
less phase1-dryrun-output.txt
```

**Validation**: Check that confidence scores are ≥70% and suggested annotations match context.

### 3. Execute Migration (45 minutes)

```bash
# Create branch
git checkout -b ignore-hygiene-phase1

# Apply annotations
while read -r file; do
  TARGET_FILE="$file" DRY_RUN=false bash scripts/auto-annotate-ignores.sh
done < phase1-files.txt > phase1-apply-output.txt

# Format code
cargo fmt --all

# Verify changes
git diff | grep -A2 -B2 '#\[ignore' | less
```

### 4. Verification (30 minutes)

```bash
# Check new bare count
MODE=full bash scripts/check-ignore-hygiene.sh | grep "Bare (no reason)"

# Expected: ~91 bare (38%)

# Run tests
cargo test --workspace --no-default-features --features cpu --lib

# Expected: 152+ tests passing

# Check for placeholder text
rg '#\[ignore = "[^"]*\.\.\.[^"]*"\]' --type rust crates/ tests/ xtask/ || \
  echo "✅ No placeholder text remains"
```

### 5. Commit (10 minutes)

```bash
# Review final diff
git status
git diff --stat

# Commit
git add -A
git commit -m "feat(tests): Phase 1 #[ignore] annotation hygiene - issue-blocked tests

- Auto-annotate 46 issue-blocked tests with explicit reasons
- Reduce bare #[ignore] from 134 (56.8%) to ~91 (38.6%)
- High-confidence (≥70%) auto-annotation using scripts/auto-annotate-ignores.sh
- Categories: Issue #254, #159, #248, #260, slow tests

Phase 1 of 5-week phased rollout (SPEC-2025-006)"
```

---

## Expected Outcomes

### Before Phase 1

```
Total #[ignore] annotations: 236
Bare (no reason):            134 (56%)
```

### After Phase 1

```
Total #[ignore] annotations: 236
Bare (no reason):            ~91 (38%)
Improvement:                 43 annotations added (18% reduction)
```

---

## Rollback (If Needed)

```bash
git reset --hard HEAD
git checkout main
git branch -D ignore-hygiene-phase1
```

---

## Success Checklist

- [ ] **Pre-flight**: Current bare count = 134 (56.8%)
- [ ] **Dry-run**: All files processed with ≥70% confidence
- [ ] **Manual review**: Suggestions match file context
- [ ] **Execution**: ~46 annotations applied successfully
- [ ] **Formatting**: `cargo fmt --all -- --check` passes
- [ ] **Tests**: No regressions (152+ tests passing)
- [ ] **Validation**: No placeholder text (...) remains
- [ ] **Post-flight**: Bare count reduced to ~91 (38.6%)
- [ ] **Git**: Clean diff, only #[ignore] lines modified
- [ ] **Commit**: Descriptive commit message referencing Phase 1

---

## Troubleshooting

### "Script not found"

```bash
chmod +x scripts/check-ignore-hygiene.sh scripts/auto-annotate-ignores.sh
```

### "Low confidence" warnings

**Solution**: Skip these files in Phase 1, defer to later phases.

### Test suite fails

**Solution**: Check for syntax errors in annotations:

```bash
rg '#\[ignore = "[^"]*"[^]]' --type rust crates/ tests/ xtask/
```

---

## Next Steps (After Phase 1 Merge)

1. **Phase 2 (Week 2)**: Performance/slow tests (17 annotations)
2. **Phase 3 (Week 3)**: Network/external deps (13 annotations)
3. **Phase 4 (Week 4)**: Model/fixture deps (29 annotations)
4. **Phase 5 (Week 5)**: Quantization/parity/TODO (43 annotations)

**Final Target**: ≤5% bare (≤12 remaining) by Week 5.

---

**See Also**: `IGNORE_ANNOTATION_ACTION_PLAN_PHASE1.md` (full detailed plan)
