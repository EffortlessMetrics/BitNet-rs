# PR2 Analysis Summary: EnvGuard Migration & Serial Test Implementation

## Key Findings

### Current State
- **4 fragmented EnvGuard implementations** across codebase (primary, bitnet-common, bitnet-kernels, bitnet-inference)
- **40+ tests** mutate environment variables, but **~30% lack #[serial] annotations**
- **2 confirmed flaky tests** with ~50% failure rate in workspace runs:
  - `issue_260_strict_mode_tests::test_cross_crate_strict_mode_consistency`
  - `issue_260_strict_mode_tests::test_strict_mode_error_reporting`

### Root Cause Analysis
Cross-crate lazy static races:
1. Test A sets `BITNET_STRICT_MODE=1`
2. Test B lazily initializes `StrictModeConfig` from Test A's environment
3. Test A unsets the variable
4. Test B reads cached (wrong) value

#[serial] mitigates but doesn't fully prevent - multiple cargo test processes can still race.

### EnvGuard Implementation Comparison

| Feature | Primary (/tests) | bitnet-common | bitnet-kernels | Status |
|---------|---|---|---|---|
| `set()` | ✅ | ✅ | ✅ | All OK |
| `remove()` | ✅ | ✅ | ❌ | Missing in kernels! |
| `key()` accessor | ✅ | ❌ | ❌ | Only in primary |
| Unit tests | ✅ 7 tests | ❌ | ❌ | Only primary tested |
| Documentation | ✅ Extensive | ⚠️ Basic | ⚠️ Minimal | Varies |

### Dependency Status
- `serial_test = "3.2.0"` ✅ Already in workspace.dependencies
- `temp-env = "0.3.6"` ✅ Already in workspace.dependencies
- No version conflicts or missing dependencies

## Migration Strategy

### Phase 1: Consolidation (1 day)
- Verify primary EnvGuard completeness
- Export from tests crate public API
- Create documentation

### Phase 2: Annotations (1.5 days)
- Add `#[serial(bitnet_env)]` to 40+ tests
- Replace manual env ops with EnvGuard
- Focus on 3 high-priority files:
  - `/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs` (10 tests)
  - `/crates/bitnet-kernels/tests/strict_gpu_mode.rs` (8+ tests)
  - `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` (5 tests)

### Phase 3: Validation (1 day)
- Run full suite with `--test-threads=1`
- Verify flaky tests pass 10x consecutively
- Un-ignore flaky tests

### Phase 4: Documentation (0.5 day)
- Update CLAUDE.md
- Create test guide in /docs/development
- Update issue #441

## Environment Variables Tracked

### Critical (20+ tests each)
- `BITNET_STRICT_MODE` - Enforcement gate
- `BITNET_GPU_FAKE` - Device override
- `BITNET_DETERMINISTIC` - Inference randomness

### High (5-10 tests each)
- `BITNET_STRICT_FAIL_ON_MOCK`
- `BITNET_STRICT_REQUIRE_QUANTIZATION`
- `BITNET_STRICT_VALIDATE_PERFORMANCE`
- `BITNET_GGUF` - Model path override
- `BITNET_TOKENIZER` - Tokenizer path
- `CI` - CI environment flag

### Medium (2-5 tests each)
- `RUST_LOG` - Logging level
- `RAYON_NUM_THREADS` - Parallelism control

## Files Requiring Changes

### High Priority (Missing #[serial])
1. `/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs` - 10 tests
   - Uses manual `with_strict_mode()` helper without guard
   - All async tests affected
   - Action: Replace helper with EnvGuard, add #[serial(bitnet_env)]

2. `/crates/bitnet-kernels/tests/strict_gpu_mode.rs` - 8+ tests
   - GPU device override tests
   - Action: Add #[serial(bitnet_env)] throughout

3. `/crates/bitnet-inference/tests/ac7_deterministic_inference.rs` - Unknown count
   - Determinism control via env vars
   - Action: Add #[serial(bitnet_env)]

4. `/crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` - Unknown count
   - BITNET_DETERMINISTIC usage
   - Action: Add #[serial(bitnet_env)]

### Medium Priority (Already has some #[serial], needs improvement)
1. `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` - 20 tests, 2 flaky
   - Uses simplified EnvGuard from bitnet-common
   - Thread safety test uses custom lock instead of #[serial]
   - Action: Switch to primary EnvGuard, add #[serial(bitnet_env)] where missing

2. `/crates/bitnet-models/tests/gguf_weight_loading_*.rs` - 6 files
   - Model path configuration
   - Action: Add #[serial(bitnet_env)]

### Low Priority (Already correct)
- `/crates/bitnet-common/src/config/tests.rs` - Already has #[serial]
- `/crates/bitnet-common/src/warn_once.rs` - Already has #[serial]

## Success Criteria

- [ ] 10+ tests in strict_mode_runtime_guards pass with #[serial]
- [ ] 8+ tests in strict_gpu_mode pass with #[serial]
- [ ] Flaky tests pass 10x consecutively
- [ ] Zero remaining unsafe env ops without EnvGuard (except in support)
- [ ] All env-mutating tests have #[serial(bitnet_env)]
- [ ] `cargo test --workspace -- --test-threads=1` passes completely
- [ ] `cargo nextest run --workspace --profile ci` passes without flakiness

## Effort Estimate

| Phase | Tasks | Hours | Days | Risk |
|-------|-------|-------|------|------|
| Phase 1 | Setup | 4-6 | 1 | Low |
| Phase 2 | Annotations | 12-16 | 1.5 | Medium |
| Phase 3 | Validation | 6-10 | 1 | Medium |
| Phase 4 | Documentation | 2-4 | 0.5 | Low |
| **Total** | **4 Phases** | **24-36** | **3-4** | **Medium** |

## Implementation Checklist

### Before Starting
- [ ] Read full PR2 migration plan: `/ci/exploration/PR2_envguard_migration_plan.md`
- [ ] Verify primary EnvGuard: `cargo test -p bitnet-tests env_guard`
- [ ] Baseline test run: `cargo nextest run --workspace --profile ci`

### Phase 1
- [ ] Document EnvGuard API
- [ ] Export from tests crate

### Phase 2
- [ ] Update strict_mode_runtime_guards.rs (10 tests)
- [ ] Update strict_gpu_mode.rs (8+ tests)
- [ ] Update issue_260_strict_mode_tests.rs (5 tests)
- [ ] Update deterministic tests
- [ ] Update model loading tests (6 files)

### Phase 3
- [ ] Single-thread run: `cargo test --workspace -- --test-threads=1`
- [ ] Flaky test 10x validation
- [ ] Un-ignore flaky tests
- [ ] Nextest run: `cargo nextest run --workspace --profile ci`

### Phase 4
- [ ] Update CLAUDE.md
- [ ] Create /docs/development/test-env-guide.md
- [ ] Update issue #441
- [ ] Final full test run

## Related Issues

- **#441**: Environment variable pollution (ROOT ISSUE)
- **#260**: Mock elimination (affected by flakiness)
- **#439**: Feature gate consistency (merged, validation ongoing)
- **#469**: Tokenizer parity (affected by flakiness)

## Quick Reference

**Test EnvGuard**:
```bash
cargo test -p bitnet-tests env_guard
```

**Single-threaded full suite**:
```bash
cargo test --workspace -- --test-threads=1 --include-ignored
```

**Nextest (recommended)**:
```bash
cargo nextest run --workspace --profile ci
```

**Check for remaining unsafe ops**:
```bash
grep -r "unsafe.*env::set_var\|unsafe.*env::remove_var" \
  crates/*/tests --include="*.rs"
```

---

**Document**: `/ci/exploration/PR2_envguard_migration_plan.md` (1016 lines, 34KB)  
**Status**: ✅ Analysis Complete, Ready for PR2 Implementation  
**Next Step**: Execute Phase 1 setup
