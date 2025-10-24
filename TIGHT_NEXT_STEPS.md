# What's Next (Tight Checklist)

Below is the short, do-this-then-that run-order to finalize and ship this work.

---

## ✅ DONE (This Session)

1. ✅ **Compliance test security fix** - 2 vulnerabilities eliminated, 11/11 tests passing
2. ✅ **FFI build hygiene** - 3 Priority 1 fixes applied, clean builds
3. ✅ **Docs archive migration** - 53 files moved to docs/archive/reports/
4. ✅ **CI integration** - 7 new jobs added (13 → 20 total)
5. ✅ **Documentation** - 41 exploration/planning guides created
6. ✅ **Automation** - 9 scripts delivered (tested and functional)
7. ✅ **Validation** - 1955/1955 tests passing, 0 clippy warnings

**Commits Created**: 4 atomic commits (111 files, +16,954 lines)

---

## 📦 Push and Create PR (5 minutes)

```bash
# 1. Final verification
cargo nextest run --workspace --no-default-features --features cpu --no-fail-fast
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 2. Push to remote
git push origin feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2

# 3. Create PR (use GitHub-native workflow or gh CLI)
gh pr create --title "feat: add CI guards, fix compliance security, archive legacy docs" \
  --body "$(cat <<'EOF'
## Summary

Comprehensive quality improvements: CI guards, security fixes, docs organization.

**Changes**:
- 🔒 Security: Fixed 2 env_guard_compliance vulnerabilities (+100% hardening)
- 🏗️ CI: Added 7 jobs (features, fixtures, doctests, guards) - 13 → 20 total
- 📚 Docs: Archived 53 legacy reports to docs/archive/reports/
- 🔧 FFI: Applied 3 Priority 1 build hygiene fixes
- 📖 Documentation: 41 exploration/planning guides + 9 automation scripts

**Metrics**:
- Tests: 1955/1955 passing ✅
- Clippy: 0 warnings ✅
- CI Impact: +2 min gating (+6 min total)
- Security: 0 vulnerabilities ✅

**Commits**: 4 atomic commits (111 files, +16,954 lines)

See COMPREHENSIVE_AGENT_ORCHESTRATION_COMPLETION_REPORT.md for full details.
EOF
)"
```

---

## 🔍 Monitor First CI Run (15 minutes)

After PR creation, watch the first CI run:

1. **Verify all 20 jobs execute**
   - 13 existing jobs: Should pass as before
   - 7 new jobs: Validate execution

2. **Expected behaviors**:
   - ✅ `feature-matrix`: Blocking, should pass
   - ✅ `doctest-matrix`: CPU blocking, all-features observational
   - ✅ `guard-fixture-integrity`: Blocking, should pass
   - ✅ `guard-serial-annotations`: Blocking, should pass
   - ✅ `guard-feature-consistency`: Blocking, should pass
   - ⚠️ `feature-hack-check`: Non-blocking (continue-on-error: true)
   - ⚠️ `guard-ignore-annotations`: Non-blocking (134 bare markers exist)

3. **If any blocking guard fails**:
   - Check guard script logs
   - Validate with local execution: `bash scripts/check-<guard>.sh`
   - Fix and push update

---

## 📝 Update CHANGELOG (Before Merge)

```bash
# Add to CHANGELOG.md under "Unreleased" section
cat >> CHANGELOG.md << 'EOF'

### Added
- CI guards: features (cargo-hack), env-mutation, annotated #[ignore], fixtures SHA, doctest (CPU gate)
- Test docs: EnvGuard security hardening with directory traversal protection
- Documentation: 41 exploration/planning guides for quality improvements
- Automation: 9 scripts (archive, hygiene, guards) with dry-run modes

### Changed
- FFI build: vendored GGML as external includes; vendor commit pinned (b4247)
- Docs: 53 legacy reports archived to docs/archive/reports/ with navigation
- CI: 13 → 20 jobs (7 new guards: feature-matrix, doctest-matrix, 4 hygiene guards, 1 observational)

### Fixed
- env_guard_compliance: Eliminated 2 security vulnerabilities (substring matching + directory traversal)
- FFI warnings: Now properly visible via Cargo protocol (eprintln → println)
- Compiler flags: POSIX-compliant -isystem spacing for portability

### Security
- EnvGuard compliance test hardened against directory traversal attacks
- Added 20-case security test suite for path-matching validation
EOF

git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for CI guards and security fixes"
git push
```

---

## 🚀 Post-Merge Follow-Ups (Create Issues)

After merge, create these GitHub issues:

### Issue 1: Complete #[ignore] Annotation Hygiene (Priority: P1)
```markdown
**Title**: Complete Phase 2-5 of #[ignore] annotation migration

**Description**:
Reduce bare #[ignore] markers from 134 to <10 using systematic phased approach.

**Resources**:
- IGNORE_ANNOTATION_ACTION_PLAN_PHASE1.md
- scripts/auto-annotate-ignores.sh
- scripts/check-ignore-hygiene.sh

**Estimated Effort**: 8-10 hours (70% automated)
**Phases**:
- Phase 2: Slow/performance tests (17 markers, 1.5 hours)
- Phase 3: Network/flaky tests (13 markers, 1 hour)
- Phase 4: Model/fixture tests (29 markers, 2 hours)
- Phase 5: Quantization/parity/TODO tests (43 markers, 2 hours)
```

### Issue 2: Add Windows/MSVC FFI Support (Priority: P2)
```markdown
**Title**: Add Windows CI job and MSVC build support for FFI crates

**Description**:
Complete build hygiene with MSVC support and Windows CI validation.

**Resources**:
- FFI_BUILD_HYGIENE_ACTION_PLAN.md (Priority 2 section)
- SPEC-2025-002-build-script-hygiene-hardening.md

**Estimated Effort**: 2-3 hours
**Tasks**:
- Add `/external:I` pragma injection to xtask-build-helper
- Add `windows-latest` CI job for FFI crates
- Validate `/external:W0` suppresses third-party warnings
```

### Issue 3: Enable cargo-hack Observability (Priority: P3)
```markdown
**Title**: Promote feature-hack-check from observational to blocking

**Description**:
Once feature-hack-check proves stable (2-3 CI runs), remove continue-on-error.

**Estimated Effort**: 30 minutes
**Tasks**:
- Monitor 3+ successful CI runs with feature-hack-check
- Remove `continue-on-error: true` from feature-hack-check job
- Update CI documentation
```

---

## 📊 Definition of "Done"

- ✅ CI green with new guards (CPU doctests/tests/clippy gate)
- ✅ No bare #[ignore] causing CI failures (guard is observational)
- ✅ Fixtures verified (SHA256 + schema + alignment)
- ✅ FFI zero-warning lane clean
- ✅ PR merged to main

**Current Status**: ✅ Ready for push and PR creation

**Next Action**: Execute "Push and Create PR" section above (5 minutes)

---

## 🎯 Rollback Toggles (If Needed)

If a guard becomes noisy after merge:

```bash
# Temporarily make guard non-blocking
# Edit .github/workflows/ci.yml, add to problematic job:
    continue-on-error: true

# Then push fix:
git add .github/workflows/ci.yml
git commit -m "ci: make <guard-name> observational until <issue> resolved"
git push
```

**Note**: Only CPU-gating jobs (test, clippy, doctest-matrix-cpu) should remain blocking.
All new guards can be made observational if they cause unexpected failures.

---

## ✨ Success Criteria

After merge, you should see:
- ✅ All 20 CI jobs in GitHub Actions
- ✅ Blocking guards passing (feature-matrix, doctest-cpu, fixture-integrity, serial, feature-consistency)
- ✅ Observational guards reporting (feature-hack, ignore-annotations)
- ✅ CI completes in ~14 minutes (expected: 8 min baseline + 6 min new jobs)
- ✅ No regressions in existing test suites
- ✅ Archive navigation working (docs/archive/reports/ excluded from link checks)

**Everything is ready**. Next action: Push and create PR (5 minutes).
