# BitNet.rs Performance Issues - Executive Summary

**Analysis Date**: November 11, 2025
**Post**: PR #475 (GPU/CPU Feature Gate Unification)

---

## Quick Action Items

### 🚨 CRITICAL - Execute Immediately

```bash
# Escalate correctness blocker #393 (GGUF quantization mapping)
gh issue edit 393 --add-label "bug,priority/high,area/performance,mvp:blocker"

# Escalate infrastructure blocker #319 (KV cache memory pool)
gh issue edit 319 --add-label "priority/high,area/performance,mvp:blocker"

# Fix dispatch bug #401 (TL2 routing to TL1)
gh issue edit 401 --add-label "priority/high,area/performance,mvp:polish"

# Add blocking dependencies
gh issue comment 393 --body "**Blocking Issues**: This correctness bug blocks #346 (TL1) and #401 (TL2) implementations."
gh issue comment 346 --body "**Blocked By**: Issue #393 (GGUF quantization mapping) must be resolved first."
gh issue comment 401 --body "**Blocked By**: Issue #393 (GGUF quantization mapping) must be resolved first."
```

---

## MVP Blocker Triage

### P0 - Must Fix Before/At MVP Launch (v0.1.0)

| Issue | Title | Status | Action Required |
|-------|-------|--------|----------------|
| **#417** | QK256 CPU I2S Dequantization | ✅ Correctly labeled | Continue work - foundation in place (~1.2× uplift, targeting ≥3×) |
| **#393** | GGUF Quantization Mapping | ⚠️ **ESCALATE** | Add mvp:blocker label - correctness issue with silent corruption risk |
| **#319** | KV Cache Memory Pool | ⚠️ **ESCALATE** | Add mvp:blocker label - blocks multi-session inference |

**Total MVP Blockers**: 3 issues
**Estimated Total Effort**:
- #417: 2.5-4 days (in progress)
- #393: TBD (detailed implementation plan exists)
- #319: 2-3 weeks (detailed implementation plan exists)

---

## Post-MVP High Priority (v0.2.0)

| Issue | Title | Blocked By | Effort |
|-------|-------|------------|--------|
| **#401** | TL2 Quantization (dispatch bug) | #393 | 2.5-4 days |
| **#346** | TL1 Quantization | #393 | 2.5-4 days |

**Note**: Both are blocked by #393 correctness fix. Can proceed in parallel once #393 is resolved.

---

## Future Optimizations (v0.3.0+)

| Issue | Title | Action | Effort |
|-------|-------|--------|--------|
| **#379** | Top-K Sampling | Consolidate with #380 or prioritize one | 4-5 weeks |
| **#380** | Top-P Sampling | Consolidate with #379 or prioritize one | 2-3 weeks |

**Recommendation**: Consider unified "Sampling Optimization" effort to avoid duplication.

---

## Dependency Chain

```
MVP v0.1.0:
  ┌─ #417 (QK256) [INDEPENDENT] ← Foundation in place
  ├─ #393 (GGUF mapping) [INDEPENDENT] ← BLOCKS v0.2.0
  └─ #319 (KV cache) [INDEPENDENT] ← Multi-session blocker

Post-MVP v0.2.0:
  ┌─ #346 (TL1) ← BLOCKED BY #393
  └─ #401 (TL2) ← BLOCKED BY #393

Future v0.3.0:
  └─ #379 + #380 (Sampling) ← Consider consolidation
```

---

## Key Findings

### ✅ What's Working
- QK256 AVX2 foundation is in place with ~1.2× initial uplift
- Receipt verification system operational (Schema v1.0.0, 25/25 tests)
- Strict mode runtime guards in place (12/12 tests)
- Cross-validation framework with dual-backend support active
- Feature gates unified (PR #475 resolved #439)

### ⚠️ Critical Gaps Identified
1. **#393 GGUF Mapping**: Correctness bug causing silent inference corruption
   - Q4/Q5/Q8 wrongly mapped to BitNet I2S/TL types
   - Missing I2S/IQ2_S handling
   - Blocks TL1/TL2 implementations

2. **#319 KV Cache**: Production-blocking memory management stubs
   - Pool allocates metadata only, not real memory
   - Cache entries bypass pool, create separate allocations
   - Memory leaks and fragmentation in multi-session scenarios

3. **#401 TL2 Dispatch**: Copy-paste bug routing TL2 to TL1
   - Critical for users expecting TL2 quantization
   - Blocked by #393 for correct implementation

### 🎯 Performance Impact Summary

| Issue | Current Impact | Target Impact | User Facing |
|-------|---------------|---------------|-------------|
| #417 | ~0.1 tok/s (QK256) | ≥3 tok/s (10-20× improvement) | ✅ Critical |
| #393 | Silent corruption | Correct inference | ✅ Critical |
| #319 | Memory leaks | Stable multi-session | ✅ Critical |
| #401 | Wrong quantization | Correct TL2 | ⚠️ Medium |
| #346 | Inefficient TL1 | Proper TL1 | ⚠️ Medium |
| #379 | 150-200 tok/s | >500 tok/s | ✅ High |
| #380 | Slow sampling | 40-60% faster | ✅ High |

---

## Milestone Assignments

### v0.1.0 (MVP Launch - CRITICAL)
- #417 ✅ (already assigned)
- #393 ⚠️ (needs escalation)
- #319 ⚠️ (needs escalation)

### v0.2.0 (Post-Launch Polish)
- #401 (TL2 dispatch fix)
- #346 (TL1 optimization)

### v0.3.0 (Optimization Wave)
- #379 + #380 (Sampling - consider consolidation)

### Documentation (Continuous)
- #156 (README performance claims)
- #459 (Receipt-driven examples)
- #221 (Meta-tracking)

---

## Issue Label Summary

| Issue | Current Labels | Recommended Add | Remove |
|-------|---------------|----------------|--------|
| #393 | None | bug, priority/high, area/performance, mvp:blocker | - |
| #319 | None | priority/high, area/performance, mvp:blocker | - |
| #401 | None | priority/high, area/performance, mvp:polish | - |
| #346 | enhancement, priority/high, area/performance | mvp:polish | - |
| #379 | enhancement, priority/high, area/performance | mvp:polish | - |
| #380 | enhancement, priority/high, area/performance | mvp:polish | - |
| #156 | documentation, enhancement, priority/high, area/performance | mvp:polish | - |
| #459 | None | documentation, mvp:polish | - |
| #417 | mvp:blocker, priority/high, area/performance | ✅ Correct | - |

---

## Overlap and Duplication Analysis

### Confirmed Overlaps
- **#379 (Top-K) + #380 (Top-P)**: Significant overlap in sampling optimization
  - **Action**: Propose consolidation or sequential prioritization

- **#349 (I2S Fast Path) + #417 (QK256 Dequant)**: Possible overlap
  - **Action**: Review to determine if duplicate or complementary

### Resolved Issues
- **#439 (Feature Gate Consistency)**: ✅ Resolved by PR #475

### Potentially Resolved
- **#213 (Test Performance)**: May be resolved by EnvGuard + nextest infrastructure
  - **Action**: Review for potential closure

---

## Critical Path for MVP

```
Week 1-2:
  ├─ Fix #393 (GGUF mapping) ← UNBLOCKS TL1/TL2
  ├─ Continue #417 (QK256 optimization)
  └─ Start #319 (KV cache pool)

Week 3-4:
  ├─ Complete #417 (QK256 to ≥3× target)
  ├─ Continue #319 (2-3 week timeline)
  └─ Validate #393 fixes

Week 5-6 (Post-MVP):
  ├─ #401 (TL2 dispatch fix)
  ├─ #346 (TL1 optimization)
  └─ Finalize #319

Week 7+ (Optimization Wave):
  └─ #379/#380 (Sampling consolidation)
```

---

## Risk Assessment

### High Risk (Immediate Attention)
1. **#393**: Correctness bug with silent corruption potential
   - **Mitigation**: Escalate immediately, assign to experienced developer
   - **Timeline**: Critical for MVP integrity

2. **#319**: Memory management stubs causing leaks
   - **Mitigation**: 2-3 week dedicated implementation effort
   - **Timeline**: Can be addressed immediately post-MVP if needed

### Medium Risk (Monitor)
1. **#401**: TL2 dispatch bug affects TL2 users
   - **Mitigation**: Clear path once #393 resolved
   - **Timeline**: v0.2.0 acceptable

### Low Risk (Managed)
1. **#379/#380**: Performance issues, not correctness
   - **Mitigation**: Defer to v0.3.0, consider consolidation
   - **Timeline**: Can be deferred without major impact

---

## Recommendations for Orchestrator

### Immediate Actions (Today)
1. Execute critical GitHub CLI commands (escalate #393, #319, #401)
2. Add blocking relationships (#393 → #346/#401)
3. Update milestone assignments

### Short-Term (This Week)
1. Review #349 vs #417 for consolidation
2. Review #213 for potential closure
3. Propose consolidation plan for #379/#380
4. Assign developers to MVP blockers

### Medium-Term (2-4 Weeks)
1. Track progress on MVP blockers (#393, #417, #319)
2. Prepare for v0.2.0 TL1/TL2 implementations
3. Plan sampling optimization approach

### Long-Term (Post-MVP)
1. Update performance documentation (#156, #459)
2. Execute sampling optimization wave
3. Monitor for new performance issues

---

## Key Metrics to Track

### MVP Readiness
- [ ] #417 QK256 performance: Current ~1.2×, Target ≥3×
- [ ] #393 GGUF mapping: Tests passing, no silent corruption
- [ ] #319 KV cache: Zero leaks in 24-hour soak test

### Post-MVP Quality
- [ ] #401 TL2: Correct dispatch verified
- [ ] #346 TL1: ≤1e-4 round-trip error
- [ ] Sampling: >30% throughput improvement

### Documentation
- [ ] README claims match receipt data
- [ ] Performance guide updated
- [ ] Example outputs with real benchmarks

---

*For detailed analysis and full implementation context, see: `/home/steven/code/Rust/BitNet-rs/docs/analysis/performance-issues-analysis-2025-11-11.md`*
