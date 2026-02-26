# BitNet-rs Performance Issues Analysis - November 11, 2025

## Executive Summary

This document analyzes all open performance-related issues in the BitNet-rs repository following the completion of PR #475 (GPU/CPU feature gate unification) and the recent guardrail wave. The analysis focuses on:

1. Current status and relationship to recent work
2. Whether issues are resolved, duplicated, or still actionable
3. Recommended labels and milestone assignments
4. Priority rankings (P0/P1/P2)
5. Dependencies on other issues
6. Concrete GitHub CLI commands for issue management

## Key Context from Recent Work

- **PR #475** unified GPU/CPU feature gates, resolving Issue #439
- **Receipt verification system** (Schema v1.0.0) with 8 validation gates is operational (25/25 tests passing)
- **Strict mode runtime guards** are in place (12/12 tests passing)
- **QK256 AVX2 foundation** is established (~1.2× uplift, targeting ≥3×)
- **Cross-validation framework** with dual-backend support (BitNet.cpp + llama.cpp) is active
- **EnvGuard environment isolation** enables robust parallel test execution

---

## Critical Performance Issues (P0 - MVP Blockers)

### Issue #417: QK256 CPU I2S Dequantization Optimization

**Status**: OPEN - MVP Blocker
**Current State**: Already labeled as `mvp:blocker`, `priority/high`, `area/performance`

**Analysis**:
- **Impact**: Critical for QK256 inference performance (~0.1 tok/s → target ≥3 tok/s)
- **Recent Work**: QK256 AVX2 dequantization foundation is in place with ~1.2× initial uplift
- **Path Forward**: Needs nibble-LUT + FMA tiling + prefetch optimizations to reach ≥3× target
- **Dependencies**: None - can proceed independently
- **Relationship to PR #475**: No direct conflict; feature gates are unified

**Current Status Details** (from comments):
- Scalar LUT path with byte-level lookup table is sketched
- AVX2/AVX-512/NEON SIMD lane designs are documented
- Runtime kernel selection (feature detection) is planned
- Accuracy targets: ≥99.8% correlation vs FP32, ≤1e-5 max abs err
- Performance target: 10-20 tok/s for 2B models (vs current ~0.1 tok/s)

**Recommendation**: **KEEP AS mvp:blocker**
- This is the most critical performance blocker for production inference
- Foundation work is in place, but optimization target not yet reached
- Assign to: **Milestone: MVP v0.1.0**

**GitHub CLI Commands**:
```bash
# No changes needed - already correctly labeled
# Verify current state
gh issue view 417 --json labels,milestone,assignees
```

---

### Issue #417: Related Sub-Issues for SIMD Optimization

**Potential Sub-Issues to Track**:
1. Scalar branch-free LUT path (0.5 day)
2. AVX2 lane implementation (0.5-1 day)
3. NEON lane implementation (0.5 day)
4. AVX-512 lane implementation (0.5-1 day)
5. Kernel benchmarking and ledger integration (0.25-0.5 day)

**Total Estimated Effort**: ~2.5-4 days (from issue analysis)

---

## High-Priority Performance Issues (P1 - Post-MVP High Impact)

### Issue #346: TL1 CPU Quantization Production Implementation

**Status**: OPEN
**Current State**: Labeled as `enhancement`, `priority/high`, `area/performance`

**Analysis**:
- **Impact**: High - affects TL1 quantization correctness and performance
- **Current Implementation**: Simplified linear scaling without proper lookup table
- **Problem**: Stores 4-bit values as full bytes (inefficient), no actual table lookup
- **Target**: 4-bit packed nibbles (2 per byte) with real LUT quantization
- **Dependencies**: None - independent optimization
- **Relationship to PR #475**: No conflicts

**Current Status Details**:
- Issue has 3 detailed implementation comments with code sketches
- Scalar LUT + 4-bit pack/unpack approach designed
- AVX2/NEON SIMD lanes planned
- Accuracy target: ≤1e-4 round-trip error, TL1 ≥99.6% vs FP32
- Estimated effort: ~2.5-4 days (similar to #417 pattern)

**Recommendation**: **KEEP as priority/high, assign to MVP v0.2.0**
- Not a blocker for initial MVP (v0.1.0) but important for quantization quality
- Can proceed after #417 stabilizes
- Should add: `mvp:polish` label to indicate post-MVP-launch enhancement

**GitHub CLI Commands**:
```bash
# Add mvp:polish label for post-MVP tracking
gh issue edit 346 --add-label "mvp:polish"

# Assign to v0.2.0 milestone (if exists, otherwise skip)
# gh issue edit 346 --milestone "v0.2.0"

# Verify changes
gh issue view 346 --json labels,milestone
```

---

### Issue #401: TL2 CPU Quantizer Production Implementation

**Status**: OPEN
**Current State**: No labels applied

**Analysis**:
- **Impact**: High - TL2 quantization simulation stubs blocking production deployment
- **Current Implementation**: Multiple simulation stubs:
  - AVX-512 fallback stub (immediate fallback to AVX2)
  - Inefficient block-specific lookup table creation
  - Non-optimized vectorized lookup (scalar access in SIMD code)
  - Architecture-specific no-op implementations
- **Performance Impact**: 50-70% loss on AVX-512, 80-90% on ARM platforms
- **Dependencies**: Should be coordinated with #346 (TL1) for consistent approach
- **Relationship to PR #475**: Feature gates unified, no conflicts

**Current Status Details** (from comments):
- Copy-paste bug: TL2 dispatch calls `quantize_tl1()` instead of `quantize_tl2()`
- Device-aware quantizer needs proper TL2 routing
- Missing real TL2 kernels (scalar + AVX2/AVX-512/NEON)
- SIMD-friendly LUT and scale management needed
- Estimated effort: ~2.5-4 days (similar pattern to #346/#417)

**Recommendation**: **CRITICAL - needs immediate labeling and prioritization**
- Add labels: `priority/high`, `area/performance`, `mvp:polish`
- This is higher priority than #346 due to dispatch bug
- Assign to: **Milestone: MVP v0.2.0** (post-launch fix)
- **Blocker severity**: Medium-High (affects TL2 users, but fewer than QK256)

**GitHub CLI Commands**:
```bash
# Add missing labels
gh issue edit 401 --add-label "priority/high,area/performance,mvp:polish"

# Assign to v0.2.0 milestone (if exists)
# gh issue edit 401 --milestone "v0.2.0"

# Add comment noting dispatch bug severity
gh issue comment 401 --body "**Priority Update**: This issue contains a critical dispatch bug where TL2 routes to TL1 quantization. Recommended for v0.2.0 milestone as high-priority post-MVP fix."

# Verify changes
gh issue view 401 --json labels,milestone
```

---

### Issue #379: Top-K Sampling Optimization

**Status**: OPEN
**Current State**: Labeled as `enhancement`, `priority/high`, `area/performance`

**Analysis**:
- **Impact**: High - affects all text generation workloads
- **Current Implementation**: Inefficient O(n log n) full sort for every token
- **Problem**: Memory allocation overhead, redundant NaN filtering, suboptimal data structures
- **Target**: O(n + k log k) heap-based selection with pre-allocated buffers
- **Performance Target**: 150-200 tok/s → >500 tok/s with top-k=50
- **Dependencies**: None - independent optimization
- **Relationship to PR #475**: No conflicts; applies to all inference modes

**Current Status Details**:
- One detailed implementation comment with drop-in code
- Heap-based top-k selection designed (O(n + k log k))
- Memory pool for sampling buffers planned
- SIMD-accelerated implementations sketched (AVX2/NEON)
- Device-aware optimization with adaptive algorithm selection
- Estimated effort: 4-5 weeks (1 senior engineer)

**Recommendation**: **KEEP as priority/high, consider for MVP v0.2.0**
- Not blocking MVP launch (current sampling works, just slow)
- High impact for production throughput
- Should add: `mvp:polish` label
- **Note**: Longer timeline (4-5 weeks) suggests this might be v0.3.0 candidate

**GitHub CLI Commands**:
```bash
# Add mvp:polish label
gh issue edit 379 --add-label "mvp:polish"

# Consider v0.3.0 milestone given 4-5 week timeline
# gh issue edit 379 --milestone "v0.3.0"

# Verify changes
gh issue view 379 --json labels,milestone
```

---

### Issue #380: Top-P (Nucleus) Sampling Optimization

**Status**: OPEN
**Current State**: Labeled as `enhancement`, `priority/high`, `area/performance`

**Analysis**:
- **Impact**: High - affects text generation quality and performance
- **Current Implementation**: Dual inconsistent implementations, repeated sorting, tensor conversion overhead
- **Problem**: Performance bottlenecks in every sampling step
- **Target**: 40-60% latency reduction, 30-50% memory reduction
- **Dependencies**: Should coordinate with #379 (Top-K) for unified sampling approach
- **Relationship to PR #475**: No conflicts

**Current Status Details**:
- No implementation comments yet (less mature than #379)
- Dual implementation inconsistency identified
- Performance bottlenecks documented
- Estimated effort: 2-3 weeks

**Recommendation**: **CONSOLIDATE with #379 or defer to v0.3.0**
- These two issues (#379 and #380) overlap significantly in scope
- Should either merge into single "Sampling Optimization" epic OR prioritize one first
- Both target similar timeframes and have similar impacts
- **Suggestion**: Comment on both issues proposing unified sampling optimization

**GitHub CLI Commands**:
```bash
# Add mvp:polish label for consistency
gh issue edit 380 --add-label "mvp:polish"

# Add cross-reference comment
gh issue comment 380 --body "**Note**: This issue has significant overlap with #379 (Top-K Sampling). Consider consolidating into a unified sampling optimization effort or prioritizing one approach first to avoid duplicated work."

# Add same comment to #379
gh issue comment 379 --body "**Note**: This issue has significant overlap with #380 (Top-P Sampling). Consider consolidating into a unified sampling optimization effort or prioritizing one approach first to avoid duplicated work."

# Verify changes
gh issue view 380 --json labels,milestone
```

---

### Issue #319: KV Cache Memory Pool Production Implementation

**Status**: OPEN
**Current State**: No labels applied

**Analysis**:
- **Impact**: CRITICAL - blocks production deployment for multi-session workloads
- **Current Implementation**: Production-blocking stubs:
  - Memory pool allocates metadata only, not real memory
  - Cache entries ignore pool and create separate Vec allocations
  - Deallocation uses hardcoded offset 0
- **Problem**: Memory fragmentation, leaks, unpredictable scaling
- **Target**: Zero-copy cache entries, proper memory pool with real buffers
- **Dependencies**: None - critical infrastructure issue
- **Relationship to PR #475**: No conflicts; affects all inference modes
- **Estimated Effort**: 2-3 weeks (from issue analysis)

**Current Status Details**:
- One detailed implementation comment with production-ready design
- Real contiguous arena with typed views designed
- Zero-copy KV entries with pool slices planned
- Proper memory layout and allocation/deallocation logic sketched
- Safety invariants documented

**Recommendation**: **ESCALATE to P0 - MVP Blocker for multi-session scenarios**
- This is more critical than it appears - affects production server viability
- Add labels: `priority/high`, `area/performance`, `mvp:blocker`
- Assign to: **Milestone: MVP v0.1.0** (should be fixed before or shortly after launch)
- **Blocker severity**: HIGH for multi-session inference, MEDIUM for single-session

**GitHub CLI Commands**:
```bash
# Add critical labels
gh issue edit 319 --add-label "priority/high,area/performance,mvp:blocker"

# Assign to MVP v0.1.0 milestone
# gh issue edit 319 --milestone "v0.1.0"

# Add escalation comment
gh issue comment 319 --body "**Priority Escalation**: This issue contains production-blocking memory management stubs that prevent proper multi-session inference. Recommended for MVP v0.1.0 milestone as critical infrastructure fix. Estimated effort: 2-3 weeks."

# Verify changes
gh issue view 319 --json labels,milestone
```

---

### Issue #393: GGUF Quantization Type Mapping Correctness

**Status**: OPEN
**Current State**: No labels applied

**Analysis**:
- **Impact**: HIGH - affects model loading correctness and compatibility
- **Current Implementation**: Fundamental flaws:
  - Incorrect GGUF tensor type mapping (Q4/Q5/Q8 wrongly mapped to BitNet I2S/TL types)
  - Missing I2S and IQ2_S handling
  - Table lookup parameters unused in TL1/TL2 layers
  - Dead code in DeviceAwareQuantizer (IQ2S, FP32 not implemented)
- **Problem**: Silent data corruption risk, incorrect quantization detection
- **Target**: Correct type mapping, proper I2S/IQ2S detection, functional TL1/TL2 lookups
- **Dependencies**: **Blocks #346 and #401** (TL1/TL2 implementations need correct mapping)
- **Relationship to PR #475**: No conflicts; critical correctness issue
- **Estimated Effort**: Not specified in issue

**Current Status Details**:
- Two very detailed implementation comments (including one massive comment with full implementation plan)
- Four-phase solution proposed:
  1. Fix GGUF tensor type mapping
  2. Implement missing quantization types
  3. Fix table lookup implementation
  4. Add FFI bridge for GGML compatibility
- Extensive cross-validation strategy

**Recommendation**: **ESCALATE to P0 - MVP Blocker for correctness**
- This is a **correctness bug**, not just performance
- Incorrect quantization can cause silent inference corruption
- **Blocks #346 and #401** - must be fixed first
- Add labels: `bug`, `priority/high`, `area/performance`, `mvp:blocker`
- Assign to: **Milestone: MVP v0.1.0** (CRITICAL pre-launch fix)

**GitHub CLI Commands**:
```bash
# Add critical labels (including bug label since this is correctness)
gh issue edit 393 --add-label "bug,priority/high,area/performance,mvp:blocker"

# Assign to MVP v0.1.0 milestone
# gh issue edit 393 --milestone "v0.1.0"

# Add blocking relationship comments
gh issue comment 393 --body "**Blocking Issues**: This correctness bug blocks #346 (TL1) and #401 (TL2) implementations. Must be resolved before table lookup quantization can be properly implemented. Recommended for immediate MVP v0.1.0 milestone."

gh issue comment 346 --body "**Blocked By**: Issue #393 (GGUF quantization mapping) must be resolved first to ensure correct TL1 implementation."

gh issue comment 401 --body "**Blocked By**: Issue #393 (GGUF quantization mapping) must be resolved first to ensure correct TL2 implementation."

# Verify changes
gh issue view 393 --json labels,milestone
```

---

## Medium-Priority Performance Issues (P1 - Important but not MVP-blocking)

### Issues Related to Quantization Performance

**Issue #349**: Optimize I2S Quantizer Fast Path
- **Status**: OPEN, labeled as `enhancement`, `priority/high`, `area/performance`
- **Analysis**: Overlaps significantly with #417 (QK256 dequantization)
- **Recommendation**: **Review for potential consolidation with #417**
- Consider if this is a duplicate or addresses a different aspect of I2S optimization

**GitHub CLI Commands**:
```bash
# Add cross-reference comment
gh issue comment 349 --body "**Note**: This issue may overlap with #417 (QK256 CPU I2S Dequantization). Please review both issues to determine if they should be consolidated or if they address distinct aspects of I2S optimization."

# Verify relationship
gh issue view 349 --json title,body,labels
gh issue view 417 --json title,body,labels
```

---

### Issues Related to Sampling Performance

Both #379 (Top-K) and #380 (Top-P) are covered above with consolidation recommendation.

---

### Infrastructure and Test Performance

**Issue #213**: Optimize test execution performance and timeout issues
- **Status**: OPEN, labeled as `priority/low`, `area/performance`
- **Analysis**: With EnvGuard and nextest infrastructure in place, this may be partially resolved
- **Recommendation**: **Review and potentially close or downgrade**
- Recent guardrail work may have addressed many timeout issues

**GitHub CLI Commands**:
```bash
# Add review comment
gh issue comment 213 --body "**Status Review**: With recent EnvGuard environment isolation (#[serial(bitnet_env)]) and nextest infrastructure improvements, many test timeout issues may be resolved. Recommend reviewing if this issue is still relevant or can be closed."

# If verified as resolved, close with explanation
# gh issue close 213 --comment "Resolved by EnvGuard environment isolation and nextest infrastructure improvements. Test execution is now stable with 5-minute timeout protection and clean parallel execution."
```

---

## Lower-Priority or Deferred Performance Issues (P2)

### Documentation and Benchmarking

**Issue #156**: Update README performance claims with real benchmark data
**Issue #459**: Replace performance claims with receipt-driven examples
- **Status**: Both OPEN, documentation-focused
- **Analysis**: These are important for credibility but not blocking functionality
- **Recommendation**: Defer to v0.2.0 or later, after performance optimizations stabilize
- Should be done after #417, #346, #401 are completed to have accurate numbers

**GitHub CLI Commands**:
```bash
# Add mvp:polish and defer milestone
gh issue edit 156 --add-label "mvp:polish"
gh issue edit 459 --add-label "mvp:polish"

# Add comment explaining deferral
gh issue comment 156 --body "**Timeline Note**: Defer until after core performance optimizations (#417, #346, #401) are completed to ensure accurate benchmark data."
gh issue comment 459 --body "**Timeline Note**: Defer until after core performance optimizations (#417, #346, #401) are completed to ensure accurate receipt-driven examples."
```

---

### Feature-Specific Performance

**Issue #221**: Performance Validation Gap: From Architecture Excellence to Proven Speed
- **Status**: OPEN, labeled as `priority/high`, `area/performance`
- **Analysis**: Meta-issue tracking overall performance validation
- **Recommendation**: Keep as tracking issue, link to specific performance issues

**GitHub CLI Commands**:
```bash
# Add cross-references to concrete issues
gh issue comment 221 --body "**Tracking**: This meta-issue tracks overall performance validation. Key concrete issues:\n- #417 (QK256 dequantization) - MVP blocker\n- #346 (TL1 quantization) - v0.2.0\n- #401 (TL2 quantization) - v0.2.0\n- #319 (KV cache memory pool) - MVP blocker\n- #393 (GGUF quantization mapping) - MVP blocker\n- #379 (Top-K sampling) - v0.3.0\n- #380 (Top-P sampling) - v0.3.0"
```

---

## Issues Potentially Resolved by Recent Work

### Issue #439: Feature Gate Consistency

**Status**: CLOSED (resolved by PR #475)
- **Analysis**: PR #475 unified GPU/CPU feature gates
- **Action**: Verify closed status, no further action needed

**GitHub CLI Commands**:
```bash
# Verify issue is properly closed with reference to PR #475
gh issue view 439 --json state,closedAt,labels

# If not closed, close it with reference to PR #475
# gh issue close 439 --comment "Resolved by PR #475: GPU/CPU feature gate unification completed. All device selection and fallback tests validated."
```

---

## Summary Recommendations by Priority

### P0 - MVP Blockers (Immediate Action Required)

1. **#417**: QK256 CPU I2S Dequantization - Already correctly labeled
2. **#393**: GGUF Quantization Mapping - **ESCALATE to mvp:blocker**
3. **#319**: KV Cache Memory Pool - **ESCALATE to mvp:blocker**

### P1 - High Priority Post-MVP (v0.2.0 Milestone)

4. **#401**: TL2 Quantization - Add labels, assign to v0.2.0
5. **#346**: TL1 Quantization - Add mvp:polish, assign to v0.2.0

### P1 - Sampling Optimization (v0.3.0 or later)

6. **#379** and **#380**: Consolidate or prioritize one sampling approach

### P2 - Documentation and Polish

7. **#156**, **#459**: Performance documentation - Defer until optimizations complete
8. **#221**: Meta-tracking issue - Update with cross-references

### Review for Closure

9. **#213**: Test performance - May be resolved by recent infrastructure work
10. **#439**: Feature gates - Should be closed (PR #475)

---

## GitHub CLI Command Summary

### Critical Actions (Run These First)

```bash
# Escalate #393 (GGUF quantization) to MVP blocker
gh issue edit 393 --add-label "bug,priority/high,area/performance,mvp:blocker"
gh issue comment 393 --body "**Blocking Issues**: This correctness bug blocks #346 (TL1) and #401 (TL2) implementations. Must be resolved before table lookup quantization can be properly implemented. Recommended for immediate MVP v0.1.0 milestone."

# Escalate #319 (KV cache) to MVP blocker
gh issue edit 319 --add-label "priority/high,area/performance,mvp:blocker"
gh issue comment 319 --body "**Priority Escalation**: This issue contains production-blocking memory management stubs that prevent proper multi-session inference. Recommended for MVP v0.1.0 milestone as critical infrastructure fix. Estimated effort: 2-3 weeks."

# Label #401 (TL2) appropriately
gh issue edit 401 --add-label "priority/high,area/performance,mvp:polish"
gh issue comment 401 --body "**Priority Update**: This issue contains a critical dispatch bug where TL2 routes to TL1 quantization. Recommended for v0.2.0 milestone as high-priority post-MVP fix."

# Add mvp:polish to post-MVP high-priority issues
gh issue edit 346 --add-label "mvp:polish"
gh issue edit 379 --add-label "mvp:polish"
gh issue edit 380 --add-label "mvp:polish"

# Add blocking relationships
gh issue comment 346 --body "**Blocked By**: Issue #393 (GGUF quantization mapping) must be resolved first to ensure correct TL1 implementation."
gh issue comment 401 --body "**Blocked By**: Issue #393 (GGUF quantization mapping) must be resolved first to ensure correct TL2 implementation."

# Add consolidation notes for sampling issues
gh issue comment 379 --body "**Note**: This issue has significant overlap with #380 (Top-P Sampling). Consider consolidating into a unified sampling optimization effort or prioritizing one approach first to avoid duplicated work."
gh issue comment 380 --body "**Note**: This issue has significant overlap with #379 (Top-K Sampling). Consider consolidating into a unified sampling optimization effort or prioritizing one approach first to avoid duplicated work."
```

### Documentation and Polish Actions

```bash
# Defer documentation issues
gh issue edit 156 --add-label "mvp:polish"
gh issue edit 459 --add-label "mvp:polish"
gh issue comment 156 --body "**Timeline Note**: Defer until after core performance optimizations (#417, #346, #401) are completed to ensure accurate benchmark data."
gh issue comment 459 --body "**Timeline Note**: Defer until after core performance optimizations (#417, #346, #401) are completed to ensure accurate receipt-driven examples."

# Update meta-tracking issue
gh issue comment 221 --body "**Tracking**: This meta-issue tracks overall performance validation. Key concrete issues:\n- #417 (QK256 dequantization) - MVP blocker\n- #346 (TL1 quantization) - v0.2.0\n- #401 (TL2 quantization) - v0.2.0\n- #319 (KV cache memory pool) - MVP blocker\n- #393 (GGUF quantization mapping) - MVP blocker\n- #379 (Top-K sampling) - v0.3.0\n- #380 (Top-P sampling) - v0.3.0"
```

### Review and Potential Closure

```bash
# Review test performance issue
gh issue comment 213 --body "**Status Review**: With recent EnvGuard environment isolation (#[serial(bitnet_env)]) and nextest infrastructure improvements, many test timeout issues may be resolved. Recommend reviewing if this issue is still relevant or can be closed."

# Check if #439 is properly closed
gh issue view 439 --json state,closedAt,labels

# Review #349 for potential consolidation with #417
gh issue comment 349 --body "**Note**: This issue may overlap with #417 (QK256 CPU I2S Dequantization). Please review both issues to determine if they should be consolidated or if they address distinct aspects of I2S optimization."
```

---

## Dependency Graph

```
MVP v0.1.0 Blockers:
  #393 (GGUF mapping) [BLOCKS] → #346 (TL1), #401 (TL2)
  #417 (QK256 dequant) [INDEPENDENT]
  #319 (KV cache pool) [INDEPENDENT]

Post-MVP v0.2.0:
  #346 (TL1) [BLOCKED BY #393]
  #401 (TL2) [BLOCKED BY #393]

Future v0.3.0:
  #379 (Top-K) [CONSOLIDATE?] ← → #380 (Top-P)

Documentation (after optimizations):
  #156, #459 [BLOCKED BY #417, #346, #401]
```

---

## Priority Matrix

| Issue | Current Labels | Recommended Labels | Milestone | Priority | Effort | Blockers |
|-------|---------------|-------------------|-----------|----------|--------|----------|
| #417  | mvp:blocker, priority/high, area/performance | ✅ Correct | v0.1.0 | P0 | 2.5-4 days | None |
| #393  | None | **bug, priority/high, area/performance, mvp:blocker** | v0.1.0 | P0 | TBD | None; BLOCKS #346, #401 |
| #319  | None | **priority/high, area/performance, mvp:blocker** | v0.1.0 | P0 | 2-3 weeks | None |
| #401  | None | **priority/high, area/performance, mvp:polish** | v0.2.0 | P1 | 2.5-4 days | #393 |
| #346  | enhancement, priority/high, area/performance | **+mvp:polish** | v0.2.0 | P1 | 2.5-4 days | #393 |
| #379  | enhancement, priority/high, area/performance | **+mvp:polish** | v0.3.0 | P1 | 4-5 weeks | None; OVERLAPS #380 |
| #380  | enhancement, priority/high, area/performance | **+mvp:polish** | v0.3.0 | P1 | 2-3 weeks | None; OVERLAPS #379 |
| #156  | documentation, enhancement, priority/high, area/performance | **+mvp:polish** | Post-v0.2.0 | P2 | TBD | #417, #346, #401 |
| #459  | None | **documentation, mvp:polish** | Post-v0.2.0 | P2 | TBD | #417, #346, #401 |
| #213  | priority/low, area/performance | Review for closure | TBD | P2 | N/A | Possibly resolved |
| #221  | priority/high, area/performance | Keep as tracker | Ongoing | P1 | N/A | Meta-issue |
| #349  | enhancement, priority/high, area/performance | Review vs #417 | TBD | P1 | TBD | May overlap #417 |
| #439  | CLOSED | ✅ Resolved by PR #475 | N/A | N/A | N/A | None |

---

## Milestone Recommendations

### MVP v0.1.0 (Pre-Launch Critical)
- **#417**: QK256 CPU I2S Dequantization (already assigned)
- **#393**: GGUF Quantization Mapping Correctness (escalate)
- **#319**: KV Cache Memory Pool Implementation (escalate)

**Rationale**: These three issues block production viability:
- #417: Performance blocker for QK256 inference
- #393: Correctness blocker for quantization (silent corruption risk)
- #319: Infrastructure blocker for multi-session scenarios

### MVP v0.2.0 (Post-Launch High-Priority)
- **#401**: TL2 Quantization (blocked by #393)
- **#346**: TL1 Quantization (blocked by #393)

**Rationale**: TL1/TL2 are important for quantization quality but not blockers for initial launch. Must resolve #393 first.

### Future v0.3.0 (Optimization Wave)
- **#379**: Top-K Sampling (or consolidated sampling optimization)
- **#380**: Top-P Sampling (or consolidated sampling optimization)

**Rationale**: Sampling optimizations have high impact but longer timelines (2-5 weeks). Consider consolidating into single effort.

### Documentation (Continuous)
- **#156**: README performance claims
- **#459**: Receipt-driven examples
- **#221**: Meta-tracking issue

**Rationale**: Should be updated after each major performance optimization is completed.

---

## Conclusion

The BitNet-rs performance landscape post-PR #475 shows:

1. **Three critical MVP blockers** identified:
   - #417 (QK256) - already recognized
   - #393 (GGUF) - needs escalation (correctness + blocks TL1/TL2)
   - #319 (KV cache) - needs escalation (infrastructure)

2. **Clear post-MVP path** with TL1/TL2 optimizations (#346, #401) blocked by #393

3. **Sampling optimizations** (#379, #380) should be consolidated or sequenced to avoid duplication

4. **Recent infrastructure work** (PR #475, EnvGuard, receipts) has resolved #439 and may have resolved #213

**Next Steps**:
1. Execute critical GitHub CLI commands to escalate #393 and #319
2. Add blocking relationships between #393 → #346/#401
3. Consolidate or sequence sampling optimizations (#379 + #380)
4. Review #213 and #349 for potential closure/consolidation
5. Update milestones based on recommendations above

---

*Analysis Date: November 11, 2025*
*Analyzed By: BitNet-rs Research Specialist*
*Context: Post-PR #475 (GPU/CPU Feature Gate Unification)*
