# bitnet-rs GPU/CUDA Issues Analysis
**Date**: 2025-11-11
**Context**: Post-PR #475 (GPU feature gate unification - MERGED 2025-11-03)
**Analyst**: bitnet-rs GitHub Research Specialist

---

## Executive Summary

**Total GPU/CUDA Open Issues**: 89 (filtered from full issue list)
**Critical Context**: Issue #439 (GPU feature gate consistency) **RESOLVED** in PR #475
**GPU Infrastructure Status**: Implemented but requires validation and production hardening
**Key Gap**: Receipt verification system needs GPU kernel validation gates

### Issue Status Distribution

| Status | Count | Notes |
|--------|-------|-------|
| Active/Recent (updated 2025-10+) | 6 | Issues #432, #436, #450, #455, #394, #386 |
| Stale (updated 2025-09) | 14+ | Stub replacement, memory management, capability detection |
| Architecture/Planning | 5+ | #436 Roadmap, #450 CUDA backend MVP, #85 infrastructure |
| Resolved by PR #475 | 1 | #439 feature gate unification |

---

## Key Issue Deep-Dive

### 1. Issue #450: CUDA Backend, Receipts, and Bench Harness ⚡ HIGH PRIORITY

**Status**: Open (Created: 2025-10-13, Updated: 2025-10-13)
**Current Labels**: None (NEEDS LABELING)
**Context**: Post-PR #475, aligns with GPU infrastructure validation needs

#### Summary
Comprehensive CUDA integration initiative covering backend abstraction, profiling infrastructure, and receipt emission for GPU validation gates.

#### Goals
- **Backend Abstraction**: Clean `cpu | wasm | cuda` trait-based architecture
- **CUDA MVP**: Token generation on real GPU (batch=1, small model)
- **Profiling Hooks**: NVTX ranges for Nsight Systems/Compute integration
- **Receipt Emission**: Machine-readable `receipt.json` with env + metrics + artifacts
- **Microbench Harness**: Fixed prompts across backends with CSV/JSON results
- **CLI Thresholds**: Configurable gates (tokens/sec, wall-time, determinism)
- **Reproducibility**: Dockerfiles and scripts for consistent runs

#### Deliverables
1. Backend trait with `Cuda` implementation (`--features cuda`)
2. CUDA inference MVP with real GPU token generation
3. NVTX profiling support for Nsight tooling
4. Receipt emission with GPU-specific metadata
5. Microbench harness comparing CPU/WASM/CUDA
6. CLI with backend selection, seeds, thresholds
7. `Dockerfile.cuda` and documentation

#### Acceptance Criteria
- ✅ CUDA backend generates tokens on real GPU
- ✅ NVTX ranges present for Nsight trace collection
- ✅ Each run emits receipt with environment + metrics + artifacts
- ✅ Microbench produces comparable TPS and p95 wall-time
- ✅ CLI returns non-zero on threshold breach
- ✅ Docs and Dockerfile enable reproducibility

#### Dependency Analysis
**Blocked by**: None (PR #475 resolved feature gate dependencies)
**Blocks**:
- Issue #455 (GPU receipt gate)
- Issue #432 (GPU test race conditions)
- Multiple stub replacement issues (#317, #364, #374, #293, etc.)

#### Recommended Actions
```bash
# Label this issue
gh issue edit 450 --add-label "area/gpu,priority/high,mvp:blocker,enhancement"

# Assign to MVP v0.1.0 milestone
gh issue edit 450 --milestone "MVP v0.1.0"

# Add comment with PR #475 context
gh issue comment 450 --body "**Context Update**: PR #475 (GPU feature gate unification) merged 2025-11-03. Feature gate consistency (Issue #439) is now RESOLVED. This unblocks CUDA backend implementation with unified \`#[cfg(any(feature=\"gpu\", feature=\"cuda\"))]\` predicates across the workspace.

**Implementation Status**:
- ✅ Feature gates unified
- ✅ Device capability detection in place (\`gpu_compiled()\`, \`gpu_available_runtime()\`)
- ⏳ CUDA inference MVP pending
- ⏳ Receipt emission system needs GPU kernel validation

**Next Steps**:
1. Implement \`Backend\` trait with \`Cuda\` variant
2. Integrate NVTX profiling hooks
3. Emit receipts with GPU-specific metadata (CUDA version, compute capability)
4. Wire into \`xtask benchmark --backend cuda\` (relates to #455)

**Related Issues**: #455 (GPU receipt gate), #432 (GPU test race), #439 (RESOLVED)"
```

---

### 2. Issue #432: GPU Test Race Condition ⚠️ MEDIUM-HIGH PRIORITY

**Status**: Open (Created: 2025-10-04, **Updated: 2025-11-11** - RECENT ACTIVITY)
**Current Labels**: None (NEEDS LABELING)
**Context**: Affected by PR #475 EnvGuard implementation

#### Summary
GPU tests fail with "all zero results" when run in parallel (`--test-threads=4+`) but pass serially. Root cause is CUDA kernel execution race condition with missing stream synchronization.

#### Affected Tests
- `bitnet-kernels::gpu::tests::gpu_kernel_tests::test_cuda_matmul_correctness`
- `bitnet-kernels::gpu::tests::gpu_kernel_tests::test_batch_processing`
- `bitnet-inference::ac8_gpu_performance_tests::test_ac8_gpu_performance_baselines`

#### Root Cause Analysis
1. **Missing Stream Synchronization**: CUDA kernel launches followed by D2H copy without explicit `stream.synchronize()`
2. **Stub Implementation**: `CudaKernel::synchronize_all()` is currently a no-op (logs only, no driver call)
3. **Batch API Hazard**: `batch_matmul_i2s()` enqueues multiple kernels and returns without barrier
4. **Parallel Test Interference**: Multiple tests access GPU concurrently without serialization

#### Proposed Fix (3 Layers)

**1. Runtime Correctness**
```rust
// Make synchronization real
pub fn synchronize_all(&self) -> Result<()> {
    self.stream.synchronize().map_err(|e| KernelError::GpuError {
        reason: format!("Stream sync failed: {:?}", e),
    })?;
    Ok(())
}

// Fence before D2H copy
unsafe { builder.launch(cfg) }?;
self.stream.synchronize()?;  // ADD THIS
let c_host: Vec<f32> = self.stream.memcpy_dtov(&c_dev)?;

// Batch API barrier
for (a, b, c, m, n, k) in batches.iter_mut() {
    self.launch_matmul(a, b, c, *m, *n, *k)?;
}
self.synchronize_all()?;  // ADD THIS
```

**2. Test Isolation**
```rust
use serial_test::serial;

#[test]
#[serial]  // Serialize GPU tests
fn test_cuda_matmul_correctness() { /* ... */ }
```

**3. CI Configuration**
- GPU test job with `RUST_TEST_THREADS=1` or `-- --test-threads=1`
- Keep CPU tests parallelized

#### PR #475 Context
**Comment from 2025-11-11**: EnvGuard environment isolation added with `#[serial(bitnet_env)]` annotations. This provides:
- Thread-safe environment variable manipulation
- Automatic restoration on drop
- Parallel test execution safety for env-mutating tests

**Status**: EnvGuard addresses *environment variable* races but NOT CUDA stream synchronization races. This issue still requires CUDA-specific fixes.

#### Recommended Actions
```bash
# Label appropriately
gh issue edit 432 --add-label "bug,area/gpu,area/testing,priority/high,flaky-test"

# Assign to MVP v0.1.0 milestone
gh issue edit 432 --milestone "MVP v0.1.0"

# Add clarifying comment
gh issue comment 432 --body "**Status Update (2025-11-11)**: PR #475 added EnvGuard for environment variable isolation, which helps with \`BITNET_GPU_FAKE\` manipulation in tests. However, this issue's **root cause is CUDA stream synchronization**, not environment variables.

**Remaining Work**:
1. ✅ Environment isolation (EnvGuard from PR #475)
2. ⏳ Implement real \`synchronize_all()\` (replace stub)
3. ⏳ Add stream barriers before D2H copies
4. ⏳ Add barrier after batch kernel enqueues
5. ⏳ Serialize GPU kernel tests with \`#[serial]\`

**Verification Plan**:
\`\`\`bash
# Should pass with --test-threads=1
cargo test -p bitnet-kernels --no-default-features --features gpu -- --test-threads=1

# Should still pass after serialization annotations
cargo test -p bitnet-kernels --no-default-features --features gpu
\`\`\`

**Estimated Effort**: 2-3 hours (runtime fixes + test annotations)"
```

---

### 3. Issue #439: GPU Feature Gate Consistency ✅ RESOLVED

**Status**: CLOSED (Resolved by PR #475, merged 2025-11-03)
**Labels**: N/A (issue closed)
**Milestone**: MVP v0.1.0

#### Resolution Summary
PR #475 successfully unified GPU feature gates across the entire workspace:

**Implemented Changes**:
1. ✅ **AC1 - Kernel Gates Unified**: All CUDA symbols now use `#[cfg(any(feature="gpu", feature="cuda"))]`
2. ✅ **AC2 - Build Script Parity**: Both `CARGO_FEATURE_GPU` and `CARGO_FEATURE_CUDA` checked
3. ✅ **AC3 - Shared Helpers**: `gpu_compiled()` and `gpu_available_runtime()` in `bitnet-kernels/src/device_features.rs`
4. ✅ **AC4 - Feature Matrix**: All 4 combinations passing (no-features, cpu, gpu, cpu+gpu)
5. ✅ **AC5 - xtask Preflight**: `cargo run -p xtask -- preflight` validates GPU detection
6. ✅ **AC8 - Gitignore**: `**/*.proptest-regressions` pattern added

**Impact on Open Issues**:
- Unblocks Issue #450 (CUDA backend implementation)
- Simplifies Issue #432 (GPU test race - feature gates now consistent)
- Enables all stub replacement issues (#317, #364, #374, etc.) to use unified predicates

#### Related Issues Still Open
While #439 is resolved, several related GPU issues remain:
- #450: CUDA backend MVP (unblocked)
- #455: GPU receipt gate (needs #450)
- #432: GPU test race conditions (partially addressed by EnvGuard)
- Multiple stub replacement issues (now have consistent feature gates)

---

### 4. Issue #455: GPU Receipt Gate with Skip-Clean Fallback 🎯 HIGH PRIORITY

**Status**: Open (Created: 2025-10-14, Updated: 2025-10-14)
**Current Labels**: None (NEEDS LABELING)
**Context**: Follow-up to PR #452 (Receipt Verification Gate)

#### Summary
Add GPU receipt verification that runs on CUDA-capable runners and cleanly skips on CPU-only runners without blocking PR merges.

#### Acceptance Criteria
- ✅ `xtask benchmark --backend cuda` writes GPU receipt to `ci/inference-gpu.json`
- ✅ Receipt contains GPU-specific metadata (CUDA version, GPU model, compute capability)
- ✅ CI job runs on GPU runners and verifies with `--require-gpu-kernels`
- ✅ CI job exits 0 with clear "skipped (no CUDA)" message on CPU-only runners
- ✅ Job remains green when skipped (doesn't block PR merge)

#### Implementation Notes
**Files**:
- `.github/workflows/model-gates.yml`: Add `gpu-receipt-gate` job
- `xtask/src/main.rs`: Add `--backend cuda` support to benchmark

**Receipt Format** (from #450):
```json
{
  "commit_sha": "abc123",
  "model_sha": "bitnet-0.3.1+q1",
  "backend": "cuda",
  "cuda_version": "12.4",
  "driver_version": "550.54",
  "sm_arch": "sm90",
  "precision": "1bit-accum-fp16",
  "tokens_per_sec": 0.52,
  "gate_wall_time_ms": 2980,
  "kernels": ["gemm_fp16", "i2s_gpu_dequant", "cuda_softmax"]
}
```

#### Dependency Analysis
**Depends on**:
- Issue #450 (CUDA backend implementation for `--backend cuda`)
- PR #452 (Receipt verification infrastructure - assumed merged)

**Blocks**:
- GPU CI lane activation
- GPU performance regression detection
- GPU kernel validation gates

#### Recommended Actions
```bash
# Label appropriately
gh issue edit 455 --add-label "area/ci,area/gpu,priority/high,enhancement"

# Assign to milestone
gh issue edit 455 --milestone "MVP v0.1.0"

# Add context comment
gh issue comment 455 --body "**Dependency Status**:
- ✅ PR #475: GPU feature gates unified (Issue #439 resolved)
- ⏳ Issue #450: CUDA backend implementation (BLOCKS THIS)
- ✅ PR #452: Receipt verification gate (assumed complete)

**Implementation Approach**:
1. Extend \`xtask benchmark\` with \`--backend cuda\` flag
2. Emit GPU-specific receipt to \`ci/inference-gpu.json\`:
   - CUDA version, driver version, SM architecture
   - GPU model name (e.g., \"RTX 5070 Ti\")
   - Compute capability (e.g., \"8.9\")
   - GPU kernel list (validates real GPU execution)
3. Add \`gpu-receipt-gate\` job to \`.github/workflows/model-gates.yml\`
4. Implement skip logic: detect CUDA availability, exit 0 with message if unavailable
5. Verify with \`--require-gpu-kernels\` flag (from #450 spec)

**Verification Commands**:
\`\`\`bash
# CPU-only runner (should skip cleanly)
cargo run -p xtask -- benchmark --backend cuda || echo \"Skipped: no CUDA\"

# GPU runner (should generate receipt)
cargo run -p xtask -- benchmark --backend cuda --model model.gguf --tokens 128
cargo run -p xtask -- verify-receipt ci/inference-gpu.json --require-gpu-kernels
\`\`\`

**Estimated Effort**: 1-2 days (depends on #450 completion)"
```

---

## GPU Stub Replacement Issues (Post-MVP)

These issues represent technical debt and production hardening work. With PR #475 merged, they now have consistent feature gates for implementation.

### High Priority Stubs

#### Issue #374: GPU Utilization Monitoring
**Summary**: Replace hardcoded `0.85` placeholder with real CUDA monitoring
**Labels**: `enhancement`, `priority/medium`, `area/performance`, `area/infrastructure`
**Status**: Awaiting NVML integration
**Recommendation**: Post-MVP (not blocking inference functionality)

#### Issue #364: Mixed Precision Detection
**Summary**: Replace stub `supports_mixed_precision()` with hardware capability checking
**Labels**: `enhancement`, `area/performance`
**Status**: Stub returns config value instead of querying GPU
**Recommendation**: Combine with #293 (Tensor Core detection), implement together

#### Issue #293: Tensor Core Support Detection
**Summary**: Replace stub `supports_tensor_cores()` with real hardware capability detection
**Labels**: `enhancement`, `priority/high`, `area/performance`
**Status**: Related to #364 (mixed precision)
**Recommendation**: High priority for GPU performance optimization

#### Issue #317: GPU Forward Pass Implementation
**Summary**: Replace placeholder GPU forward pass with real CUDA-accelerated inference
**Labels**: None (NEEDS LABELING)
**Status**: Critical stub - affects core inference functionality
**Recommendation**: **MVP BLOCKER** - should be resolved with #450 CUDA backend

### Medium Priority Stubs

#### Issue #363: GPU Discovery and Memory Detection
**Summary**: Replace `BITNET_GPU_FAKE` environment variable simulation with real discovery
**Status**: Environment variable workaround implemented in PR #475
**Recommendation**: Post-MVP (current workaround sufficient for testing)

#### Issue #366: GPU Memory Query
**Summary**: Replace hardcoded 8GB placeholder with actual CUDA memory query
**Recommendation**: Post-MVP (memory management optimization)

#### Issue #367: GPU Memory Deallocation
**Summary**: Implement real GPU memory deallocation in `GpuMemoryManager`
**Labels**: `enhancement`, `priority/high`, `area/performance`
**Recommendation**: Post-MVP (memory leak prevention)

#### Issue #313: GPU Memory Manager Stubs
**Summary**: Replace stub allocation functions with real CUDA memory management
**Recommendation**: Post-MVP (comprehensive memory management)

### Lower Priority / Infrastructure

#### Issue #215: GPU Test Coverage Enhancement
**Summary**: Expand GPU testing across hardware configurations and failure scenarios
**Labels**: `enhancement`, `priority/medium`, `area/performance`
**Comment from 2025-09-19**: Recent GPU infrastructure improvements provide foundation
**Recommendation**: Ongoing effort, prioritize after core functionality stable

#### Issue #322: Dynamic GPU Device Detection
**Summary**: Implement dynamic GPU device detection and management in inference engine
**Recommendation**: Post-MVP (infrastructure enhancement)

#### Issue #303: Dynamic GPU Workspace Size
**Summary**: Replace 6GB hardcoded assumption with dynamic GPU memory detection
**Recommendation**: Post-MVP (optimization)

---

## Quantization-Specific GPU Issues

### Issue #394: TL1Quantizer CUDA Integration Verification
**Status**: Open (Updated: 2025-09-29)
**Summary**: Verify `TL1Quantizer::quantize_cuda` integration and test coverage

**Analysis**:
- Method IS integrated (called from `TL1Quantizer::quantize` at line 171)
- Feature-gated properly: `#[cfg(feature = "cuda")]`
- Concern: Potential test coverage gap

**Recommendation**:
```bash
gh issue edit 394 --add-label "area/quantization,area/gpu,area/testing,priority/medium"
gh issue comment 394 --body "**Status Update (Post-PR #475)**:

Integration confirmed at \`TL1Quantizer::quantize\` (line 171):
\`\`\`rust
if !device.is_cpu() {
    #[cfg(feature = \"cuda\")]
    {
        if device.is_cuda()
            && bitnet_kernels::gpu::cuda::is_cuda_available()
            && let Ok(res) = self.quantize_cuda(tensor)
        {
            return Ok(res);
        }
    }
}
\`\`\`

**Validation Needed**:
1. ✅ Feature gate consistent (uses \`cuda\` feature)
2. ⏳ Add test coverage for GPU quantization path
3. ⏳ Cross-validate TL1 CUDA quantization vs CPU implementation
4. ⏳ Verify graceful fallback when GPU unavailable

**Test Commands**:
\`\`\`bash
# Ensure TL1 GPU path is tested
cargo test -p bitnet-quantization --no-default-features --features gpu test_tl1
cargo test -p bitnet-quantization --no-default-features --features gpu tl1_cuda

# Cross-validation
BITNET_GPU_FAKE=cuda cargo test -p bitnet-quantization tl1_quantize
\`\`\`"
```

### Issue #356: I2SQuantizer CUDA Dead Code Investigation
**Status**: Open (Created: 2025-09-27)
**Summary**: `quantize_cuda` method flagged as potentially unused, needs architectural integration review

**Recommendation**: Similar to #394, verify integration and add test coverage. Label as `area/quantization,area/gpu,priority/medium`.

---

## Architecture and Planning Issues

### Issue #436: Roadmap
**Status**: Open (Updated: 2025-10-14)
**Labels**: None
**Recommendation**: Review for GPU-related milestones and priorities. Ensure alignment with #450 (CUDA backend MVP).

### Issue #85: Infrastructure Stubs and Mocks
**Status**: Open
**Labels**: `enhancement`, `priority/low`, `area/infrastructure`
**Summary**: GPU multi-device, health monitoring, caching infrastructure
**Recommendation**: Post-MVP; covers multiple stub replacement issues listed above.

---

## Cross-Validation and Testing Issues

### Issue #414: Missing GPU Acceleration Cross-Validation Tests
**Status**: Open
**Labels**: `enhancement`, `priority/high`, `area/performance`, `area/infrastructure`
**Summary**: AC1 GPU acceleration cross-validation tests missing
**Recommendation**:
```bash
gh issue edit 414 --add-label "area/testing,area/gpu,mvp:consideration"
gh issue comment 414 --body "**Post-PR #475 Status**:

GPU feature gates now unified. Ready for cross-validation test implementation.

**Test Requirements**:
1. CPU vs GPU inference parity validation
2. Quantization accuracy cross-check (I2S, TL1, TL2)
3. Performance baseline validation (relates to #450 receipt system)
4. Mixed precision accuracy tolerance testing

**Dependencies**:
- Issue #450: CUDA backend MVP (for real GPU execution)
- Issue #432: GPU test race conditions (for stable test execution)

**Proposed Test Structure**:
\`\`\`rust
#[test]
#[cfg(any(feature = \"gpu\", feature = \"cuda\"))]
fn test_gpu_cpu_inference_parity() {
    let cpu_result = inference_cpu(model, input);
    let gpu_result = inference_gpu(model, input);
    assert_tensors_close(&cpu_result, &gpu_result, 1e-3);
}
\`\`\`"
```

### Issue #270: Nightly GPU Correctness Tests
**Status**: Open
**Summary**: Implement nightly GPU correctness test suite
**Recommendation**: Post-MVP; aligns with #215 (GPU test coverage enhancement).

---

## Priority Ranking and Milestone Assignment

### MVP v0.1.0 Blockers (CRITICAL PATH)

1. **Issue #450**: CUDA Backend, Receipts, Bench Harness ⚡
   - **Priority**: P0 (Critical)
   - **Labels**: `area/gpu`, `priority/high`, `mvp:blocker`, `enhancement`
   - **Milestone**: MVP v0.1.0
   - **Blocks**: #455, #432 resolution, multiple stub issues
   - **Estimated Effort**: 2-3 weeks
   - **Unblocked by**: PR #475 (RESOLVED)

2. **Issue #432**: GPU Test Race Condition ⚠️
   - **Priority**: P1 (High)
   - **Labels**: `bug`, `area/gpu`, `area/testing`, `priority/high`, `flaky-test`
   - **Milestone**: MVP v0.1.0
   - **Estimated Effort**: 2-3 hours
   - **Status**: Partially addressed by PR #475 (EnvGuard), needs CUDA sync fixes

3. **Issue #455**: GPU Receipt Gate 🎯
   - **Priority**: P1 (High)
   - **Labels**: `area/ci`, `area/gpu`, `priority/high`, `enhancement`
   - **Milestone**: MVP v0.1.0
   - **Depends on**: #450
   - **Estimated Effort**: 1-2 days

4. **Issue #317**: GPU Forward Pass Implementation
   - **Priority**: P0 (Critical) - Core Inference
   - **Labels**: NEEDS: `area/gpu`, `priority/critical`, `mvp:blocker`
   - **Milestone**: MVP v0.1.0
   - **Note**: Should be resolved as part of #450

### MVP v0.1.0 Considerations (HIGH VALUE)

5. **Issue #414**: GPU Acceleration Cross-Validation Tests
   - **Priority**: P2 (High Value)
   - **Labels**: `enhancement`, `priority/high`, `area/performance`, `area/infrastructure`, `area/testing`, `area/gpu`, `mvp:consideration`
   - **Milestone**: MVP v0.1.0 or v0.2.0
   - **Depends on**: #450

6. **Issue #293**: Tensor Core Support Detection
   - **Priority**: P2 (Performance Optimization)
   - **Labels**: `enhancement`, `priority/high`, `area/performance`, `area/gpu`
   - **Milestone**: MVP v0.1.0 or v0.2.0
   - **Combine with**: #364 (mixed precision detection)

7. **Issue #394**: TL1Quantizer CUDA Integration Verification
   - **Priority**: P2 (Correctness)
   - **Labels**: NEEDS: `area/quantization`, `area/gpu`, `area/testing`, `priority/medium`
   - **Milestone**: MVP v0.1.0 or v0.2.0

### Post-MVP v0.2.0+ (PRODUCTION HARDENING)

8. **Issue #374**: GPU Utilization Monitoring
   - **Priority**: P3 (Production Ops)
   - **Labels**: `enhancement`, `priority/medium`, `area/performance`, `area/infrastructure`, `area/gpu`
   - **Milestone**: v0.2.0

9. **Issue #364**: Mixed Precision Detection
   - **Priority**: P3 (Optimization)
   - **Labels**: `enhancement`, `area/performance`, `area/gpu`
   - **Milestone**: v0.2.0
   - **Combine with**: #293

10. **Issue #363**: GPU Discovery and Memory Detection
    - **Priority**: P3 (Infrastructure)
    - **Labels**: NEEDS: `area/gpu`, `priority/medium`, `enhancement`
    - **Milestone**: v0.2.0

11. **Issue #367**: GPU Memory Deallocation
    - **Priority**: P3 (Memory Management)
    - **Labels**: `enhancement`, `priority/high`, `area/performance`, `area/gpu`
    - **Milestone**: v0.2.0

12. **Issue #215**: GPU Test Coverage Enhancement
    - **Priority**: P3 (Testing Infrastructure)
    - **Labels**: `enhancement`, `priority/medium`, `area/performance`, `area/gpu`, `area/testing`
    - **Milestone**: Ongoing

13. **Issue #85**: Infrastructure Stubs and Mocks
    - **Priority**: P4 (Long-term Cleanup)
    - **Labels**: `enhancement`, `priority/low`, `area/infrastructure`, `area/gpu`
    - **Milestone**: v0.3.0+

---

## Dependency Graph

```
PR #475 (MERGED) - GPU Feature Gate Unification
    ↓ UNBLOCKS
Issue #450 - CUDA Backend MVP ⚡ CRITICAL PATH
    ↓ BLOCKS
    ├─ Issue #455 - GPU Receipt Gate 🎯
    ├─ Issue #317 - GPU Forward Pass (part of #450)
    └─ Issue #414 - GPU Cross-Validation Tests

Issue #432 - GPU Test Race Condition ⚠️
    ↓ PARTIALLY ADDRESSED BY
PR #475 (EnvGuard) + Needs CUDA Stream Sync

Issue #450 ─┐
Issue #432 ─┤ ENABLES
Issue #455 ─┘
    ↓
Stub Replacement Wave:
    ├─ #374 GPU Utilization
    ├─ #364 Mixed Precision Detection
    ├─ #293 Tensor Core Detection
    ├─ #363 GPU Discovery
    ├─ #367 GPU Memory Deallocation
    ├─ #366 GPU Memory Query
    ├─ #313 GPU Memory Manager
    ├─ #322 Dynamic Device Detection
    └─ #303 Dynamic Workspace Size

Issue #394 - TL1 CUDA Integration
Issue #356 - I2S CUDA Integration
    ↓ VALIDATES
Quantization GPU Path Correctness
```

---

## GitHub CLI Commands for Triage

### Critical Path (Execute Now)

```bash
# Issue #450 - CUDA Backend MVP
gh issue edit 450 --add-label "area/gpu,priority/high,mvp:blocker,enhancement" --milestone "MVP v0.1.0"
gh issue comment 450 --body "**Context Update (2025-11-11)**: PR #475 merged, Issue #439 RESOLVED. Feature gates unified. Ready for CUDA backend implementation.

**Unblocked**: Workspace now has consistent \`#[cfg(any(feature=\"gpu\", feature=\"cuda\"))]\` predicates and \`gpu_compiled()\`/\`gpu_available_runtime()\` helpers in \`bitnet-kernels/src/device_features.rs\`.

**Critical Path**: This issue BLOCKS #455 (GPU receipt gate), #432 resolution, and multiple stub replacement issues.

**Implementation Priority**: P0 - MVP Blocker
**Estimated Effort**: 2-3 weeks"

# Issue #432 - GPU Test Race Condition
gh issue edit 432 --add-label "bug,area/gpu,area/testing,priority/high,flaky-test" --milestone "MVP v0.1.0"
gh issue comment 432 --body "**Status Update (2025-11-11)**: PR #475 added EnvGuard for environment variable isolation (\`#[serial(bitnet_env)]\`). This addresses *env var* races but NOT CUDA stream synchronization races.

**Remaining Work**:
1. ✅ Environment isolation (EnvGuard from PR #475)
2. ⏳ Implement real \`synchronize_all()\` (replace stub)
3. ⏳ Add stream barriers before D2H copies
4. ⏳ Serialize GPU kernel tests with \`#[serial]\`

**Priority**: P1 - High (test stability)
**Estimated Effort**: 2-3 hours"

# Issue #455 - GPU Receipt Gate
gh issue edit 455 --add-label "area/ci,area/gpu,priority/high,enhancement" --milestone "MVP v0.1.0"
gh issue comment 455 --body "**Dependency Status**:
- ✅ PR #475: GPU feature gates unified
- ⏳ Issue #450: CUDA backend implementation (BLOCKS THIS)

**Priority**: P1 - High (GPU validation gate)
**Depends on**: #450 completion
**Estimated Effort**: 1-2 days"

# Issue #317 - GPU Forward Pass (part of #450)
gh issue edit 317 --add-label "area/gpu,priority/critical,mvp:blocker,enhancement" --milestone "MVP v0.1.0"
gh issue comment 317 --body "**Status**: This issue should be resolved as part of Issue #450 (CUDA Backend MVP). Current placeholder forward pass will be replaced with real CUDA-accelerated inference.

**Priority**: P0 - Critical (core inference functionality)
**Part of**: Issue #450"

# Issue #414 - GPU Cross-Validation Tests
gh issue edit 414 --add-label "area/testing,area/gpu,mvp:consideration" --milestone "MVP v0.1.0"
gh issue comment 414 --body "**Post-PR #475 Status**: GPU feature gates unified. Ready for cross-validation test implementation.

**Dependencies**: Issue #450 (CUDA backend), Issue #432 (test stability)
**Priority**: P2 - High Value
**Estimated Effort**: 3-5 days"
```

### High Value Issues

```bash
# Issue #293 - Tensor Core Detection
gh issue edit 293 --add-label "area/gpu" --milestone "MVP v0.1.0"
gh issue comment 293 --body "**Post-PR #475**: Unified feature gates enable implementation. Consider combining with Issue #364 (mixed precision detection) for comprehensive hardware capability detection.

**Priority**: P2 - Performance Optimization
**Combine with**: #364"

# Issue #394 - TL1 CUDA Integration
gh issue edit 394 --add-label "area/quantization,area/gpu,area/testing,priority/medium"
gh issue comment 394 --body "**Integration Confirmed**: Method called from \`TL1Quantizer::quantize\` line 171. Feature-gated properly.

**Validation Needed**: Test coverage for GPU quantization path and cross-validation vs CPU.

**Priority**: P2 - Correctness Verification"

# Issue #356 - I2S CUDA Integration
gh issue edit 356 --add-label "area/quantization,area/gpu,priority/medium"
gh issue comment 356 --body "**Similar to Issue #394**: Verify integration and add test coverage for I2S CUDA quantization path."
```

### Post-MVP Issues

```bash
# Issue #374 - GPU Utilization Monitoring
gh issue edit 374 --add-label "area/gpu" --milestone "v0.2.0"

# Issue #364 - Mixed Precision Detection
gh issue edit 364 --add-label "area/gpu" --milestone "v0.2.0"

# Issue #363 - GPU Discovery
gh issue edit 363 --add-label "area/gpu,priority/medium,enhancement" --milestone "v0.2.0"

# Issue #367 - GPU Memory Deallocation
gh issue edit 367 --add-label "area/gpu" --milestone "v0.2.0"

# Issue #366 - GPU Memory Query
gh issue edit 366 --add-label "area/gpu,priority/medium,enhancement" --milestone "v0.2.0"

# Issue #313 - GPU Memory Manager
gh issue edit 313 --add-label "area/gpu,priority/medium,enhancement" --milestone "v0.2.0"

# Issue #322 - Dynamic Device Detection
gh issue edit 322 --add-label "area/gpu,priority/medium,enhancement" --milestone "v0.2.0"

# Issue #303 - Dynamic Workspace Size
gh issue edit 303 --add-label "area/gpu,priority/medium,enhancement" --milestone "v0.2.0"

# Issue #215 - GPU Test Coverage
gh issue edit 215 --add-label "area/gpu,area/testing"

# Issue #270 - Nightly GPU Tests
gh issue edit 270 --add-label "area/gpu,area/testing,priority/low" --milestone "v0.2.0"

# Issue #85 - Infrastructure Stubs
gh issue edit 85 --add-label "area/gpu" --milestone "v0.3.0"
```

---

## Recommended Immediate Actions

### For Orchestrator/Project Manager

1. **Review and Execute Labeling Commands** (above) for proper issue categorization
2. **Prioritize Issue #450** (CUDA Backend MVP) as critical path blocker
3. **Assign Issue #432** (GPU Test Race) for immediate resolution (2-3 hours)
4. **Create Milestone v0.2.0** for post-MVP GPU hardening issues
5. **Update Project Board** with GPU issue swim lane:
   - Critical Path: #450, #432, #455, #317
   - High Value: #414, #293, #394
   - Post-MVP: #374, #364, #363, #367, #366, #313, #322, #303

### For Implementation Team

1. **Issue #450** - CUDA Backend MVP:
   - Implement `Backend` trait abstraction
   - Add CUDA variant with `--features cuda`
   - Integrate NVTX profiling hooks
   - Emit receipts with GPU metadata
   - Wire into `xtask benchmark --backend cuda`

2. **Issue #432** - GPU Test Race:
   - Implement real `synchronize_all()` (replace stub)
   - Add stream barriers before D2H copies
   - Add barriers after batch kernel enqueues
   - Annotate GPU tests with `#[serial]`
   - Verify with `--test-threads=1` and parallel execution

3. **Issue #455** - GPU Receipt Gate:
   - Depends on #450 completion
   - Add GPU receipt verification to CI
   - Implement skip-clean fallback for CPU-only runners
   - Validate GPU kernel presence in receipts

### For Testing/QA

1. **Validate PR #475 Integration**:
   - Confirm EnvGuard prevents environment variable races
   - Verify unified feature gates across workspace
   - Test `gpu_compiled()` and `gpu_available_runtime()` helpers

2. **GPU Test Coverage** (Post-#450):
   - Cross-validation tests (CPU vs GPU inference)
   - Quantization accuracy tests (I2S, TL1, TL2)
   - Performance baseline validation
   - Mixed precision correctness tests

3. **Receipt Validation** (Post-#455):
   - Verify GPU receipts contain CUDA metadata
   - Validate GPU kernel presence
   - Test skip behavior on CPU-only runners

---

## Key Metrics and Success Criteria

### MVP v0.1.0 Success Criteria

- ✅ **GPU Feature Gates Unified** (PR #475 - COMPLETE)
- ⏳ **CUDA Backend MVP Functional** (Issue #450)
  - Token generation on real GPU
  - NVTX profiling support
  - Receipt emission with GPU metadata
- ⏳ **GPU Test Stability** (Issue #432)
  - Zero flaky GPU tests
  - Parallel execution stable
- ⏳ **GPU Receipt Validation** (Issue #455)
  - CI job active on GPU runners
  - Clean skip on CPU-only runners
- ⏳ **Core GPU Inference Working** (Issue #317 via #450)
  - Real forward pass on GPU
  - Non-placeholder results

### Performance Baselines (from #450 spec)

- **CPU Baseline**: 10-20 tok/s (from Issue #261)
- **GPU Target**: 50-100 tok/s (from Issue #261)
- **Receipt Validation**: Accurate backend reporting prevents 5-10x silent fallback

### Quality Gates (from #450 spec)

- **Throughput Gate**: `tokens_per_sec >= MIN_TPS` (configurable)
- **Latency Gate**: `gate_wall_time_ms <= MAX_MS` (configurable)
- **Determinism Gate**: Fixed seed produces identical outputs
- **GPU Kernel Gate**: GPU backend receipts contain ≥1 GPU kernel ID

---

## Conclusion

bitnet-rs GPU infrastructure has made significant progress with PR #475 resolving the critical feature gate consistency issue (#439). The path forward is clear:

1. **Immediate**: Resolve #432 (GPU test race - 2-3 hours)
2. **Critical Path**: Implement #450 (CUDA backend MVP - 2-3 weeks)
3. **Follow-up**: Deploy #455 (GPU receipt gate - 1-2 days after #450)
4. **Validation**: Implement #414 (cross-validation tests)
5. **Production Hardening**: Address stub replacement issues in v0.2.0

The repository now has consistent feature gates, environment isolation (EnvGuard), and device capability helpers—providing a solid foundation for GPU inference implementation. The main blocker is completing the CUDA backend MVP (#450), which unblocks the entire GPU validation and testing pipeline.

**Recommended Next Step**: Prioritize Issue #450 as P0 MVP blocker and assign engineering resources immediately.

---

**Analysis Complete**: 2025-11-11
**Total Issues Analyzed**: 25+ GPU/CUDA-related issues
**Critical Path Identified**: #450 → #455 → Stub Replacement Wave
**PR #475 Impact**: Feature gate unification RESOLVED (#439), unblocks implementation
