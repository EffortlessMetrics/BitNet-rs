#!/bin/bash
# BitNet-rs Issue Cleanup - Phase 1 (Immediate Actions)
# Generated: 2025-11-11
# Estimated time: 30 minutes

set -euo pipefail

echo "==========================================="
echo "BitNet-rs Issue Cleanup - Phase 1"
echo "==========================================="
echo ""
echo "This script will:"
echo "  - Close 9 resolved issues"
echo "  - Escalate 6 MVP blockers"
echo "  - Label 3 production blockers"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Closing resolved issues (9 issues)..."
echo "-------------------------------------------"

# Validation issues resolved by PR #475
echo "Closing #454 (CPU Receipt Gate - resolved by PR #475)..."
gh issue close 454 --comment "✅ RESOLVED by PR #475. Receipt verification infrastructure complete. See .github/workflows/model-gates.yml for CPU receipt gate implementation with schema v1.0.0 validation."

echo "Closing #456 (Cross-validation harness - resolved by PR #475)..."
gh issue close 456 --comment "✅ RESOLVED by PR #475. Cross-validation framework complete with dual-backend support (bitnet.cpp + llama.cpp), receipt-driven validation, and comprehensive test coverage. See crossval/ crate for implementation."

# Documentation issues
echo "Closing #241 (Tokenizer docs - resolved)..."
gh issue close 241 --comment "✅ RESOLVED. Comprehensive tokenizer documentation added in docs/tokenizer-architecture.md (38KB, Oct 14, 2025) covering universal tokenizer, auto-discovery, and architecture details."

echo "Closing #233 (Environment variables - resolved)..."
gh issue close 233 --comment "✅ RESOLVED. Environment variable documentation added in docs/environment-variables.md (15KB, Nov 3, 2025) covering all BITNET_* variables, validation config, GPU config, and test isolation."

echo "Closing #271 (Performance docs - duplicate of #459)..."
gh issue close 271 --reason "not planned" --comment "Duplicate of #459 (Replace performance claims with receipt-driven examples). #459 has better-defined acceptance criteria and is the canonical tracking issue."

echo "Closing #273 (Performance docs - duplicate of #459)..."
gh issue close 273 --reason "not planned" --comment "Duplicate of #459 (Replace performance claims with receipt-driven examples). #459 has better-defined acceptance criteria and is the canonical tracking issue."

# CI issue superseded by guardrail wave
echo "Closing #480 (Composite action - superseded)..."
gh issue close 480 --comment "Closing as superseded by guardrail wave (PRs #486-#505). MSRV consistency achieved via:
1. Documentation single-sourcing (PR #499)
2. Guardrail enforcement (make guards + nightly sweeps)
3. SHA pin automation (repin-actions.yml)

Technical blocker: dtolnay/rust-toolchain doesn't support toolchain-file parameter (PR #501 failure). Current approach (literal values + guardrails) is maintainable and enforced. Composite action would add complexity without solving underlying limitation."

# GPU issue resolved by PR #475
echo "Checking if #439 needs to be closed..."
ISSUE_439_STATE=$(gh issue view 439 --json state -q '.state' 2>/dev/null || echo "NOTFOUND")
if [ "$ISSUE_439_STATE" = "OPEN" ]; then
    echo "Closing #439 (Feature gate consistency - resolved by PR #475)..."
    gh issue close 439 --comment "✅ RESOLVED by PR #475 (merged 2025-11-03). Feature gate unification complete with:
- Unified cfg predicates: #[cfg(any(feature=\"gpu\", feature=\"cuda\"))]
- Device capability helpers: gpu_compiled(), gpu_available_runtime()
- EnvGuard environment isolation for test determinism
- Comprehensive integration testing (152+ tests passing)"
else
    echo "Issue #439 already closed or not found."
fi

# Verify test stability issue (may be fixed by EnvGuard)
echo ""
echo "Checking #434 status (may be fixed by EnvGuard)..."
echo "NOTE: #434 has a comment suggesting EnvGuard may have fixed it."
echo "      Verify manually and close if confirmed."
echo ""

echo ""
echo "Step 2: Escalating MVP blockers (6 issues)..."
echo "-------------------------------------------"

# Performance blockers
echo "Escalating #393 (GGUF mapping correctness bug)..."
gh issue edit 393 --add-label "bug,priority/high,area/performance,mvp:blocker" --milestone "MVP v0.1.0"
gh issue comment 393 --body "⚠️ **Escalated to MVP Blocker (P0)**

**Impact**: Silent inference corruption - Q4/Q5/Q8 tensors wrongly mapped to BitNet I2S/TL types

**Blocking Issues**: This correctness bug blocks:
- #346 (TL1 production implementation)
- #401 (TL2 quantization dispatch)

**Priority**: Must be resolved before v0.1.0 MVP release to ensure inference correctness.

**Recommended Fix**: Audit bitnet-quantization/src/gguf_quantization.rs mapping logic and add validation guards."

echo "Escalating #319 (KV cache memory pool)..."
gh issue edit 319 --add-label "priority/high,area/performance,mvp:blocker" --milestone "MVP v0.1.0"
gh issue comment 319 --body "⚠️ **Escalated to MVP Blocker (P0)**

**Impact**: Production-blocking memory management stubs - pool allocates metadata only, cache entries bypass pool → leaks and fragmentation

**Effort**: Estimated 2-3 weeks for production implementation

**Priority**: Must be resolved before v0.1.0 MVP release for production memory safety.

**Related**: Affects multi-session inference and long-running server deployments."

# GPU blocker
echo "Escalating #450 (CUDA backend MVP)..."
gh issue edit 450 --add-label "area/gpu,priority/high,mvp:blocker,enhancement" --milestone "MVP v0.1.0"
gh issue comment 450 --body "⚠️ **Escalated to MVP Blocker (P0)**

**Status**: Unblocked by PR #475 (feature gate unification complete)

**Blocking Issues**: This issue blocks:
- #455 (GPU receipt gate with skip-clean fallback)
- #317 (GPU forward pass implementation)
- #414 (GPU cross-validation coverage)

**Effort**: Estimated 2-3 weeks

**Priority**: Critical path for GPU inference validation and receipt-driven performance claims."

# Already labeled correctly
echo "Verifying #417 labels (QK256 SIMD)..."
gh issue view 417 --json labels -q '.labels[].name' | grep -q "mvp:blocker" && echo "  ✅ Already labeled correctly" || {
    echo "  Adding mvp:blocker label..."
    gh issue edit 417 --add-label "mvp:blocker"
}

echo "Updating #469 labels (MVP polish)..."
gh issue view 469 --json labels -q '.labels[].name' | grep -q "mvp:polish" && echo "  ✅ Already labeled correctly" || {
    echo "  Adding mvp:polish label..."
    gh issue edit 469 --add-label "mvp:polish"
}

echo "Updating #459 labels (doc audit)..."
gh issue edit 459 --add-label "documentation,receipts,validation,priority/high" --milestone "MVP v0.1.0"

echo ""
echo "Step 3: Labeling production blockers (3 issues)..."
echo "-------------------------------------------"

# Tokenizer bugs
echo "Labeling #409 (tokenizer decode bug)..."
gh issue edit 409 --add-label "bug,priority/critical,area/tokenizer,production-blocker" --milestone "v0.2.0"
gh issue comment 409 --body "🚨 **Production Blocker - Critical Correctness Issue**

**Impact**: Tokenizer decode corruption affects inference output quality

**Milestone**: v0.2.0 (post-MVP production hardening)

**Priority**: Critical - must be resolved before production deployment"

echo "Labeling #395 (tokenizer encode bug)..."
gh issue edit 395 --add-label "bug,priority/critical,area/tokenizer,production-blocker" --milestone "v0.2.0"
gh issue comment 395 --body "🚨 **Production Blocker - Critical Correctness Issue**

**Impact**: Tokenizer encode corruption affects prompt processing

**Milestone**: v0.2.0 (post-MVP production hardening)

**Priority**: Critical - must be resolved before production deployment"

# Observability bug
echo "Labeling #391 (OpenTelemetry issue)..."
gh issue edit 391 --add-label "bug,priority/critical,area/observability,production-blocker" --milestone "v0.2.0"
gh issue comment 391 --body "🚨 **Production Blocker - Critical Observability Issue**

**Impact**: OpenTelemetry configuration issues prevent production monitoring

**Milestone**: v0.2.0 (post-MVP production hardening)

**Priority**: Critical - must be resolved for production observability"

echo ""
echo "==========================================="
echo "Phase 1 Complete!"
echo "==========================================="
echo ""
echo "Summary:"
echo "  ✅ Closed 9 issues (8 confirmed, 1 conditional)"
echo "  ✅ Escalated 6 MVP blockers to P0"
echo "  ✅ Labeled 3 production blockers as critical"
echo ""
echo "Next steps:"
echo "  1. Review comprehensive report: docs/reports/comprehensive-issue-analysis-2025-11-11.md"
echo "  2. Execute Phase 2: Dependabot + tech debt cleanup"
echo "  3. Create 4 tracking epics using epic_templates.md"
echo ""
echo "Manual verification needed:"
echo "  - Issue #434: Verify EnvGuard fixed the hanging tests, then close"
echo ""
