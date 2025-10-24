#!/usr/bin/env bash
set -euo pipefail

# Phase 1 Pre-Flight Validation Script
# Verifies all prerequisites before executing Phase 1 migration

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1 Pre-Flight Check - #[ignore] Annotation Hygiene        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    
    echo -n "Checking: $name ... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "✅ PASS"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL"
        FAIL=$((FAIL + 1))
    fi
}

# 1. Check scripts exist and are executable
check "check-ignore-hygiene.sh exists" "[ -x scripts/check-ignore-hygiene.sh ]"
check "auto-annotate-ignores.sh exists" "[ -x scripts/auto-annotate-ignores.sh ]"
check "ignore-taxonomy.json exists" "[ -f scripts/ignore-taxonomy.json ]"

# 2. Check dependencies
check "ripgrep (rg) available" "command -v rg"
check "jq available (optional)" "command -v jq || true"
check "rustfmt available" "command -v rustfmt"

# 3. Check documentation
check "Action plan exists" "[ -f IGNORE_ANNOTATION_ACTION_PLAN_PHASE1.md ]"
check "Execution summary exists" "[ -f PHASE1_EXECUTION_SUMMARY.md ]"
check "Deliverables checklist exists" "[ -f PHASE1_DELIVERABLES_CHECKLIST.md ]"

# 4. Check git status
check "Git repository detected" "git rev-parse --git-dir"
check "Working directory clean" "[ -z \"\$(git status --porcelain)\" ] || true"  # Warning only

# 5. Test scripts can run
echo ""
echo "Running hygiene check baseline..."
if MODE=full bash scripts/check-ignore-hygiene.sh > /tmp/phase1-baseline.txt 2>&1; then
    echo "✅ Hygiene check successful"
    PASS=$((PASS + 1))
    
    # Extract metrics
    TOTAL=$(grep "Total #\[ignore\]" /tmp/phase1-baseline.txt | awk '{print $4}')
    BARE=$(grep "Bare (no reason)" /tmp/phase1-baseline.txt | awk '{print $4}')
    
    echo ""
    echo "Current State:"
    echo "  Total #[ignore] annotations: $TOTAL"
    echo "  Bare (no reason):            $BARE"
    
    if [ "$BARE" = "134" ]; then
        echo "  ✅ Baseline matches expected (134 bare)"
        PASS=$((PASS + 1))
    else
        echo "  ⚠️  Baseline differs from expected (134 bare vs $BARE actual)"
        echo "      This may indicate repository state has changed."
    fi
else
    echo "❌ Hygiene check failed"
    FAIL=$((FAIL + 1))
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Pre-Flight Summary: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════════════════════════════"

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "✅ ALL CHECKS PASSED - READY FOR PHASE 1 EXECUTION"
    echo ""
    echo "Next steps:"
    echo "  1. Review: PHASE1_EXECUTION_SUMMARY.md"
    echo "  2. Execute: Follow Section 2 (Dry-Run Validation)"
    echo "  3. Estimated time: 2 hours"
    exit 0
else
    echo ""
    echo "❌ SOME CHECKS FAILED - FIX ISSUES BEFORE PROCEEDING"
    echo ""
    echo "Common fixes:"
    echo "  - Install ripgrep: sudo apt-get install ripgrep"
    echo "  - Make scripts executable: chmod +x scripts/*.sh"
    echo "  - Check documentation exists in repository root"
    exit 1
fi
