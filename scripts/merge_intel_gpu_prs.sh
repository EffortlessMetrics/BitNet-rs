#!/bin/bash
# Merge Intel GPU PRs in dependency order
# Usage: ./scripts/merge_intel_gpu_prs.sh [--dry-run]

set -euo pipefail

DRY_RUN="${DRY_RUN:-false}"
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="true"
fi

# Dependency-ordered PR list
WAVE1_FOUNDATION=(1032 1033 1034)
WAVE1_INFRA=(1035 1036)
WAVE1_CORE=(1037 1038 1039)
WAVE1_TESTS=(1040 1041 1042)
WAVE2=(1043 1044 1045 1046 1047 1050 1051 1052)
WAVE3=(1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064)

merge_pr() {
    local pr=$1
    echo "Merging PR #$pr..."
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [DRY RUN] gh pr merge $pr --merge"
    else
        gh pr merge "$pr" --merge --auto || echo "  WARN: PR #$pr merge failed"
    fi
}

echo "=== Intel GPU PR Merge Queue ==="
echo "Dry run: $DRY_RUN"
echo ""

echo "--- Wave 1: Foundation ---"
for pr in "${WAVE1_FOUNDATION[@]}"; do merge_pr "$pr"; done

echo "--- Wave 1: Infrastructure ---"
for pr in "${WAVE1_INFRA[@]}"; do merge_pr "$pr"; done

echo "--- Wave 1: Core ---"
for pr in "${WAVE1_CORE[@]}"; do merge_pr "$pr"; done

echo "--- Wave 1: Tests ---"
for pr in "${WAVE1_TESTS[@]}"; do merge_pr "$pr"; done

echo "--- Wave 2 ---"
for pr in "${WAVE2[@]}"; do merge_pr "$pr"; done

echo "--- Wave 3 ---"
for pr in "${WAVE3[@]}"; do merge_pr "$pr"; done

echo ""
echo "=== Done ==="
