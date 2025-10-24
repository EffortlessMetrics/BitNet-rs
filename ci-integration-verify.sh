#!/bin/bash
# CI Integration Verification Script
# Runs all pre-flight and post-integration checks
# Usage: ./ci-integration-verify.sh [pre|post]

set -euo pipefail

MODE="${1:-pre}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "CI Integration Verification Script"
echo "Mode: $MODE"
echo "========================================="
echo ""

# Pre-integration checks
if [[ "$MODE" == "pre" ]]; then
  echo "=== PRE-INTEGRATION CHECKS ==="
  echo ""

  # Check 1: Verify branch
  echo -n "1. Checking branch... "
  BRANCH=$(git branch --show-current)
  if [[ "$BRANCH" == "feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2" ]]; then
    echo -e "${GREEN}✅ Correct branch${NC}"
  else
    echo -e "${YELLOW}⚠️  On branch: $BRANCH (expected: feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2)${NC}"
  fi

  # Check 2: Verify CI workflow exists
  echo -n "2. Checking CI workflow... "
  if [[ -f ".github/workflows/ci.yml" ]]; then
    LINES=$(wc -l < .github/workflows/ci.yml)
    echo -e "${GREEN}✅ Exists (${LINES} lines)${NC}"
  else
    echo -e "${RED}❌ NOT FOUND${NC}"
    exit 1
  fi

  # Check 3: Verify YAML fragments exist
  echo -n "3. Checking YAML fragments... "
  FRAG_COUNT=$(ls ci/yaml-fragments/*.yml 2>/dev/null | grep -v README | wc -l)
  if [[ "$FRAG_COUNT" -eq 7 ]]; then
    echo -e "${GREEN}✅ All 7 fragments present${NC}"
  else
    echo -e "${RED}❌ Found $FRAG_COUNT fragments (expected 7)${NC}"
    exit 1
  fi

  # Check 4: Verify guard scripts exist
  echo -n "4. Checking guard scripts... "
  GUARD_SCRIPTS=(
    "scripts/check-ignore-annotations.sh"
    "scripts/check-serial-annotations.sh"
    "scripts/check-feature-gates.sh"
    "scripts/validate-fixtures.sh"
  )
  MISSING=0
  for script in "${GUARD_SCRIPTS[@]}"; do
    if [[ ! -x "$script" ]]; then
      echo -e "${RED}❌ Missing or not executable: $script${NC}"
      MISSING=$((MISSING + 1))
    fi
  done
  if [[ $MISSING -eq 0 ]]; then
    echo -e "${GREEN}✅ All 4 guard scripts present and executable${NC}"
  else
    exit 1
  fi

  # Check 5: Count current jobs
  echo -n "5. Checking current job count... "
  JOB_COUNT=$(grep "^  [a-z-]*:" .github/workflows/ci.yml | wc -l)
  echo -e "${GREEN}✅ Current: $JOB_COUNT entries${NC}"

  echo ""
  echo "=== TESTING GUARD SCRIPTS ==="
  echo ""

  # Test guard-ignore-annotations
  echo "6. Testing guard-ignore-annotations..."
  if bash scripts/check-ignore-annotations.sh > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ PASS${NC}"
  else
    echo -e "   ${RED}❌ FAIL - Fix violations before integration${NC}"
    bash scripts/check-ignore-annotations.sh
    exit 1
  fi

  # Test guard-serial-annotations
  echo "7. Testing guard-serial-annotations..."
  if bash scripts/check-serial-annotations.sh > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ PASS${NC}"
  else
    echo -e "   ${RED}❌ FAIL - Fix violations before integration${NC}"
    bash scripts/check-serial-annotations.sh
    exit 1
  fi

  # Test guard-feature-consistency
  echo "8. Testing guard-feature-consistency..."
  if bash scripts/check-feature-gates.sh > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ PASS${NC}"
  else
    echo -e "   ${YELLOW}⚠️  WARNINGS (non-blocking)${NC}"
  fi

  # Test guard-fixture-integrity
  echo "9. Testing guard-fixture-integrity..."
  if bash scripts/validate-fixtures.sh > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ PASS${NC}"
  else
    echo -e "   ${RED}❌ FAIL - Fix fixture issues before integration${NC}"
    bash scripts/validate-fixtures.sh
    exit 1
  fi

  echo ""
  echo "=== VALIDATING YAML FRAGMENTS ==="
  echo ""

  # Validate YAML syntax of fragments
  echo "10. Validating YAML fragment syntax..."
  YAML_ERRORS=0
  for fragment in ci/yaml-fragments/*.yml; do
    [[ "$fragment" == *"README"* ]] && continue
    if python3 -c "import yaml; yaml.safe_load(open('$fragment'))" 2>/dev/null; then
      echo -e "    ${GREEN}✅${NC} $(basename $fragment)"
    else
      echo -e "    ${RED}❌${NC} $(basename $fragment)"
      YAML_ERRORS=$((YAML_ERRORS + 1))
    fi
  done

  if [[ $YAML_ERRORS -eq 0 ]]; then
    echo -e "   ${GREEN}✅ All fragments valid${NC}"
  else
    echo -e "   ${RED}❌ $YAML_ERRORS fragments have syntax errors${NC}"
    exit 1
  fi

  echo ""
  echo "========================================="
  echo -e "${GREEN}✅ PRE-INTEGRATION CHECKS PASSED${NC}"
  echo "Ready to proceed with integration."
  echo "========================================="

# Post-integration checks
elif [[ "$MODE" == "post" ]]; then
  echo "=== POST-INTEGRATION CHECKS ==="
  echo ""

  # Check 1: Validate YAML syntax
  echo -n "1. Validating YAML syntax... "
  if python3 -m yaml .github/workflows/ci.yml > /dev/null 2>&1; then
    echo -e "${GREEN}✅ VALID${NC}"
  else
    echo -e "${RED}❌ INVALID - Restore from backup immediately${NC}"
    exit 1
  fi

  # Check 2: Count lines
  echo -n "2. Checking line count... "
  LINES=$(wc -l < .github/workflows/ci.yml)
  EXPECTED_MIN=1000
  EXPECTED_MAX=1150
  if [[ $LINES -ge $EXPECTED_MIN && $LINES -le $EXPECTED_MAX ]]; then
    echo -e "${GREEN}✅ $LINES lines (expected ~1077)${NC}"
  else
    echo -e "${YELLOW}⚠️  $LINES lines (expected ~1077) - verify integration${NC}"
  fi

  # Check 3: Count jobs
  echo -n "3. Checking job count... "
  JOB_COUNT=$(grep "^  [a-z-]*:" .github/workflows/ci.yml | wc -l)
  EXPECTED_JOBS=22  # 15 before (including non-jobs like push/schedule) + 7 new
  if [[ $JOB_COUNT -eq $EXPECTED_JOBS ]]; then
    echo -e "${GREEN}✅ $JOB_COUNT entries (expected $EXPECTED_JOBS)${NC}"
  else
    echo -e "${YELLOW}⚠️  $JOB_COUNT entries (expected $EXPECTED_JOBS)${NC}"
  fi

  # Check 4: Verify new jobs exist
  echo "4. Verifying new jobs exist..."
  NEW_JOBS=(
    "feature-hack-check"
    "feature-matrix"
    "doctest-matrix"
    "guard-ignore-annotations"
    "guard-fixture-integrity"
    "guard-serial-annotations"
    "guard-feature-consistency"
  )
  MISSING_JOBS=0
  for job in "${NEW_JOBS[@]}"; do
    if grep -q "^  ${job}:" .github/workflows/ci.yml; then
      echo -e "   ${GREEN}✅${NC} $job"
    else
      echo -e "   ${RED}❌${NC} $job (MISSING)"
      MISSING_JOBS=$((MISSING_JOBS + 1))
    fi
  done

  if [[ $MISSING_JOBS -eq 0 ]]; then
    echo -e "   ${GREEN}✅ All 7 new jobs found${NC}"
  else
    echo -e "   ${RED}❌ $MISSING_JOBS jobs missing${NC}"
    exit 1
  fi

  # Check 5: Check for duplicates
  echo -n "5. Checking for duplicate job names... "
  DUPLICATES=$(grep "^  [a-z-]*:" .github/workflows/ci.yml | sort | uniq -d | wc -l)
  if [[ $DUPLICATES -eq 0 ]]; then
    echo -e "${GREEN}✅ No duplicates${NC}"
  else
    echo -e "${RED}❌ Found $DUPLICATES duplicate job names${NC}"
    grep "^  [a-z-]*:" .github/workflows/ci.yml | sort | uniq -d
    exit 1
  fi

  # Check 6: Verify dependencies
  echo "6. Verifying job dependencies..."
  python3 << 'PYTHON'
import yaml
import sys

with open('.github/workflows/ci.yml', 'r') as f:
    workflow = yaml.safe_load(f)

jobs = workflow.get('jobs', {})

expected_deps = {
    'feature-hack-check': 'test',
    'feature-matrix': 'test',
    'doctest-matrix': 'test',
    'guard-ignore-annotations': None,
    'guard-fixture-integrity': None,
    'guard-serial-annotations': None,
    'guard-feature-consistency': None,
}

errors = 0
for job_name, expected_dep in expected_deps.items():
    if job_name not in jobs:
        print(f"   \033[0;31m❌\033[0m {job_name}: NOT FOUND")
        errors += 1
        continue

    actual_deps = jobs[job_name].get('needs', None)

    if expected_dep is None:
        if actual_deps is None:
            print(f"   \033[0;32m✅\033[0m {job_name}: no dependencies (correct)")
        else:
            print(f"   \033[1;33m⚠️\033[0m {job_name}: has dependencies {actual_deps} (expected none)")
            errors += 1
    else:
        if actual_deps == expected_dep:
            print(f"   \033[0;32m✅\033[0m {job_name}: depends on {expected_dep} (correct)")
        else:
            print(f"   \033[0;31m❌\033[0m {job_name}: depends on {actual_deps} (expected {expected_dep})")
            errors += 1

sys.exit(errors)
PYTHON

  if [[ $? -eq 0 ]]; then
    echo -e "   ${GREEN}✅ All dependencies correct${NC}"
  else
    echo -e "   ${RED}❌ Dependency errors found${NC}"
    exit 1
  fi

  # Check 7: Review diff
  echo "7. Reviewing git diff..."
  if git diff --quiet .github/workflows/ci.yml; then
    echo -e "   ${YELLOW}⚠️  No changes detected (did integration complete?)${NC}"
  else
    INSERTIONS=$(git diff --stat .github/workflows/ci.yml | grep -oP '\d+(?= insertion)')
    DELETIONS=$(git diff --stat .github/workflows/ci.yml | grep -oP '\d+(?= deletion)' || echo "0")
    echo -e "   ${GREEN}✅${NC} +$INSERTIONS insertions, -$DELETIONS deletions"
    if [[ "$DELETIONS" != "0" ]]; then
      echo -e "   ${YELLOW}⚠️  Unexpected deletions detected - review carefully${NC}"
    fi
  fi

  echo ""
  echo "========================================="
  echo -e "${GREEN}✅ POST-INTEGRATION CHECKS PASSED${NC}"
  echo "Integration appears successful."
  echo "Next: Commit, push, and monitor CI."
  echo "========================================="

else
  echo "Invalid mode: $MODE"
  echo "Usage: $0 [pre|post]"
  exit 1
fi
