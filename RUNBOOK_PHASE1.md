# Phase 1 Detailed Runbook: Safety Patches

**Objective:** Replace unsafe `Factory-AI/droid-action` refs with safe action SHA; disable raw artifact upload  
**Scope:** 7 repos, 1 PR each  
**Sequence:** OpenRacing → adze → SwiftMTP-dev → SwiftMailSort → shiplog → perl-lsp → pkm-python  
**Time per repo:** 1-2 days  
**Total Phase 1:** ~2 weeks

---

## Phase 1 Repo Template

Execute this procedure for **each repo in sequence**. Copy and paste the repo name as you go.

### Repo: [REPO_NAME]

**Estimated time:** 4-6 hours (excluding code review wait time)  
**Date started:** ___________  
**Date completed:** ___________

---

## Step 1: Access Verification & Clone

```bash
#!/bin/bash
set -e  # Exit on error

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo "=========================================="
echo "Phase 1: $REPO"
echo "=========================================="
echo ""
echo "Step 1: Access Verification & Clone"
echo ""

# Verify access
echo "✓ Verifying access to EffortlessMetrics/$REPO..."
if ! gh repo view EffortlessMetrics/$REPO &>/dev/null; then
  echo "❌ FAILED: No access to EffortlessMetrics/$REPO"
  exit 1
fi
echo "✅ Access confirmed"

# Clone
echo "✓ Cloning repository..."
mkdir -p /tmp/droid-migration
cd /tmp/droid-migration
rm -rf $REPO  # Clean if exists
gh repo clone EffortlessMetrics/$REPO
cd $REPO

echo "✓ Verifying clone..."
pwd
git status
echo "✅ Clone successful"
```

**Expected output:**
```
========================================
Phase 1: OpenRacing
==========================================

Step 1: Access Verification & Clone

✓ Verifying access to EffortlessMetrics/OpenRacing...
✅ Access confirmed
✓ Cloning repository...
✓ Verifying clone...
/tmp/droid-migration/OpenRacing
On branch main
nothing to commit, working tree clean
✅ Clone successful
```

---

## Step 2: Examine Current Workflows

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 2: Examine Current Workflows"
echo ""

# Verify workflow files exist
echo "✓ Checking for droid workflow files..."
if [ ! -f .github/workflows/droid.yml ]; then
  echo "❌ FAILED: .github/workflows/droid.yml not found"
  exit 1
fi
if [ ! -f .github/workflows/droid-review.yml ]; then
  echo "❌ FAILED: .github/workflows/droid-review.yml not found"
  exit 1
fi
echo "✅ Both workflow files exist"

# Find Factory-AI action refs
echo ""
echo "✓ Searching for Factory-AI/droid-action refs..."
echo ""

echo "In droid-review.yml:"
grep -n "Factory-AI/droid-action" .github/workflows/droid-review.yml || echo "  (not found)"

echo ""
echo "In droid.yml:"
grep -n "Factory-AI/droid-action" .github/workflows/droid.yml || echo "  (not found)"

echo ""
echo "✅ Current refs identified"
echo ""
echo "NOTE: Record line numbers above for editing in Step 3"
```

**Expected output:**
```
Step 2: Examine Current Workflows

✓ Checking for droid workflow files...
✅ Both workflow files exist

✓ Searching for Factory-AI/droid-action refs...

In droid-review.yml:
     12:      uses: Factory-AI/droid-action@main

In droid.yml:
     15:      uses: Factory-AI/droid-action@main

✅ Current refs identified

NOTE: Record line numbers above for editing in Step 3
```

---

## Step 3: Create Phase 1 Branch

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 3: Create Phase 1 Branch"
echo ""

BRANCH="ci/safe-droid-action"

echo "✓ Creating branch: $BRANCH..."
git checkout -b $BRANCH

echo "✓ Verifying branch..."
git branch
echo ""
echo "✅ Branch created and checked out"
```

**Expected output:**
```
Step 3: Create Phase 1 Branch

✓ Creating branch: ci/safe-droid-action...
✓ Verifying branch...
* ci/safe-droid-action
  main

✅ Branch created and checked out
```

---

## Step 4: Edit droid-review.yml

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 4: Edit droid-review.yml"
echo ""

FILE=".github/workflows/droid-review.yml"

echo "✓ Creating backup: ${FILE}.bak..."
cp $FILE ${FILE}.bak

echo "✓ Finding Factory-AI/droid-action line..."
LINE_NUM=$(grep -n "Factory-AI/droid-action" $FILE | head -1 | cut -d: -f1)
echo "  Line: $LINE_NUM"

echo "✓ Replacing Factory-AI/droid-action with safe action..."
# Handle both @main and @v5 and @<sha>
sed -i "s|Factory-AI/droid-action@[^#]*|EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5; raw debug artifact upload disabled|g" $FILE

echo "✓ Verifying replacement..."
grep -A1 -B1 "droid-action-safe" $FILE | head -5

echo ""
echo "✓ Checking for upload_debug_artifacts flag..."
if grep -q "upload_debug_artifacts: false" $FILE; then
  echo "  Already present ✅"
else
  echo "  Not found, adding..."
  # Find the factory_api_key line and add after it
  FACTORY_LINE=$(grep -n "factory_api_key" $FILE | head -1 | cut -d: -f1)
  sed -i "${FACTORY_LINE}a\\        upload_debug_artifacts: false" $FILE
  echo "  Added ✅"
fi

echo ""
echo "✓ Validating YAML syntax..."
python3 -c "import yaml; yaml.safe_load(open('$FILE'))" && echo "  ✅ Valid YAML" || (echo "  ❌ Invalid YAML"; exit 1)

echo ""
echo "✅ droid-review.yml successfully patched"
```

**Expected output:**
```
Step 4: Edit droid-review.yml

✓ Creating backup: .github/workflows/droid-review.yml.bak...
✓ Finding Factory-AI/droid-action line...
  Line: 12
✓ Replacing Factory-AI/droid-action with safe action...
✓ Verifying replacement...
       uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5; raw debug artifact upload disabled

✓ Checking for upload_debug_artifacts flag...
  Not found, adding...
  Added ✅

✓ Validating YAML syntax...
  ✅ Valid YAML

✅ droid-review.yml successfully patched
```

---

## Step 5: Edit droid.yml

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 5: Edit droid.yml"
echo ""

FILE=".github/workflows/droid.yml"

echo "✓ Creating backup: ${FILE}.bak..."
cp $FILE ${FILE}.bak

echo "✓ Finding Factory-AI/droid-action line..."
LINE_NUM=$(grep -n "Factory-AI/droid-action" $FILE | head -1 | cut -d: -f1)
echo "  Line: $LINE_NUM"

echo "✓ Replacing Factory-AI/droid-action with safe action..."
sed -i "s|Factory-AI/droid-action@[^#]*|EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5; raw debug artifact upload disabled|g" $FILE

echo "✓ Verifying replacement..."
grep -A1 -B1 "droid-action-safe" $FILE | head -5

echo ""
echo "✓ Checking for upload_debug_artifacts flag..."
if grep -q "upload_debug_artifacts: false" $FILE; then
  echo "  Already present ✅"
else
  echo "  Not found, adding..."
  FACTORY_LINE=$(grep -n "factory_api_key" $FILE | head -1 | cut -d: -f1)
  sed -i "${FACTORY_LINE}a\\        upload_debug_artifacts: false" $FILE
  echo "  Added ✅"
fi

echo ""
echo "✓ Validating YAML syntax..."
python3 -c "import yaml; yaml.safe_load(open('$FILE'))" && echo "  ✅ Valid YAML" || (echo "  ❌ Invalid YAML"; exit 1)

echo ""
echo "✅ droid.yml successfully patched"
```

**Expected output:** Same as Step 4, but for droid.yml

---

## Step 6: Review Changes

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 6: Review Changes"
echo ""

echo "✓ Git diff (staged)..."
git diff .github/workflows/droid*.yml

echo ""
echo "✓ Verify both files are modified..."
git status

echo ""
echo "Items to verify in the diff:"
echo "  ☐ Both droid.yml and droid-review.yml are modified"
echo "  ☐ Factory-AI/droid-action replaced with safe action SHA"
echo "  ☐ upload_debug_artifacts: false added"
echo "  ☐ No other changes present"
echo ""
echo "✅ Changes reviewed"
```

**Expected output:**
```
Step 6: Review Changes

✓ Git diff (staged)...
diff --git a/.github/workflows/droid-review.yml b/.github/workflows/droid-review.yml
index abc...def
--- a/.github/workflows/droid-review.yml
+++ b/.github/workflows/droid-review.yml
@@ -10,10 +10,11 @@ jobs:
     steps:
       - name: Run Droid Auto Review
-        uses: Factory-AI/droid-action@main
+        uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5
         with:
           factory_api_key: ${{ secrets.FACTORY_API_KEY }}
+          upload_debug_artifacts: false

Items to verify:
  ☐ Both droid.yml and droid-review.yml are modified
  ☐ Factory-AI/droid-action replaced with safe action SHA
  ☐ upload_debug_artifacts: false added
  ☐ No other changes present

✅ Changes reviewed
```

---

## Step 7: Commit Changes

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 7: Commit Changes"
echo ""

echo "✓ Staging files..."
git add .github/workflows/droid*.yml

echo "✓ Verifying staged files..."
git diff --cached --name-only

echo ""
echo "✓ Creating commit..."
git commit -m "ci: use safe Droid action

- Replace Factory-AI/droid-action with EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f
- Add upload_debug_artifacts: false
- Preserve existing Droid behavior except for disabling raw debug artifact upload

Refs: MIGRATION_PLAN.md, IMPLEMENTATION_OPENRACING.md"

echo ""
echo "✓ Verifying commit..."
git log --oneline -2

echo ""
echo "✅ Commit created successfully"
```

**Expected output:**
```
Step 7: Commit Changes

✓ Staging files...
✓ Verifying staged files...
.github/workflows/droid-review.yml
.github/workflows/droid.yml

✓ Creating commit...
[ci/safe-droid-action a1b2c3d] ci: use safe Droid action
 2 files changed, 5 insertions(+), 3 deletions(-)

✓ Verifying commit...
a1b2c3d ci: use safe Droid action
e4f5g6h Previous commit

✅ Commit created successfully
```

---

## Step 8: Push to Remote

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 8: Push to Remote"
echo ""

BRANCH="ci/safe-droid-action"

echo "✓ Pushing branch: $BRANCH..."
git push -u origin $BRANCH

echo ""
echo "✅ Branch pushed successfully"
echo ""
echo "Next step: Go to GitHub to create a PR or use Step 9 command"
```

**Expected output:**
```
Step 8: Push to Remote

✓ Pushing branch: ci/safe-droid-action...
remote: Create a pull request for 'ci/safe-droid-action' on GitHub by visiting:
remote:      https://github.com/EffortlessMetrics/OpenRacing/pull/new/ci/safe-droid-action

✅ Branch pushed successfully

Next step: Go to GitHub to create a PR or use Step 9 command
```

---

## Step 9: Create Pull Request

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration/$REPO

echo ""
echo "Step 9: Create Pull Request"
echo ""

BRANCH="ci/safe-droid-action"
TITLE="ci: use safe Droid action"

echo "✓ Creating PR..."
PR_URL=$(gh pr create \
  --repo EffortlessMetrics/$REPO \
  --base main \
  --head $BRANCH \
  --title "$TITLE" \
  --body "## Summary

- Switch Droid workflows to \`EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f\`.
- Add \`upload_debug_artifacts: false\`.
- Preserve existing Droid behavior except for disabling raw debug artifact upload.

## Why

The upstream Factory action can upload raw \`\$HOME/.factory/**\` and \`droid-prompts/**\`. In BYOK mode that can include resolved provider credentials. Normal Droid runs should not upload raw debug artifacts.

## Validation

- [x] Repo workflow/static checks pass.
- [ ] Same-repo PR smoke run succeeds.
- [ ] No raw artifact named \`droid-review-debug-<run_id>\` is uploaded.

## Non-goals

- No permission reduction.
- No model/provider change except MiniMax BYOK convergence if already intended.
- No \`review_depth: deep\`.
- No \`pull_request_target\`.

Refs: MIGRATION_PLAN.md, IMPLEMENTATION_OPENRACING.md")

echo "✅ PR created successfully"
echo "PR URL: $PR_URL"
echo ""
echo "Next steps:"
echo "  1. Visit $PR_URL"
echo "  2. Wait for checks to pass"
echo "  3. Request review from code reviewer"
echo "  4. After approval, merge the PR"
```

**Expected output:**
```
Step 9: Create Pull Request

✓ Creating PR...
✅ PR created successfully
PR URL: https://github.com/EffortlessMetrics/OpenRacing/pull/123

Next steps:
  1. Visit https://github.com/EffortlessMetrics/OpenRacing/pull/123
  2. Wait for checks to pass
  3. Request review from code reviewer
  4. After approval, merge the PR
```

---

## Step 10: Code Review & Approval

**Duration:** 1-24 hours  
**Action:** Manual (outside of this script)

### Checklist for Code Reviewer

- [ ] PR title is "ci: use safe Droid action"
- [ ] Safe action SHA is exactly: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- [ ] `upload_debug_artifacts: false` is present in both workflows
- [ ] Checkout action pinned to SHA (if present)
- [ ] No unrelated changes in the PR
- [ ] Workflow YAML is syntactically correct
- [ ] Both droid.yml and droid-review.yml are modified
- [ ] Approve the PR

---

## Step 11: Merge PR

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo ""
echo "Step 11: Merge PR"
echo ""

echo "✓ Finding latest PR on ci/safe-droid-action..."
PR_NUMBER=$(gh pr list --repo EffortlessMetrics/$REPO \
  --head ci/safe-droid-action \
  --state open \
  --json number \
  --jq '.[0].number')

if [ -z "$PR_NUMBER" ]; then
  echo "❌ FAILED: No open PR found on ci/safe-droid-action"
  exit 1
fi

echo "  PR #$PR_NUMBER"

echo ""
echo "✓ Checking PR status..."
gh pr view $PR_NUMBER --repo EffortlessMetrics/$REPO

echo ""
echo "✓ Merging PR #$PR_NUMBER..."
gh pr merge $PR_NUMBER \
  --repo EffortlessMetrics/$REPO \
  --merge \
  --delete-branch

echo ""
echo "✅ PR merged successfully"
```

**Expected output:**
```
Step 11: Merge PR

✓ Finding latest PR on ci/safe-droid-action...
  PR #123

✓ Checking PR status...
[PR details shown]

✓ Merging PR #123...
✓ Pull request #123 merged successfully
✓ Deleted branch ci/safe-droid-action

✅ PR merged successfully
```

---

## Step 12: Smoke Test Phase 1

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo ""
echo "Step 12: Smoke Test Phase 1"
echo ""

echo "⏳ Waiting 30 seconds for merged main to be available..."
sleep 30

# Create test branch
cd /tmp/droid-migration/$REPO
git fetch origin
git checkout main
git pull origin main

git checkout -b smoke-test-phase1
echo "# Smoke Test Phase 1" >> README.md
git add README.md
git commit -m "smoke: test phase 1 safe action deployment"
git push -u origin smoke-test-phase1

echo ""
echo "✓ Creating draft PR for smoke test..."
gh pr create \
  --repo EffortlessMetrics/$REPO \
  --base main \
  --head smoke-test-phase1 \
  --draft \
  --title "[smoke-test] Droid phase 1 safe action" \
  --body "Smoke test for Phase 1 safe action deployment.

Verify:
- Droid Auto Review triggers
- No raw droid-review-debug-<run_id> artifacts
- Workflow completes successfully"

echo ""
echo "⏳ Waiting 3 minutes for Droid workflow to start..."
sleep 180

echo ""
echo "✓ Checking workflow status..."
RUN_ID=$(gh run list \
  --repo EffortlessMetrics/$REPO \
  --head smoke-test-phase1 \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')

if [ -z "$RUN_ID" ]; then
  echo "⚠️  No workflow run found yet. Check GitHub Actions manually."
  echo "   Go to: https://github.com/EffortlessMetrics/$REPO/actions"
else
  echo "  Run ID: $RUN_ID"
  gh run view $RUN_ID --repo EffortlessMetrics/$REPO
fi

echo ""
echo "✓ Checking for artifacts..."
mkdir -p /tmp/artifacts-$REPO
gh run download $RUN_ID --repo EffortlessMetrics/$REPO --dir /tmp/artifacts-$REPO 2>/dev/null || true

echo "  Artifacts found:"
ls -la /tmp/artifacts-$REPO/ | grep -i droid || echo "  (none)"

echo ""
echo "✓ Checking for unsafe artifact..."
if ls /tmp/artifacts-$REPO/*droid-review-debug* 2>/dev/null; then
  echo "❌ FAILED: Raw debug artifact found! (unsafe)"
  exit 1
else
  echo "✅ No raw debug artifacts (SAFE)"
fi

echo ""
echo "✓ Deleting test PR..."
gh pr delete smoke-test-phase1 --repo EffortlessMetrics/$REPO --yes
git push origin --delete smoke-test-phase1

echo ""
echo "✅ Smoke Test Phase 1 PASSED for $REPO"
```

**Expected output:**
```
Step 12: Smoke Test Phase 1

⏳ Waiting 30 seconds for merged main to be available...

✓ Creating draft PR for smoke test...
✓ Pull request #124 created

⏳ Waiting 3 minutes for Droid workflow to start...

✓ Checking workflow status...
  Run ID: 12345678
  ✓ completed [success]

✓ Checking for artifacts...
  Artifacts found:
  (none or summary-* only)

✓ Checking for unsafe artifact...
✅ No raw debug artifacts (SAFE)

✓ Deleting test PR...
✓ Deleted branch smoke-test-phase1

✅ Smoke Test Phase 1 PASSED for OpenRacing
```

---

## Step 13: Mark Completion & Move to Next Repo

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo ""
echo "Step 13: Mark Completion"
echo ""

echo "✅ $REPO Phase 1 COMPLETE"
echo ""
echo "Update TRACKING_SHEET.md:"
echo "  - Mark $REPO Phase 1 ✅ COMPLETE"
echo "  - Record PR number and merge date"
echo ""
echo "Next repo: adze"
```

---

## Full Phase 1 One-Liner (For Experienced Users)

```bash
#!/bin/bash

for REPO in OpenRacing adze SwiftMTP-dev SwiftMailSort shiplog perl-lsp pkm-python; do
  echo "Processing $REPO..."
  cd /tmp/droid-migration
  gh repo clone EffortlessMetrics/$REPO
  cd $REPO
  
  # Create branch
  git checkout -b ci/safe-droid-action
  
  # Patch both workflows
  for FILE in .github/workflows/droid*.yml; do
    sed -i "s|Factory-AI/droid-action@[^#]*|EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f|g" $FILE
    FACTORY_LINE=$(grep -n "factory_api_key" $FILE | head -1 | cut -d: -f1)
    sed -i "${FACTORY_LINE}a\\        upload_debug_artifacts: false" $FILE
  done
  
  # Commit and push
  git add .github/workflows/
  git commit -m "ci: use safe Droid action"
  git push -u origin ci/safe-droid-action
  
  # Create PR
  gh pr create \
    --repo EffortlessMetrics/$REPO \
    --base main \
    --head ci/safe-droid-action \
    --title "ci: use safe Droid action" \
    --body "Phase 1 safety patch - use safe Droid action, disable artifacts"
  
  echo "✅ $REPO Phase 1 PR created"
  echo ""
done
```

---

## Phase 1 Complete

When all 7 repos have:
- ✅ Phase 1 PR merged
- ✅ Smoke test passed
- ✅ No unsafe artifacts
- ✅ Marked complete in TRACKING_SHEET.md

Move to **Phase 2: Baseline Convergence** (RUNBOOK_PHASE2.md)
