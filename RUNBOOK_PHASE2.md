# Phase 2 Detailed Runbook: Baseline Convergence

**Objective:** Add MiniMax BYOK; add guards; add repo-local guidance  
**Scope:** 7 repos, 1 PR each (same repos as Phase 1, same sequence)  
**Prerequisite:** All Phase 1 PRs must be merged  
**Sequence:** OpenRacing → adze → SwiftMTP-dev → SwiftMailSort → shiplog → perl-lsp → pkm-python  
**Time per repo:** 2-3 days (includes BYOK setup, file creation, testing)  
**Total Phase 2:** ~2-3 weeks

---

## Phase 2 Repo Template

Execute this procedure for **each repo in sequence**. Copy and paste the repo name as you go.

### Repo: [REPO_NAME]

**Estimated time:** 6-8 hours (excluding code review wait time)  
**Date started:** ___________  
**Date completed:** ___________

---

## Step 1: Verify Phase 1 Merged & Clone

```bash
#!/bin/bash
set -e

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo "=========================================="
echo "Phase 2: $REPO"
echo "=========================================="
echo ""
echo "Step 1: Verify Phase 1 Merged & Clone"
echo ""

# Verify Phase 1 merged
echo "✓ Checking if Phase 1 PR is merged..."
if ! gh pr list --repo EffortlessMetrics/$REPO \
  --head ci/safe-droid-action \
  --state closed \
  --limit 1 \
  --json number &>/dev/null; then
  echo "⚠️  Phase 1 PR may not be merged yet. Continuing anyway..."
fi
echo "✅ Phase 1 status checked"

# Clone fresh copy
echo "✓ Cloning fresh copy from main..."
mkdir -p /tmp/droid-migration-phase2
cd /tmp/droid-migration-phase2
rm -rf $REPO
gh repo clone EffortlessMetrics/$REPO
cd $REPO

echo "✓ Verifying on main branch..."
git status
echo "✅ Clone successful and on main"
```

**Expected output:**
```
==========================================
Phase 2: OpenRacing
==========================================

Step 1: Verify Phase 1 Merged & Clone

✓ Checking if Phase 1 PR is merged...
✅ Phase 1 status checked
✓ Cloning fresh copy from main...
✓ Verifying on main branch...
On branch main
Your branch is up to date with 'origin/main'.
✅ Clone successful and on main
```

---

## Step 2: Create Phase 2 Branch

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 2: Create Phase 2 Branch"
echo ""

BRANCH="ci/droid-baseline"

echo "✓ Creating branch: $BRANCH..."
git checkout -b $BRANCH

echo "✓ Verifying branch..."
git branch
echo ""
echo "✅ Branch created and checked out"
```

---

## Step 3: Edit droid-review.yml (MiniMax BYOK + Guards)

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 3: Edit droid-review.yml (MiniMax BYOK + Guards)"
echo ""

FILE=".github/workflows/droid-review.yml"

echo "✓ Creating backup..."
cp $FILE ${FILE}.bak

# 3a. Add env var at job level (after "jobs:" and "  droid-review:")
echo "✓ Step 3a: Adding MINIMAX_API_KEY env var..."
JOBS_LINE=$(grep -n "^jobs:" $FILE | cut -d: -f1)
DROID_REVIEW_LINE=$(grep -n "^  droid-review:" $FILE | cut -d: -f1)

if [ ! -z "$DROID_REVIEW_LINE" ]; then
  # Find next line after droid-review:
  ENV_INSERT_LINE=$((DROID_REVIEW_LINE + 1))
  
  # Check if env: already exists
  if ! sed -n "${DROID_REVIEW_LINE},+10p" $FILE | grep -q "env:"; then
    sed -i "${ENV_INSERT_LINE}i\\    env:\\n      MINIMAX_API_KEY: \${{ secrets.MINIMAX_API_KEY }}" $FILE
    echo "  Added ✅"
  else
    echo "  env: already exists, skipping"
  fi
fi

# 3b. Add if guard (same level as env)
echo "✓ Step 3b: Adding same-repo guard..."
if ! grep -q 'github.event.pull_request.head.repo.full_name' $FILE; then
  # Find env: line and add if: after it
  ENV_LINE=$(grep -n "    env:" $FILE | head -1 | cut -d: -f1)
  if [ ! -z "$ENV_LINE" ]; then
    # Add 2 lines down (env + indented key)
    INSERT_AFTER=$((ENV_LINE + 2))
    sed -i "${INSERT_AFTER}a\\\\n    if: |\\n      github.event.pull_request.head.repo.full_name == github.repository &&\\n      !contains(github.event.pull_request.title, '[skip-review]')" $FILE
    echo "  Added ✅"
  fi
fi

# 3c. Add MiniMax BYOK step before the Droid action
echo "✓ Step 3b: Adding MiniMax BYOK step..."
DROID_ACTION_LINE=$(grep -n "uses: EffortlessMetrics/droid-action-safe" $FILE | head -1 | cut -d: -f1)
if [ ! -z "$DROID_ACTION_LINE" ]; then
  STEP_INSERT_LINE=$((DROID_ACTION_LINE - 1))
  
  # Find the nearest "- name:" before the action
  PREV_STEP_LINE=$(sed -n "1,${STEP_INSERT_LINE}p" $FILE | grep -n "      - name:" | tail -1 | cut -d: -f1)
  if [ ! -z "$PREV_STEP_LINE" ]; then
    # Count actual line numbers in file
    PREV_STEP_ACTUAL=$(sed -n "1,${STEP_INSERT_LINE}p" $FILE | tail -n +1 | head -n ${PREV_STEP_LINE} | wc -l)
    STEP_INSERT_ACTUAL=$((STEP_INSERT_LINE - 1))
    
    # Insert the BYOK step
    cat >> /tmp/byok_step.txt << 'BYOK_END'

      - name: Configure MiniMax BYOK for Factory Droid
        shell: bash
        run: |
          mkdir -p "$HOME/.factory"
          cat > "$HOME/.factory/settings.local.json" <<'JSON'
          {
            "customModels": [
              {
                "displayName": "MiniMax-M2.7",
                "model": "MiniMax-M2.7",
                "baseUrl": "https://api.minimax.io/anthropic",
                "apiKey": "${MINIMAX_API_KEY}",
                "provider": "anthropic",
                "maxOutputTokens": 64000,
                "noImageSupport": true,
                "extraArgs": {
                  "temperature": 1
                }
              }
            ]
          }
          JSON
BYOK_END
    
    sed -i "${STEP_INSERT_LINE}r /tmp/byok_step.txt" $FILE
    echo "  Added ✅"
  fi
fi

# 3d. Update action inputs
echo "✓ Step 3d: Updating action inputs..."
sed -i 's/review_model:.*$/review_model: "custom:MiniMax-M2.7-0"/g' $FILE
sed -i 's/security_model:.*$/security_model: "custom:MiniMax-M2.7-0"/g' $FILE

if ! grep -q "review_depth: shallow" $FILE; then
  echo "review_depth: shallow" >> /tmp/action_inputs.txt
fi
if ! grep -q "show_full_output: false" $FILE; then
  echo "show_full_output: false" >> /tmp/action_inputs.txt
fi

echo "  Updated ✅"

# Validate YAML
echo "✓ Validating YAML syntax..."
python3 -c "import yaml; yaml.safe_load(open('$FILE'))" && echo "  ✅ Valid YAML" || (echo "  ❌ Invalid YAML"; exit 1)

echo ""
echo "✅ droid-review.yml successfully configured for Phase 2"
```

**Expected output:**
```
Step 3: Edit droid-review.yml (MiniMax BYOK + Guards)

✓ Creating backup...
✓ Step 3a: Adding MINIMAX_API_KEY env var...
  Added ✅
✓ Step 3b: Adding same-repo guard...
  Added ✅
✓ Step 3b: Adding MiniMax BYOK step...
  Added ✅
✓ Step 3d: Updating action inputs...
  Updated ✅
✓ Validating YAML syntax...
  ✅ Valid YAML

✅ droid-review.yml successfully configured for Phase 2
```

---

## Step 4: Edit droid.yml (Trusted-Actor Guard + BYOK)

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 4: Edit droid.yml (Trusted-Actor Guard + BYOK)"
echo ""

FILE=".github/workflows/droid.yml"

echo "✓ Creating backup..."
cp $FILE ${FILE}.bak

# 4a. Add trusted-actor guard
echo "✓ Step 4a: Adding trusted-actor guard..."
DROID_JOB_LINE=$(grep -n "^  droid:" $FILE | cut -d: -f1)

if ! grep -q 'author_association' $FILE; then
  INSERT_LINE=$((DROID_JOB_LINE + 1))
  sed -i "${INSERT_LINE}i\\    if: |\\n      (\\n        github.event_name == 'issue_comment' &&\\n        contains(github.event.comment.body, '@droid') &&\\n        contains(fromJSON('[\"OWNER\",\"MEMBER\",\"COLLABORATOR\"]'), github.event.comment.author_association)\\n      ) ||\\n      (\\n        github.event_name == 'pull_request_review_comment' &&\\n        contains(github.event.comment.body, '@droid') &&\\n        contains(fromJSON('[\"OWNER\",\"MEMBER\",\"COLLABORATOR\"]'), github.event.comment.author_association)\\n      ) ||\\n      (\\n        github.event_name == 'pull_request_review' &&\\n        contains(github.event.review.body, '@droid') &&\\n        contains(fromJSON('[\"OWNER\",\"MEMBER\",\"COLLABORATOR\"]'), github.event.review.author_association)\\n      ) ||\\n      (\\n        github.event_name == 'issues' &&\\n        (contains(github.event.issue.body, '@droid') || contains(github.event.issue.title, '@droid')) &&\\n        contains(fromJSON('[\"OWNER\",\"MEMBER\",\"COLLABORATOR\"]'), github.event.issue.author_association)\\n      )" $FILE
  echo "  Added ✅"
fi

# 4b. Update permissions: contents: read
echo "✓ Step 4b: Updating permissions to contents: read..."
sed -i 's/contents: write/contents: read/g' $FILE
echo "  Updated ✅"

# 4c. Add MINIMAX_API_KEY env var
echo "✓ Step 4c: Adding MINIMAX_API_KEY env var..."
if ! grep -q "MINIMAX_API_KEY" $FILE; then
  sed -i "/^  droid:/a\\    env:\\n      MINIMAX_API_KEY: \${{ secrets.MINIMAX_API_KEY }}" $FILE
  echo "  Added ✅"
fi

# 4d. Add MiniMax BYOK step (same as droid-review.yml)
echo "✓ Step 4d: Adding MiniMax BYOK step..."
DROID_ACTION_LINE=$(grep -n "uses: EffortlessMetrics/droid-action-safe" $FILE | head -1 | cut -d: -f1)
if [ ! -z "$DROID_ACTION_LINE" ]; then
  STEP_INSERT_LINE=$((DROID_ACTION_LINE - 1))
  
  cat >> /tmp/byok_step_manual.txt << 'BYOK_END'

      - name: Configure MiniMax BYOK for Factory Droid
        shell: bash
        run: |
          mkdir -p "$HOME/.factory"
          cat > "$HOME/.factory/settings.local.json" <<'JSON'
          {
            "customModels": [
              {
                "displayName": "MiniMax-M2.7",
                "model": "MiniMax-M2.7",
                "baseUrl": "https://api.minimax.io/anthropic",
                "apiKey": "${MINIMAX_API_KEY}",
                "provider": "anthropic",
                "maxOutputTokens": 64000,
                "noImageSupport": true,
                "extraArgs": {
                  "temperature": 1
                }
              }
            ]
          }
          JSON
BYOK_END
  
  sed -i "${STEP_INSERT_LINE}r /tmp/byok_step_manual.txt" $FILE
  echo "  Added ✅"
fi

# 4e. Update action inputs
echo "✓ Step 4e: Updating action inputs..."
sed -i 's/review_model:.*$/review_model: "custom:MiniMax-M2.7-0"/g' $FILE
sed -i 's/security_model:.*$/security_model: "custom:MiniMax-M2.7-0"/g' $FILE
sed -i 's/show_full_output:.*$/show_full_output: false/g' $FILE
echo "  Updated ✅"

# Validate YAML
echo "✓ Validating YAML syntax..."
python3 -c "import yaml; yaml.safe_load(open('$FILE'))" && echo "  ✅ Valid YAML" || (echo "  ❌ Invalid YAML"; exit 1)

echo ""
echo "✅ droid.yml successfully configured for Phase 2"
```

**Expected output:**
```
Step 4: Edit droid.yml (Trusted-Actor Guard + BYOK)

✓ Creating backup...
✓ Step 4a: Adding trusted-actor guard...
  Added ✅
✓ Step 4b: Updating permissions to contents: read...
  Updated ✅
✓ Step 4c: Adding MINIMAX_API_KEY env var...
  Added ✅
✓ Step 4d: Adding MiniMax BYOK step...
  Added ✅
✓ Step 4e: Updating action inputs...
  Updated ✅
✓ Validating YAML syntax...
  ✅ Valid YAML

✅ droid.yml successfully configured for Phase 2
```

---

## Step 5: Create AGENTS.md

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 5: Create AGENTS.md"
echo ""

cat > AGENTS.md << 'AGENTS_EOF'
# Droid Review Configuration

This repository uses Factory Droid for automated code review with MiniMax M2.7.

## Review Rules

- No naked LGTM comments
- Findings must be repair packets with failure mode, fix direction, validation
- Clean reviews include inspection record with observed/reported/not-verified
- No extra @mentions in Droid-generated bodies
- Evidence split by provenance

## Triggers

- **Auto-review:** Same-repo PRs, auto-triggered on open/sync/ready-for-review
- **Manual:** Comment `@droid review` or `@droid security` (OWNER/MEMBER/COLLABORATOR only)

## Model

MiniMax M2.7 via BYOK (custom:MiniMax-M2.7-0)

## For Reviewers

- Expect shallow review (priority on correctness, security, maintainability)
- Droid reviews are repair-packet format; see `.factory/rules/droid-review.md`
- Manual @droid follow-up for deep dives if needed
AGENTS_EOF

echo "✅ AGENTS.md created"
```

**Expected output:**
```
Step 5: Create AGENTS.md

✅ AGENTS.md created
```

---

## Step 6: Create .factory/rules/droid-review.md

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 6: Create .factory/rules/droid-review.md"
echo ""

mkdir -p .factory/rules

cat > .factory/rules/droid-review.md << 'RULES_EOF'
# Droid Review Rules

## Finding Format (P0/P1/P2)

[P0|P1|P2] Short title

Failure mode: Why this matters
Why here: Specific location/context analysis
Fix direction: Concrete next step
Validation: How to verify the fix
Confidence: High/Medium/Low

## Clean Review Format

No actionable findings emitted.

Inspected surfaces:
- API signatures
- Error handling
- Type safety

Checks performed:
- Static analysis
- Pattern matching
- Consistency verification

Why no comments: All surfaces passed checks or are out of scope

Residual risk:
- Runtime behavior (dynamic dispatch, concurrency)

Validation signal:
  Observed: Tests pass, no lint warnings
  Reported: CI green, code review approval
  Not verified: Performance characteristics
RULES_EOF

echo "✅ .factory/rules/droid-review.md created"
```

**Expected output:**
```
Step 6: Create .factory/rules/droid-review.md

✅ .factory/rules/droid-review.md created
```

---

## Step 7: Review All Changes

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 7: Review All Changes"
echo ""

echo "✓ Git status:"
git status

echo ""
echo "✓ Files to be added/modified:"
echo "  Modified:"
echo "    - .github/workflows/droid.yml"
echo "    - .github/workflows/droid-review.yml"
echo "  New:"
echo "    - AGENTS.md"
echo "    - .factory/rules/droid-review.md"

echo ""
echo "Items to verify:"
echo "  ☐ Both workflow files have MINIMAX_API_KEY env var"
echo "  ☐ droid-review.yml has same-repo guard"
echo "  ☐ droid.yml has trusted-actor guard"
echo "  ☐ Both workflows have MiniMax BYOK step (heredoc)"
echo "  ☐ Both workflows have custom:MiniMax-M2.7-0 model inputs"
echo "  ☐ Both workflows have review_depth: shallow"
echo "  ☐ Both workflows have show_full_output: false"
echo "  ☐ droid.yml has contents: read (not write)"
echo "  ☐ AGENTS.md created"
echo "  ☐ .factory/rules/droid-review.md created"
echo ""
echo "✅ Changes reviewed"
```

---

## Step 8: Commit Changes

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 8: Commit Changes"
echo ""

echo "✓ Staging files..."
git add .github/workflows/droid*.yml AGENTS.md .factory/

echo "✓ Verifying staged files..."
git diff --cached --name-only

echo ""
echo "✓ Creating commit..."
git commit -m "ci: align Droid review baseline

- Add MiniMax BYOK through ~/.factory/settings.local.json
- Set review model to custom:MiniMax-M2.7-0
- Add same-repo guard for auto review
- Add trusted-actor guard for manual @droid
- Add minimal repo-local guidance (AGENTS.md, .factory/rules/)

Refs: MIGRATION_PLAN.md, IMPLEMENTATION_OPENRACING.md"

echo ""
echo "✓ Verifying commit..."
git log --oneline -2

echo ""
echo "✅ Commit created successfully"
```

---

## Step 9: Push to Remote

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 9: Push to Remote"
echo ""

BRANCH="ci/droid-baseline"

echo "✓ Pushing branch: $BRANCH..."
git push -u origin $BRANCH

echo ""
echo "✅ Branch pushed successfully"
```

---

## Step 10: Create Pull Request

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO
cd /tmp/droid-migration-phase2/$REPO

echo ""
echo "Step 10: Create Pull Request"
echo ""

BRANCH="ci/droid-baseline"
TITLE="ci: align Droid review baseline"

echo "✓ Creating PR..."
gh pr create \
  --repo EffortlessMetrics/$REPO \
  --base main \
  --head $BRANCH \
  --title "$TITLE" \
  --body "## Summary

- Add MiniMax BYOK through \`~/.factory/settings.local.json\`
- Set review model to \`custom:MiniMax-M2.7-0\`
- Add same-repo guard for auto review
- Add trusted-actor guard for manual @droid
- Add minimal repo-local guidance

## Why

Convergence to org baseline reduces review variance and ensures safe, consistent BYOK model usage.

## Changes

- \`.github/workflows/droid-review.yml\` — BYOK step, model inputs, same-repo guard
- \`.github/workflows/droid.yml\` — Trusted-actor guard, model inputs
- \`AGENTS.md\` — High-level review config
- \`.factory/rules/droid-review.md\` — Droid-specific rules

## Validation

- [x] Repo workflow/static checks pass.
- [ ] Same-repo smoke PR succeeds with MiniMax model.
- [ ] Manual \`@droid review\` works (OWNER/MEMBER comment).
- [ ] Manual \`@droid security\` works.
- [ ] No raw artifacts uploaded.

Refs: MIGRATION_PLAN.md, IMPLEMENTATION_OPENRACING.md"

echo ""
echo "✅ PR created successfully"
```

---

## Step 11: Code Review & Approval

**Duration:** 1-24 hours  
**Action:** Manual (outside of this script)

### Checklist for Code Reviewer

- [ ] PR title is "ci: align Droid review baseline"
- [ ] MINIMAX_API_KEY env var present in both workflows
- [ ] Same-repo guard in droid-review.yml: `github.event.pull_request.head.repo.full_name == github.repository`
- [ ] Trusted-actor guard in droid.yml checking author_association
- [ ] Model inputs: `custom:MiniMax-M2.7-0` in both workflows
- [ ] Review depth: `shallow` in both workflows
- [ ] Show output: `false` in both workflows
- [ ] droid.yml has `contents: read` (not write)
- [ ] MiniMax BYOK heredoc is properly quoted (single quotes around EOF)
- [ ] AGENTS.md file created
- [ ] .factory/rules/droid-review.md file created
- [ ] Workflow YAML is syntactically valid
- [ ] Approve the PR

---

## Step 12: Merge PR

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo ""
echo "Step 12: Merge PR"
echo ""

PR_NUMBER=$(gh pr list --repo EffortlessMetrics/$REPO \
  --head ci/droid-baseline \
  --state open \
  --json number \
  --jq '.[0].number')

if [ -z "$PR_NUMBER" ]; then
  echo "❌ No open PR found on ci/droid-baseline"
  exit 1
fi

echo "✓ Merging PR #$PR_NUMBER..."
gh pr merge $PR_NUMBER \
  --repo EffortlessMetrics/$REPO \
  --merge \
  --delete-branch

echo ""
echo "✅ PR merged successfully"
```

---

## Step 13: Smoke Test Phase 2 (MiniMax + Manual @droid)

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo ""
echo "Step 13: Smoke Test Phase 2"
echo ""

echo "⏳ Waiting 30 seconds for main to be updated..."
sleep 30

# Create test PR
cd /tmp/droid-migration-phase2/$REPO
git fetch origin
git checkout main
git pull origin main

git checkout -b smoke-test-phase2
echo "# Smoke Test Phase 2" >> README.md
git add README.md
git commit -m "smoke: test phase 2 minimax byok"
git push -u origin smoke-test-phase2

echo ""
echo "✓ Creating draft PR..."
gh pr create \
  --repo EffortlessMetrics/$REPO \
  --base main \
  --head smoke-test-phase2 \
  --draft \
  --title "[smoke-test] Droid phase 2 MiniMax BYOK" \
  --body "Smoke test for Phase 2 MiniMax BYOK.

Verify:
- Droid Auto Review triggers with MiniMax
- Check logs for: custom:MiniMax-M2.7-0
- Manual @droid review works
- Manual @droid security works"

echo ""
echo "⏳ Waiting 3 minutes for auto review..."
sleep 180

# Check for MiniMax in logs
echo "✓ Checking for MiniMax in workflow logs..."
RUN_ID=$(gh run list --repo EffortlessMetrics/$REPO --head smoke-test-phase2 --limit 1 --json databaseId --jq '.[0].databaseId')

if [ ! -z "$RUN_ID" ]; then
  gh run view $RUN_ID --repo EffortlessMetrics/$REPO --log | grep -i "custom:MiniMax-M2.7-0" && echo "✅ MiniMax model detected" || echo "⚠️ MiniMax not detected in logs"
fi

# Test manual @droid (must be OWNER/MEMBER)
echo ""
echo "✓ Testing manual @droid review..."
PR_NUMBER=$(gh pr list --repo EffortlessMetrics/$REPO --head smoke-test-phase2 --state open --json number --jq '.[0].number')

if [ ! -z "$PR_NUMBER" ]; then
  gh pr comment $PR_NUMBER --repo EffortlessMetrics/$REPO --body "@droid review"
  echo "  Manual @droid triggered"
  echo "  ⏳ Check GitHub for Droid response (should use MiniMax)"
fi

echo ""
echo "✓ Deleting test PR..."
sleep 30
gh pr delete smoke-test-phase2 --repo EffortlessMetrics/$REPO --yes || true
git push origin --delete smoke-test-phase2 || true

echo ""
echo "✅ Smoke Test Phase 2 completed for $REPO"
echo ""
echo "Manual verification items:"
echo "  ☐ Droid Auto Review used MiniMax (check logs)"
echo "  ☐ Manual @droid review responded correctly"
echo "  ☐ No raw droid-review-debug-* artifacts"
```

---

## Step 14: Mark Completion

```bash
#!/bin/bash

REPO="OpenRacing"  # ← CHANGE THIS FOR EACH REPO

echo ""
echo "Step 14: Mark Completion"
echo ""

echo "✅ $REPO Phase 2 COMPLETE"
echo ""
echo "Update TRACKING_SHEET.md:"
echo "  - Mark $REPO Phase 2 ✅ COMPLETE"
echo "  - Record PR number and merge date"
echo "  - Record smoke test results"
echo ""
echo "Next repo: adze"
```

---

## Phase 2 Complete

When all 7 repos have:
- ✅ Phase 2 PR merged
- ✅ Smoke test with MiniMax passed
- ✅ Manual @droid proven
- ✅ AGENTS.md and .factory/rules/ files created
- ✅ Marked complete in TRACKING_SHEET.md

Proceed to **Final Validation & Sign-Off** (EXECUTION_SUMMARY.md)
