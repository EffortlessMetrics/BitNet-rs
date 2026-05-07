# EffortlessMetrics Factory Action Migration: Master Runbook

**Mode:** Manual execution (step-by-step)  
**Target:** 7 EffortlessMetrics repositories  
**Effort:** 3-4 weeks, ~1-2 days per repo  
**Safety:** All changes are auditable; each step can be verified before proceeding

---

## Prerequisites

### For Every Executor

- [ ] GitHub CLI (`gh`) installed and authenticated
- [ ] Git installed (version 2.30+)
- [ ] Read access to all 7 target repos
- [ ] Write access to create branches and PRs in all 7 target repos
- [ ] (Ops only) Access to GitHub secrets management for the org

### Phase 0: Ops Completion (Blocking)

**MUST be done before Phase 1 starts**

- [ ] MiniMax Token Plan key rotated (compromised key replaced)
- [ ] New MINIMAX_API_KEY secret created in GitHub org
- [ ] Secret scoped to exactly these 7 repos (no more, no less):
  - OpenRacing
  - adze
  - SwiftMTP-dev
  - SwiftMailSort
  - shiplog
  - perl-lsp
  - pkm-python
- [ ] FACTORY_API_KEY verified as valid (still active, not expired)
- [ ] Phase 0 sign-off: Ops confirms in #droid-migration

### For This Runbook

- [ ] Safe action SHA memorized or bookmarked: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- [ ] Model name memorized: `custom:MiniMax-M2.7-0`
- [ ] Artifact flag memorized: `upload_debug_artifacts: false`
- [ ] TRACKING_SHEET.md available for progress tracking

---

## Repo Access Verification

Run this once to verify you have access to all 7 repos:

```bash
#!/bin/bash

REPOS=(
  "OpenRacing"
  "adze"
  "SwiftMTP-dev"
  "SwiftMailSort"
  "shiplog"
  "perl-lsp"
  "pkm-python"
)

echo "Verifying access to all 7 target repos..."
for repo in "${REPOS[@]}"; do
  echo -n "Checking EffortlessMetrics/$repo ... "
  if gh repo view EffortlessMetrics/$repo &>/dev/null; then
    echo "✅ OK"
  else
    echo "❌ FAIL (no read access or repo not found)"
  fi
done

echo ""
echo "If all 7 show ✅, you're ready to proceed."
```

**Expected output:** 7 checkmarks, 0 failures

---

## Common Procedures (All Phases)

### Procedure A: Clone and Setup Local Repo

**Input:** Repo name (e.g., `OpenRacing`)  
**Output:** Local repo ready for edits

```bash
REPO="OpenRacing"  # Change per repo

# 1. Clone
cd /tmp  # or your work directory
gh repo clone EffortlessMetrics/$REPO
cd $REPO

# 2. Verify location
pwd
# Expected: /tmp/OpenRacing (or your path)

# 3. Check current branch
git status
# Expected: On branch main, nothing to commit, working tree clean

# 4. Verify workflows directory
ls -la .github/workflows/droid*.yml
# Expected: droid.yml and droid-review.yml present
```

### Procedure B: Create Feature Branch

**Input:** Branch name (e.g., `ci/safe-droid-action`)  
**Output:** New branch, ready for edits

```bash
BRANCH="ci/safe-droid-action"  # or ci/droid-baseline for Phase 2

# 1. Create and checkout branch
git checkout -b $BRANCH

# 2. Verify
git branch
# Expected: * ci/safe-droid-action listed

# 3. Verify it's tracking nothing yet (new branch)
git status
# Expected: On branch ci/safe-droid-action, nothing to commit
```

### Procedure C: Examine Current Workflow File

**Input:** Workflow filename (e.g., `.github/workflows/droid-review.yml`)  
**Output:** Current action ref identified

```bash
FILE=".github/workflows/droid-review.yml"

# 1. Search for Factory-AI action
grep -n "Factory-AI/droid-action" $FILE

# Expected output (one of these):
#   12:      uses: Factory-AI/droid-action@main
#   15:      uses: Factory-AI/droid-action@v5
#   18:      uses: Factory-AI/droid-action@<40-char-sha>

# Note the line number (e.g., 12) for later edits
```

### Procedure D: Edit File with sed (Safe)

**Input:** File path, line number, old text, new text  
**Output:** File edited, backup created

```bash
FILE=".github/workflows/droid-review.yml"
LINE=12  # From Procedure C

# 1. Create backup
cp $FILE ${FILE}.bak

# 2. Use sed to replace (single line, safe)
sed -i.bak2 "s|Factory-AI/droid-action@main|EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5; raw debug artifact upload disabled|" $FILE

# 3. Verify the change
grep -A1 -B1 "droid-action-safe" $FILE

# If wrong, restore:
# cp ${FILE}.bak $FILE
```

### Procedure E: Add YAML Line in `with:` Block

**Input:** Workflow file, line number of `with:` section  
**Output:** `upload_debug_artifacts: false` added

```bash
FILE=".github/workflows/droid-review.yml"

# 1. Find the line with "with:" and note line number
grep -n "with:" $FILE
# Expected: 20:      with:

# 2. Find the first field under with: (note line number)
sed -n '20,30p' $FILE
# Expected:
#   20:      with:
#   21:        factory_api_key: ${{ secrets.FACTORY_API_KEY }}
#   22:        ...

# 3. Insert new line after line 21 (factory_api_key)
sed -i '21a\        upload_debug_artifacts: false' $FILE

# 4. Verify
grep -A2 "factory_api_key" $FILE
# Expected: factory_api_key on line 1, upload_debug_artifacts on line 2
```

### Procedure F: Validate YAML Syntax

**Input:** Workflow file  
**Output:** Valid YAML or error message

```bash
FILE=".github/workflows/droid-review.yml"

# 1. Check if yamllint is installed
which yamllint || echo "yamllint not found; using Python instead"

# 2a. With yamllint (preferred)
yamllint $FILE

# 2b. With Python (fallback)
python3 -c "import yaml; yaml.safe_load(open('$FILE'))" && echo "✅ Valid YAML" || echo "❌ Invalid YAML"

# Expected: No errors, "✅ Valid YAML"
```

### Procedure G: Create Commit

**Input:** Files to stage, commit message  
**Output:** Commit created on feature branch

```bash
# 1. Check what changed
git status
# Expected: Modified files listed

# 2. Stage files
git add .github/workflows/droid*.yml
# (add other files if Phase 2)

# 3. Verify staged files
git diff --cached --name-only

# 4. Create commit with message
git commit -m "ci: use safe Droid action

- Replace Factory-AI/droid-action with EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f
- Add upload_debug_artifacts: false
- Preserve existing Droid behavior except for disabling raw debug artifact upload

Refs: MIGRATION_PLAN.md"

# 5. Verify commit
git log --oneline -1
```

### Procedure H: Push Branch

**Input:** Branch name  
**Output:** Branch pushed to remote, ready for PR

```bash
BRANCH="ci/safe-droid-action"

# 1. Push with upstream tracking
git push -u origin $BRANCH

# Expected output:
# remote: Create a pull request for 'ci/safe-droid-action' on GitHub by visiting:
# remote: https://github.com/EffortlessMetrics/OpenRacing/pull/new/ci/safe-droid-action

# 2. Copy the PR URL for next step
# Save: https://github.com/EffortlessMetrics/OpenRacing/pull/new/ci/safe-droid-action
```

### Procedure I: Create GitHub PR

**Input:** PR title, body template, branch URL  
**Output:** PR created, ready for review

```bash
REPO="OpenRacing"
BRANCH="ci/safe-droid-action"
TITLE="ci: use safe Droid action"

# 1. Create PR with gh CLI
gh pr create \
  --repo EffortlessMetrics/$REPO \
  --base main \
  --head $BRANCH \
  --title "$TITLE" \
  --body "$(cat <<'EOF'
## Summary

- Switch Droid workflows to `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`.
- Add `upload_debug_artifacts: false`.
- Preserve existing Droid behavior except for disabling raw debug artifact upload.

## Why

The upstream Factory action can upload raw `$HOME/.factory/**` and `droid-prompts/**`. In BYOK mode that can include resolved provider credentials. Normal Droid runs should not upload raw debug artifacts.

## Validation

- [x] Repo workflow/static checks pass.
- [ ] Same-repo PR smoke run succeeds.
- [ ] No raw artifact named `droid-review-debug-<run_id>` is uploaded.

## Non-goals

- No permission reduction.
- No model/provider change except MiniMax BYOK convergence if already intended.
- No `review_depth: deep`.
- No `pull_request_target`.

Refs: MIGRATION_PLAN.md, IMPLEMENTATION_OPENRACING.md
EOF
)"

# Expected: PR created successfully
# Output will include PR number and URL
```

### Procedure J: Smoke Test (Draft PR)

**Input:** Repo, branch  
**Output:** Droid run verified, no artifacts leaked

```bash
REPO="OpenRacing"
BRANCH_TO_TEST="main"  # After Phase 1 PR is merged

# 1. Create a test branch
git checkout $BRANCH_TO_TEST
git pull origin $BRANCH_TO_TEST
git checkout -b smoke-test-phase1

# 2. Make a trivial change (update README)
echo "# Smoke Test" >> README.md
git add README.md
git commit -m "smoke: test phase 1 droid safe action"
git push -u origin smoke-test-phase1

# 3. Create draft PR (NOT to merge, just to trigger Droid)
gh pr create \
  --repo EffortlessMetrics/$REPO \
  --base main \
  --head smoke-test-phase1 \
  --title "[smoke-test] Droid phase 1 safe action" \
  --draft \
  --body "Smoke test for Phase 1 safe action deployment. Verify:
- Droid Auto Review triggers
- No raw droid-review-debug-<run_id> artifacts
- Workflow completes successfully"

# 4. Wait 2-3 minutes for Droid to run
echo "⏳ Waiting for Droid workflow to start (check GitHub Actions)..."
sleep 120

# 5. Inspect workflow run
gh workflow run list --repo EffortlessMetrics/$REPO --limit 1

# 6. Check for artifacts
gh run list --repo EffortlessMetrics/$REPO --limit 1 --json name,status
# Look for "Droid Auto Review" with "completed" status

# 7. Get artifacts from run
RUN_ID=$(gh run list --repo EffortlessMetrics/$REPO --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download $RUN_ID --repo EffortlessMetrics/$REPO --dir /tmp/artifacts-$RUN_ID || true

# 8. Check artifact names
ls -la /tmp/artifacts-$RUN_ID/ | grep -i droid
# Expected: NO artifact named "droid-review-debug-*"
# Allowed: No artifacts, or "summary-*" only

# 9. Delete test PR (don't merge)
gh pr delete smoke-test-phase1 --repo EffortlessMetrics/$REPO --yes

# 10. Clean up test branch
git push origin --delete smoke-test-phase1

# 11. Mark validation in TRACKING_SHEET.md
echo "✅ Phase 1 Smoke Test PASSED for $REPO"
```

### Procedure K: Check PR Status and Merge

**Input:** PR number or URL  
**Output:** PR approved and merged

```bash
REPO="OpenRacing"
PR_NUMBER=123  # From PR creation output

# 1. Check PR status
gh pr view $PR_NUMBER --repo EffortlessMetrics/$REPO

# Expected output includes:
# - Status: OPEN (or CLOSED if already merged)
# - Reviews: APPROVED or CHANGES_REQUESTED
# - Checks: All green (✓) or pending (→)

# 2. Wait for all checks to pass
echo "⏳ Waiting for checks to complete..."
# Check GitHub Actions manually or run:
gh run list --repo EffortlessMetrics/$REPO --head <branch> --limit 5

# 3. Merge PR (when approved and checks pass)
gh pr merge $PR_NUMBER \
  --repo EffortlessMetrics/$REPO \
  --merge \
  --delete-branch

# Expected: "✓ Pull request #123 merged successfully"
```

---

## Phase 1 Execution Sequence

**Goal:** Replace unsafe action refs; disable artifacts; no behavior change

### For Each Repo in Batch 1 (5 repos):

**Sequence:**
1. OpenRacing
2. adze
3. SwiftMTP-dev
4. SwiftMailSort
5. shiplog

### Per-Repo Phase 1 Checklist

```bash
REPO="OpenRacing"

# ✅ STEP 1: Clone and examine
echo "📋 STEP 1: Clone and examine current workflows"
# Run Procedure A (Clone and Setup Local Repo)
# Run Procedure C (Examine Current Workflow File) on both droid.yml and droid-review.yml
# Record current action refs for each file

# ✅ STEP 2: Create Phase 1 branch
echo "📋 STEP 2: Create Phase 1 branch"
# Run Procedure B with branch name: ci/safe-droid-action

# ✅ STEP 3: Edit droid-review.yml
echo "📋 STEP 3: Edit droid-review.yml (auto review workflow)"
# Run Procedure C to find the Factory-AI line
# Run Procedure D to replace with safe action SHA
# Run Procedure E to add upload_debug_artifacts: false (if not present)
# Run Procedure F to validate YAML

# ✅ STEP 4: Edit droid.yml
echo "📋 STEP 4: Edit droid.yml (manual @droid workflow)"
# Run Procedure C to find the Factory-AI line
# Run Procedure D to replace with safe action SHA
# Run Procedure E to add upload_debug_artifacts: false (if not present)
# Run Procedure F to validate YAML

# ✅ STEP 5: Verify both files
echo "📋 STEP 5: Verify both workflow files"
git diff .github/workflows/droid*.yml
# Review: should see 2 replacements (action ref + artifact flag additions)

# ✅ STEP 6: Commit
echo "📋 STEP 6: Commit changes"
# Run Procedure G with commit message: "ci: use safe Droid action"

# ✅ STEP 7: Push
echo "📋 STEP 7: Push to remote"
# Run Procedure H

# ✅ STEP 8: Create PR
echo "📋 STEP 8: Create GitHub PR"
# Run Procedure I with title: "ci: use safe Droid action"

# ✅ STEP 9: Wait for approval
echo "📋 STEP 9: Code review and approval"
# Check GitHub manually
# Ask code reviewer to verify:
# - Safe action SHA is correct
# - upload_debug_artifacts: false is present
# - No unrelated changes
# Approve in GitHub

# ✅ STEP 10: Merge
echo "📋 STEP 10: Merge PR"
# Run Procedure K when approved and checks pass

# ✅ STEP 11: Smoke test
echo "📋 STEP 11: Smoke test Phase 1"
# Run Procedure J (Smoke Test)
# Verify: Droid triggers, no unsafe artifacts, workflow completes

# ✅ STEP 12: Update tracking
echo "📋 STEP 12: Update tracking sheet"
# Mark $REPO Phase 1 ✅ COMPLETE in TRACKING_SHEET.md
```

---

## Phase 2 Execution Sequence

**Goal:** Add MiniMax BYOK; add guards; add guidance files

### For Each Repo (same sequence as Phase 1):

**Sequence:** Same 7 repos, same order

### Per-Repo Phase 2 Checklist

```bash
REPO="OpenRacing"

# ✅ STEP 1: Clone and examine (updated main branch)
echo "📋 STEP 1: Clone and examine updated workflows"
# Run Procedure A (Clone fresh copy of main with Phase 1 merged)
cd /tmp/OpenRacing-phase2  # different directory
gh repo clone EffortlessMetrics/$REPO

# ✅ STEP 2: Create Phase 2 branch
echo "📋 STEP 2: Create Phase 2 branch"
# Run Procedure B with branch name: ci/droid-baseline

# ✅ STEP 3: Edit droid-review.yml (add BYOK + guards)
echo "📋 STEP 3: Edit droid-review.yml"
# 3a. Add MINIMAX_API_KEY env var at job level:
#     env:
#       MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

# 3b. Add if guard at job level:
#     if: |
#       github.event.pull_request.head.repo.full_name == github.repository &&
#       !contains(github.event.pull_request.title, '[skip-review]')

# 3c. Add MiniMax BYOK step before the Droid action:
#     (see IMPLEMENTATION_OPENRACING.md for full heredoc)

# 3d. Update action inputs:
#     review_model: "custom:MiniMax-M2.7-0"
#     security_model: "custom:MiniMax-M2.7-0"
#     review_depth: shallow
#     show_full_output: false

# Run Procedure F to validate YAML

# ✅ STEP 4: Edit droid.yml (add guards, update inputs)
echo "📋 STEP 4: Edit droid.yml"
# 4a. Add trusted-actor guard (see IMPLEMENTATION_OPENRACING.md)
# 4b. Add MINIMAX_API_KEY env var
# 4c. Change permissions: contents: read (not write)
# 4d. Add same MiniMax BYOK step as droid-review.yml
# 4e. Update action inputs with model: custom:MiniMax-M2.7-0

# Run Procedure F to validate YAML

# ✅ STEP 5: Create AGENTS.md
echo "📋 STEP 5: Create AGENTS.md"
cat > AGENTS.md << 'EOF'
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
EOF

# ✅ STEP 6: Create .factory/rules/droid-review.md
echo "📋 STEP 6: Create .factory/rules/droid-review.md"
mkdir -p .factory/rules
cat > .factory/rules/droid-review.md << 'EOF'
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
EOF

# ✅ STEP 7: Verify all files
echo "📋 STEP 7: Verify all files"
git status
# Should show: modified droid*.yml, new AGENTS.md, new .factory/rules/droid-review.md

# ✅ STEP 8: Commit
echo "📋 STEP 8: Commit changes"
git add .github/workflows/droid*.yml AGENTS.md .factory/
git commit -m "ci: align Droid review baseline

- Add MiniMax BYOK through ~/.factory/settings.local.json
- Set review model to custom:MiniMax-M2.7-0
- Add same-repo guard for auto review
- Add trusted-actor guard for manual @droid
- Add minimal repo-local guidance (AGENTS.md, .factory/rules/)

Refs: MIGRATION_PLAN.md, IMPLEMENTATION_OPENRACING.md"

# ✅ STEP 9: Push
echo "📋 STEP 9: Push to remote"
# Run Procedure H

# ✅ STEP 10: Create PR
echo "📋 STEP 10: Create GitHub PR"
# Run Procedure I with title: "ci: align Droid review baseline"

# ✅ STEP 11: Code review
echo "📋 STEP 11: Code review and approval"
# Reviewer verifies:
# - BYOK heredoc syntax (single quotes around EOF)
# - Same-repo guard correct
# - Trusted-actor guard correct
# - Models: custom:MiniMax-M2.7-0
# - AGENTS.md and .factory/rules/ present
# Approve in GitHub

# ✅ STEP 12: Merge
echo "📋 STEP 12: Merge PR"
# Run Procedure K when approved and checks pass

# ✅ STEP 13: Smoke test with MiniMax
echo "📋 STEP 13: Smoke test Phase 2"
# Run Procedure J
# Also test manual @droid:
#   - Comment: "@droid review" as OWNER/MEMBER
#   - Verify Droid responds
#   - Verify logs show: custom:MiniMax-M2.7-0
#   - Comment: "@droid security" as OWNER/MEMBER
#   - Verify security scan works

# ✅ STEP 14: Update tracking
echo "📋 STEP 14: Update tracking sheet"
# Mark $REPO Phase 2 ✅ COMPLETE in TRACKING_SHEET.md
```

---

## Execution Timeline (Week by Week)

### Week 1: Phase 0 + Batch 1 Phase 1 Kickoff

| Day | Owner | Task | Repos | Duration |
|-----|-------|------|-------|----------|
| Mon-Tue | Ops | Phase 0: Key rotation, secret scoping | N/A | 1 day |
| Wed-Thu | Eng | OpenRacing Phase 1 | OpenRacing | 2 days |
| Fri | Eng | adze Phase 1 | adze | 1 day |

### Week 2: Batch 1 Phase 1 Completion + Phase 2 Start

| Day | Owner | Task | Repos | Duration |
|-----|-------|------|-------|----------|
| Mon-Tue | Eng | SwiftMTP-dev Phase 1, SwiftMailSort Phase 1 | 2 repos | 2 days |
| Wed | Eng | shiplog Phase 1 | shiplog | 1 day |
| Thu-Fri | Eng | OpenRacing Phase 2 | OpenRacing | 2 days |

### Week 3: Batch 1 Phase 2 Completion + Batch 2 Phase 1

| Day | Owner | Task | Repos | Duration |
|-----|-------|------|-------|----------|
| Mon-Tue | Eng | adze Phase 2, SwiftMTP-dev Phase 2 | 2 repos | 2 days |
| Wed-Thu | Eng | SwiftMailSort Phase 2, shiplog Phase 2 | 2 repos | 2 days |
| Fri | Eng | perl-lsp Phase 1 | perl-lsp | 1 day |

### Week 4: Batch 2 Completion + Sign-Off

| Day | Owner | Task | Repos | Duration |
|-----|-------|------|-------|----------|
| Mon-Tue | Eng | perl-lsp Phase 2, pkm-python Phase 1 | 2 repos | 2 days |
| Wed-Thu | Eng | pkm-python Phase 2 | pkm-python | 2 days |
| Fri | QA/Eng | Validation + sign-off | All 7 | 1 day |

---

## Sign-Off Procedure

### When All 7 Repos Complete Phase 1 + 2

1. **QA validation** (use TRACKING_SHEET.md checklist):
   - [ ] All 12 acceptance criteria verified
   - [ ] All smoke tests passed
   - [ ] 0 unsafe artifacts found

2. **Engineering sign-off:**
   - [ ] All 14 PRs merged
   - [ ] All repos on main with changes
   - [ ] No rollbacks needed

3. **Ops verification:**
   - [ ] No key leaks in logs/artifacts
   - [ ] MiniMax usage visible in dashboard
   - [ ] 0 security incidents

4. **Final approval:**
   - [ ] Project lead signs off
   - [ ] Phase 3 (optional) planned
   - [ ] Team notified in #droid-migration

---

## Troubleshooting During Execution

### Issue: Droid Doesn't Trigger on Smoke Test PR

**Diagnosis:**
```bash
# Check if safe action exists
gh release view 01e76b659e4b1e5f23feedc8cfabf8dc14c7485f \
  --repo EffortlessMetrics/droid-action-safe

# Check if workflows are enabled in repo
gh repo view EffortlessMetrics/$REPO --json hasDiscussionsEnabled
```

**Fix:**
- Verify safe action SHA is exactly: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Verify GitHub Actions are enabled in repo settings
- Check if PR is in same repo (not from fork)
- Verify `contents: write` permission is set

### Issue: MiniMax Model Not Used

**Diagnosis:**
```bash
# Check if secret exists
gh secret list --org EffortlessMetrics | grep MINIMAX_API_KEY

# Check workflow run logs
RUN_ID=$(gh run list --repo EffortlessMetrics/$REPO --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $RUN_ID --repo EffortlessMetrics/$REPO --log
```

**Fix:**
- Verify MINIMAX_API_KEY secret exists in repo
- Verify secret value is not empty
- Check settings.local.json heredoc syntax (must use single quotes)
- Verify API key is still valid with MiniMax

### Issue: YAML Syntax Error

**Diagnosis:**
```bash
yamllint .github/workflows/droid-review.yml
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/droid-review.yml'))"
```

**Fix:**
- Check indentation (must be 2 spaces, not tabs)
- Check heredoc quotes (must be `<<'JSON'` with single quotes)
- Check closing `JSON` is flush left (no indentation)
- Validate with yamllint before commit

### Issue: PR Review Takes Too Long

**Solution:**
- Reassign to active reviewer
- Post reminder in #droid-migration
- Merge after 24h if only minor feedback pending

---

## Command Cheat Sheet

### Quick Reference

```bash
# Clone repo
gh repo clone EffortlessMetrics/$REPO && cd $REPO

# Create branch
git checkout -b ci/safe-droid-action

# Find action refs
grep -rn "Factory-AI/droid-action" .github/workflows/

# Replace action (use exact command from Procedure D)
sed -i 's|Factory-AI/droid-action@main|EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f|' .github/workflows/droid-review.yml

# Add artifact flag (use exact command from Procedure E)
sed -i '21a\        upload_debug_artifacts: false' .github/workflows/droid-review.yml

# Validate YAML
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/droid-review.yml'))" && echo "✅ Valid"

# Commit
git add .github/workflows/ && git commit -m "ci: use safe Droid action"

# Push
git push -u origin ci/safe-droid-action

# Create PR
gh pr create --title "ci: use safe Droid action" --body "..."

# Merge
gh pr merge --repo EffortlessMetrics/$REPO --merge --delete-branch

# Smoke test
git checkout -b smoke-test && echo "test" >> README.md && git add . && git commit -m "smoke" && git push -u origin smoke-test && gh pr create --draft

# Delete test branch
gh pr delete smoke-test --yes && git push origin --delete smoke-test
```

---

## Final Checklist Before Starting

- [ ] Phase 0 complete (Ops sign-off)
- [ ] MINIMAX_API_KEY secret exists in all 7 repos
- [ ] FACTORY_API_KEY validated
- [ ] Safe action SHA memorized: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- [ ] Model name memorized: `custom:MiniMax-M2.7-0`
- [ ] All procedures understood (A through K)
- [ ] TRACKING_SHEET.md opened and ready
- [ ] Started with OpenRacing (not another repo)
- [ ] All 7 repos verified for access (Procedure: Repo Access Verification)

**If all checked, you are ready to execute Phase 1.**

Good luck! This runbook is self-contained and auditable at every step.
