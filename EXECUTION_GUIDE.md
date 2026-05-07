# EffortlessMetrics Factory Action Migration: Execution Guide

**Status:** ✅ FULLY DOCUMENTED & READY FOR EXECUTION  
**Branch:** `claude/migrate-factory-action-uHHnB`  
**Format:** Detailed step-by-step runbooks (manual execution)  
**Date:** 2026-05-07

---

## 📋 Documentation Package Contents

### Strategic & Planning Documents

| Document | Purpose | Read Time | When |
|----------|---------|-----------|------|
| `README_MIGRATION.md` | Quick-start overview for all roles | 3 min | Start here |
| `MIGRATION_PLAN.md` | Strategic phases, timeline, acceptance criteria | 10 min | Before Phase 1 |
| `EXECUTION_SUMMARY.md` | Roles, sign-off checklist, Phase 3 planning | 10 min | Before starting |

### Detailed Runbooks (Step-by-Step Execution)

| Document | Purpose | Format | Repos | Time/Repo |
|----------|---------|--------|-------|-----------|
| `RUNBOOK_MASTER.md` | Common procedures (A-K) and pre-flight checklist | Procedures | All 7 | One-time |
| `RUNBOOK_PHASE1.md` | Step-by-step Phase 1 safety patches | 13 steps | All 7 | 1-2 days |
| `RUNBOOK_PHASE2.md` | Step-by-step Phase 2 baseline convergence | 14 steps | All 7 | 2-3 days |

### Implementation & Tracking

| Document | Purpose | Format | Use For |
|----------|---------|--------|---------|
| `TRACKING_SHEET.md` | Execution progress across all 7 repos | Checklist | Track progress |
| `IMPLEMENTATION_OPENRACING.md` | Deep reference (Phase 1 + 2 details) | Reference | Understanding patterns |
| `IMPLEMENTATION_BATCH1.md` | Unified template for 3 repos | Template | Reference |
| `IMPLEMENTATION_BATCH2.md` | Template for 2 repos | Template | Reference |

**Total:** 11 documents, ~5,400 lines, fully self-contained

---

## 🚀 How to Proceed: Step by Step

### Week 0: Preparation (Your Org)

**Phase 0: Ops/Security (1 day)**

1. **Rotate MiniMax key**
   - [ ] Compromised key identified and rotated
   - [ ] New MINIMAX_API_KEY created in GitHub org secrets
   - [ ] Secret scoped to exactly these 7 repos (no more, no less)

2. **Validate credentials**
   - [ ] FACTORY_API_KEY verified (not expired)
   - [ ] MINIMAX_API_KEY verified (valid, accessible)

3. **Sign off Phase 0**
   - [ ] Ops team confirms in channel or issue
   - [ ] Engineering can now start Phase 1

---

### Week 1+: Execution (Engineering)

#### Phase 1: Safety Patches (2 weeks)

**Documents to use:**
- `RUNBOOK_MASTER.md` — Read "Prerequisites" and "Common Procedures"
- `RUNBOOK_PHASE1.md` — Execute for each repo in sequence

**Sequence (in order):**
1. OpenRacing (1-2 days)
2. adze (1-2 days)
3. SwiftMTP-dev (1-2 days)
4. SwiftMailSort (1-2 days)
5. shiplog (1-2 days)
6. perl-lsp (1-2 days)
7. pkm-python (1-2 days)

**Per repo:**
- [ ] Step 1: Clone
- [ ] Step 2: Examine current workflows
- [ ] Step 3: Create branch (ci/safe-droid-action)
- [ ] Step 4: Edit droid-review.yml
- [ ] Step 5: Edit droid.yml
- [ ] Step 6: Review changes
- [ ] Step 7: Commit
- [ ] Step 8: Push
- [ ] Step 9: Create PR
- [ ] Step 10: Code review & approval (24 hours)
- [ ] Step 11: Merge
- [ ] Step 12: Smoke test
- [ ] Step 13: Mark complete

**Code reviewer checklist (per Phase 1 PR):**
- [ ] Safe action SHA: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- [ ] `upload_debug_artifacts: false` present
- [ ] Checkout pinned to SHA
- [ ] No unrelated changes
- [ ] YAML is valid

#### Phase 2: Baseline Convergence (2-3 weeks)

**Documents to use:**
- `RUNBOOK_PHASE2.md` — Execute for each repo in same sequence

**Same sequence as Phase 1** (OpenRacing → adze → ... → pkm-python)

**Per repo:**
- [ ] Step 1: Verify Phase 1 merged & clone
- [ ] Step 2: Create branch (ci/droid-baseline)
- [ ] Step 3: Edit droid-review.yml (BYOK + guards)
- [ ] Step 4: Edit droid.yml (guards + BYOK)
- [ ] Step 5: Create AGENTS.md
- [ ] Step 6: Create .factory/rules/droid-review.md
- [ ] Step 7: Review all changes
- [ ] Step 8: Commit
- [ ] Step 9: Push
- [ ] Step 10: Create PR
- [ ] Step 11: Code review & approval (24 hours)
- [ ] Step 12: Merge
- [ ] Step 13: Smoke test (MiniMax + manual @droid)
- [ ] Step 14: Mark complete

**Code reviewer checklist (per Phase 2 PR):**
- [ ] MINIMAX_API_KEY env var in both workflows
- [ ] Same-repo guard in droid-review.yml
- [ ] Trusted-actor guard in droid.yml
- [ ] Model: `custom:MiniMax-M2.7-0` in both workflows
- [ ] MiniMax BYOK heredoc (single quotes)
- [ ] review_depth: shallow
- [ ] show_full_output: false
- [ ] contents: read in droid.yml
- [ ] AGENTS.md created
- [ ] .factory/rules/droid-review.md created
- [ ] YAML is valid

---

### Week 4+: Validation & Sign-Off

**Documents to use:**
- `TRACKING_SHEET.md` — Mark all complete
- `EXECUTION_SUMMARY.md` → "Acceptance Criteria" section

**Validation checklist (all 7 repos):**

| Criterion | Status |
|-----------|--------|
| All 7 repos use safe action SHA | ☐ |
| All 7 repos have upload_debug_artifacts: false | ☐ |
| All 7 repos have MiniMax BYOK | ☐ |
| All 7 repos use custom:MiniMax-M2.7-0 | ☐ |
| All 7 repos have same-repo guard | ☐ |
| All 7 repos have trusted-actor guard | ☐ |
| All 7 repos have AGENTS.md | ☐ |
| All 7 repos have .factory/rules/droid-review.md | ☐ |
| 0 repos upload raw droid-review-debug artifacts | ☐ |
| Manual @droid proven in 2+ repos | ☐ |
| MiniMax visible in provider dashboard | ☐ |
| No Factory-AI/droid-action refs remain | ☐ |

**Sign-off:**
- [ ] Ops verifies: No key leaks, MiniMax dashboard shows usage
- [ ] Engineering verifies: All 7 repos complete, all PRs merged
- [ ] QA verifies: All 12 acceptance criteria met, smoke tests passed
- [ ] Lead approves: Proceeds to sign-off documentation

---

## 📖 Using the Runbooks

### Runbook Format

Each runbook is structured as:
1. **Overview:** Objective, scope, time estimate
2. **Per-repo template:** 13-14 detailed steps
3. **Each step contains:**
   - Step description
   - Bash script (copy-paste ready)
   - Expected output
   - Next action

### How to Execute

**Option A: Copy-Paste Scripts**

```bash
# From RUNBOOK_PHASE1.md, Step 1:
#!/bin/bash
set -e

REPO="OpenRacing"
# ... (copy entire script)
```

Run each step in sequence, verify expected output before proceeding.

**Option B: Follow Manual Instructions**

Each step has a bash script AND manual steps. Follow either:
- Scripts: Fast, automated, less thinking
- Manual: Slower, auditable, more control

**Option C: One-Liner (Experienced Users)**

Each runbook includes a "one-liner" for each phase that runs all steps for one repo in sequence.

### Reading the Runbooks

| Want to... | Read... |
|-----------|---------|
| Understand what a procedure does | RUNBOOK_MASTER.md → Procedures A-K |
| Execute Phase 1 for a repo | RUNBOOK_PHASE1.md → Steps 1-13 |
| Execute Phase 2 for a repo | RUNBOOK_PHASE2.md → Steps 1-14 |
| Check expected output | Any runbook → "Expected output" section |
| See what a code reviewer checks | Any runbook → "Code Reviewer Checklist" section |
| Troubleshoot an issue | RUNBOOK_MASTER.md → "Troubleshooting During Execution" |

---

## ✅ Quick Reference: Key Commands

### Phase 1 Commands

```bash
# Clone repo
gh repo clone EffortlessMetrics/$REPO && cd $REPO

# Create branch
git checkout -b ci/safe-droid-action

# Replace action refs
sed -i "s|Factory-AI/droid-action@[^#]*|EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f|g" .github/workflows/droid*.yml

# Add artifact flag (find the line number first)
FACTORY_LINE=$(grep -n "factory_api_key" .github/workflows/droid-review.yml | head -1 | cut -d: -f1)
sed -i "${FACTORY_LINE}a\        upload_debug_artifacts: false" .github/workflows/droid-review.yml

# Commit and push
git add .github/workflows/ && git commit -m "ci: use safe Droid action"
git push -u origin ci/safe-droid-action

# Create PR
gh pr create --title "ci: use safe Droid action" --body "..."
```

### Phase 2 Commands

```bash
# Clone fresh main
gh repo clone EffortlessMetrics/$REPO && cd $REPO

# Create branch
git checkout -b ci/droid-baseline

# Add MiniMax BYOK step (complex, use runbook instead)
# ... see RUNBOOK_PHASE2.md Step 3

# Create guidance files
cat > AGENTS.md << 'EOF'
... (see RUNBOOK_PHASE2.md Step 5)
EOF

mkdir -p .factory/rules
cat > .factory/rules/droid-review.md << 'EOF'
... (see RUNBOOK_PHASE2.md Step 6)
EOF

# Commit and push
git add .github/workflows/ AGENTS.md .factory/ && git commit -m "ci: align Droid review baseline"
git push -u origin ci/droid-baseline

# Create PR
gh pr create --title "ci: align Droid review baseline" --body "..."
```

---

## 🎯 Success Metrics

### Per Repo (Phase 1)
- ✅ 1 PR merged
- ✅ Safe action SHA applied
- ✅ Artifact upload disabled
- ✅ Smoke test passed
- ✅ No unsafe artifacts

### Per Repo (Phase 2)
- ✅ 1 PR merged
- ✅ MiniMax BYOK configured
- ✅ Guards in place
- ✅ Guidance files created
- ✅ Smoke test with MiniMax passed
- ✅ Manual @droid works

### Overall (All 7 Repos)
- ✅ 14 PRs merged
- ✅ 12 acceptance criteria met
- ✅ 0 unsafe artifacts
- ✅ MiniMax usage visible
- ✅ Phase 3 ready (optional)

---

## 📞 Support During Execution

### If Something Goes Wrong

1. **Immediate action:** Check RUNBOOK_MASTER.md → "Troubleshooting During Execution"

2. **Common issues:**
   - Droid doesn't trigger? → Check same-repo guard, GitHub Actions enabled
   - MiniMax not used? → Check MINIMAX_API_KEY secret, heredoc syntax
   - YAML error? → Check indentation, quotation marks, heredoc format
   - PR stuck in review? → Merge with override after 24h if minor feedback

3. **Escalation:** Post in channel with:
   - Repo name
   - PR number
   - Error message
   - What you already tried

### Getting Help with Runbooks

| Question | Read... |
|----------|---------|
| What does Procedure A do? | RUNBOOK_MASTER.md → Procedure A |
| How do I execute Phase 1 Step 5? | RUNBOOK_PHASE1.md → Step 5 |
| What's expected output for Step 8? | RUNBOOK_PHASE1.md → Step 8 → Expected output |
| What does a code reviewer check? | RUNBOOK_PHASE1.md → "Code Reviewer Checklist" |

---

## 🗓️ Timeline Summary

| Phase | Duration | Repos | PRs | Total |
|-------|----------|-------|-----|-------|
| **Phase 0** (Ops) | 1 day | N/A | N/A | 1 day |
| **Phase 1** (Eng) | ~2 weeks | 7 | 7 | ~2 weeks |
| **Phase 2** (Eng) | ~2-3 weeks | 7 | 7 | ~2-3 weeks |
| **Validation** (QA) | 1 day | 7 | 14 | 1 day |
| **Total** | | | | **~4 weeks** |

**Parallel execution possible:** Phase 2 can start for Repo A while Phase 1 is finishing for Repo C.

---

## 🎬 Getting Started NOW

### For Engineering Lead

1. **Read first (15 minutes):**
   - [ ] README_MIGRATION.md (3 min)
   - [ ] MIGRATION_PLAN.md → "Executive Summary" (5 min)
   - [ ] RUNBOOK_MASTER.md → "Prerequisites" (5 min)

2. **Verify Phase 0 complete:**
   - [ ] Check with Ops team: "MiniMax key rotated? MINIMAX_API_KEY secret created?"
   - [ ] If not, wait before starting Phase 1

3. **Start Phase 1 with OpenRacing:**
   - [ ] Open RUNBOOK_PHASE1.md
   - [ ] Follow Step 1: Clone and verify access
   - [ ] Follow Step 2: Examine workflows
   - [ ] Continue through all 13 steps
   - [ ] Estimated time: 4-6 hours + 24h review time

4. **Move to next repo (adze):**
   - [ ] Repeat same runbook, just change REPO name
   - [ ] Time saving: Follow the same procedure patterns

---

## 📚 Full Document List (11 Total)

**Strategic:**
- README_MIGRATION.md
- MIGRATION_PLAN.md
- EXECUTION_SUMMARY.md
- EXECUTION_GUIDE.md (this file)

**Runbooks:**
- RUNBOOK_MASTER.md
- RUNBOOK_PHASE1.md
- RUNBOOK_PHASE2.md

**Reference:**
- IMPLEMENTATION_OPENRACING.md
- IMPLEMENTATION_ADZE.md
- IMPLEMENTATION_BATCH1.md
- IMPLEMENTATION_BATCH2.md

**Tracking:**
- TRACKING_SHEET.md

---

## ✨ Why These Runbooks?

**Safe:**
- Every step is auditable
- Expected output is shown
- No surprises
- Easy to rollback

**Complete:**
- 13-14 steps per phase
- Copy-paste scripts included
- Checklists for reviewers
- Troubleshooting section

**Flexible:**
- Can run steps manually or via scripts
- Can run per-repo or all at once
- Can adapt to your team's workflow
- No special tools required (just `gh`, `git`, `python3`)

**Auditable:**
- Each PR is independent
- Each step can be verified
- Changes are reviewable
- Progress is trackable

---

## 🚀 You're Ready

Everything is documented, step-by-step, copy-paste ready.

**Next action:** Start Phase 1 with OpenRacing using RUNBOOK_PHASE1.md

**Questions?** Check RUNBOOK_MASTER.md → "Troubleshooting During Execution"

**Stuck?** Reread the step, check expected output, verify your input

**Ready to start?** Let's go! 🎯
