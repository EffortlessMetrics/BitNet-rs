# EffortlessMetrics Factory Action Migration: Execution Summary & Sign-Off

**Plan created:** 2026-05-07  
**Branch:** `claude/migrate-factory-action-uHHnB`  
**Status:** Documentation complete; ready for execution

---

## Document Structure

This migration is fully documented across 6 files:

| File | Purpose |
|------|---------|
| `MIGRATION_PLAN.md` | Strategic overview, phases, timelines, acceptance criteria |
| `IMPLEMENTATION_OPENRACING.md` | Deep-dive template for first repo (reference for all others) |
| `IMPLEMENTATION_ADZE.md` | Batch 1 repo #2 specifics |
| `IMPLEMENTATION_BATCH1.md` | Unified template for repos 3–5 (SwiftMTP-dev, SwiftMailSort, shiplog) |
| `IMPLEMENTATION_BATCH2.md` | Template for Batch 2 repos (perl-lsp, pkm-python) |
| `EXECUTION_SUMMARY.md` | This file; sign-off and Phase 3 reference |

---

## Quick Reference: Action Items by Role

### Ops/Security Team (Phase 0)

**Prerequisite before any PRs open:**

- [ ] Rotate exposed MiniMax Token Plan key
- [ ] Update `MINIMAX_API_KEY` GitHub secret in org/repo scope
- [ ] Confirm `FACTORY_API_KEY` is still valid
- [ ] Scope `MINIMAX_API_KEY` to pilot repos only (7 repos listed below)
- [ ] Document key rotation in #droid-migration or equivalent channel

**Sign-off:** When complete, notify Engineering that Phase 1 can begin.

### Engineering Lead / Repository Maintainers

**Phase 1 (Batches 1 + 2): Safety Patches**

For each of 7 repos in order (see execution timeline):

1. **Clone/checkout repo**
2. **Create Phase 1 PR** using template in `IMPLEMENTATION_[REPO].md`
   - Branch: `ci/safe-droid-action`
   - Replace `Factory-AI/droid-action` with safe SHA
   - Add `upload_debug_artifacts: false`
   - Review and merge
3. **Smoke test:**
   - Open draft PR
   - Verify Droid triggers, no debug artifacts
   - Close PR
4. **Repeat for next repo**

**Timeline:** ~10 days (7 repos × 1–2 days each for Phase 1)

**Phase 2 (Batches 1 + 2): Baseline Convergence**

After all Phase 1 repos merged:

For each of 7 repos in same order:

1. **Create Phase 2 PR** using template
   - Branch: `ci/droid-baseline`
   - Add MiniMax BYOK
   - Add guards, model inputs, guidance files
   - Review and merge
2. **Smoke test:**
   - Open draft PR
   - Verify MiniMax model used
   - Test manual @droid review (as OWNER/MEMBER)
   - Close PR
3. **Repeat for next repo**

**Timeline:** ~10 days (7 repos × 1–2 days each for Phase 2)

**Total Phase 1 + 2:** ~3 weeks

### QA / Code Review

**Per Phase 1 PR:**

- Verify safe action SHA is correct: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Verify `upload_debug_artifacts: false` is set
- Verify checkout is pinned to SHA v5
- Confirm no behavior changes unrelated to safety

**Per Phase 2 PR:**

- Verify MiniMax BYOK heredoc syntax (quoted EOF)
- Verify same-repo guard in droid-review.yml
- Verify trusted-actor guard in droid.yml
- Verify model inputs: `custom:MiniMax-M2.7-0`
- Verify AGENTS.md and .factory/rules/ are present and reasonable
- Confirm smoke test results before merge

---

## Execution Timeline

### Week 1: Phase 0 Prep + Batch 1 Phase 1

**Mon–Tue:** Ops completes Phase 0 key rotation  
**Wed–Fri:** Batch 1 Phase 1 PRs (OpenRacing, adze, first smoke tests)

### Week 2: Batch 1 Phase 1 Complete + Phase 2 Begins

**Mon–Tue:** SwiftMTP-dev, SwiftMailSort Phase 1 + smoke  
**Wed:** shiplog Phase 1 + smoke  
**Thu–Fri:** OpenRacing Phase 2 PR, review, merge, smoke

### Week 3: Batch 1 Phase 2 Complete + Batch 2 Begins

**Mon–Tue:** adze, SwiftMTP-dev Phase 2 smoke, merge  
**Wed–Thu:** SwiftMailSort, shiplog Phase 2 smoke, merge  
**Fri:** Batch 2 Phase 1 (perl-lsp) Phase 1 PR + merge + smoke

### Week 4: Batch 2 Complete

**Mon–Tue:** perl-lsp Phase 2 + pkm-python Phase 1  
**Wed–Thu:** pkm-python Phase 2  
**Fri:** Validation + sign-off

---

## Acceptance Criteria for Phase 1 + 2 Completion

### Mandatory (Blocker for Phase 3)

- ✅ All 7 repos use safe action SHA in both workflows
- ✅ All 7 repos have `upload_debug_artifacts: false`
- ✅ All 7 repos have same-repo guard in droid-review.yml
- ✅ All 7 repos have trusted-actor guard in droid.yml
- ✅ All 7 repos have MiniMax BYOK step (settings.local.json)
- ✅ All 7 repos have model inputs: `custom:MiniMax-M2.7-0`
- ✅ 0 repositories upload raw `droid-review-debug-<run_id>` artifacts
- ✅ 0 repositories reference `Factory-AI/droid-action` directly
- ✅ 0 repositories use mutable refs (@main, @v5)
- ✅ All Phase 1 and Phase 2 PRs merged

### Strongly Recommended (Success Validation)

- ✅ MiniMax usage visible in provider dashboard (7 repos)
- ✅ Manual @droid review proven in 2+ repos
- ✅ Manual @droid security proven in 1+ repo
- ✅ Scheduled security scan proven in 1+ repo (if enabled)
- ✅ All repos have AGENTS.md
- ✅ All repos have .factory/rules/droid-review.md

### Sign-Off Checklist

**Engineering Lead:**

- [ ] All 7 Phase 1 PRs merged
- [ ] All 7 Phase 2 PRs merged
- [ ] All smoke tests green
- [ ] 0 unsafe artifacts found
- [ ] MiniMax working in 7 repos
- [ ] Signatory: __________________ Date: __________

**Ops/Security:**

- [ ] Phase 0 key rotation complete
- [ ] MINIMAX_API_KEY scoped correctly
- [ ] FACTORY_API_KEY validated
- [ ] No key leaks in build logs/artifacts
- [ ] Signatory: __________________ Date: __________

**QA/Code Review:**

- [ ] 7 repos verified for safe action SHA
- [ ] 7 repos verified for BYOK convergence
- [ ] Smoke tests validated
- [ ] No behavior regressions observed
- [ ] Signatory: __________________ Date: __________

---

## Phase 3: Org Reusable Workflows (Future Planning)

**Timeline:** After Phase 1 + 2 sign-off, weeks 5–6

**Not required for pilot success**, but enables simpler onboarding for new repos.

### Scope: Reusable Workflows in `EffortlessMetrics/.github`

Create three reusable workflows that encapsulate shared plumbing:

#### 1. `.github/workflows/droid-review-reusable.yml`

Responsibilities:
- Checkout, MiniMax BYOK, action invocation
- Inputs: security thresholds (optional)
- Secrets: FACTORY_API_KEY, MINIMAX_API_KEY

Caller repo provides:
- `on:` triggers
- `permissions:` (usually `contents: write`, `pull-requests: write`, etc.)
- `if:` guards (same-repo check)
- `concurrency:` policy

#### 2. `.github/workflows/droid-tag-reusable.yml`

For manual `@droid` commands

#### 3. `.github/workflows/droid-security-scan-reusable.yml`

For scheduled + manual security scans

### Benefits

- **Centralized:** One place to update safe action SHA
- **Consistency:** All repos use same BYOK config
- **Maintenance:** Single source of truth for MiniMax setup
- **Onboarding:** New repos just call reusable workflow, own guards/triggers

### Example Caller Workflow

```yaml
# .github/workflows/droid-review.yml in target repo

name: Droid Auto Review

on:
  pull_request:
    types: [opened, synchronize, ready_for_review, reopened]

concurrency:
  group: droid-review-${{ github.repository }}-${{ github.event.pull_request.number }}
  cancel-in-progress: false

jobs:
  droid-review:
    if: |
      github.event.pull_request.head.repo.full_name == github.repository &&
      !contains(github.event.pull_request.title, '[skip-review]')

    permissions:
      contents: write
      pull-requests: write
      issues: write
      id-token: write
      actions: read

    uses: EffortlessMetrics/.github/.github/workflows/droid-review-reusable.yml@<pinned-sha>
    secrets:
      FACTORY_API_KEY: ${{ secrets.FACTORY_API_KEY }}
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

    with:
      security_severity_threshold: high
      security_block_on_critical: true
      security_block_on_high: false
```

### Phase 3 Execution (Not in Current Scope)

- [ ] Create reusable workflows in EffortlessMetrics/.github
- [ ] Test with 1 repo
- [ ] Migrate all 7 pilot repos to reusable
- [ ] Document in EffortlessMetrics/.github README
- [ ] Prepare for Batch 3 new installs (easy onboarding)

---

## Known Constraints & Assumptions

### Constraints

1. **Phase 0 must complete first:** Key rotation and secret scoping are blockers
2. **Same-repo PRs only:** Auto review requires `github.event.pull_request.head.repo.full_name == github.repository`
3. **Trusted actors only:** Manual @droid must gate on `author_association in [OWNER, MEMBER, COLLABORATOR]`
4. **No fork secret execution:** Even if PR is from fork, secrets don't leak because of same-repo guard
5. **Model not negotiable:** `custom:MiniMax-M2.7-0` is the baseline; deviations require escalation

### Assumptions

1. **Safe action SHA is stable:** `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f` will not change during rollout
2. **MiniMax API endpoint is stable:** `https://api.minimax.io/anthropic` is correct for all repos
3. **FACTORY_API_KEY is still valid:** No migration or expiration during Phase 1–2
4. **Repos have write access to workflows:** Maintainers can merge workflow PRs
5. **Droid action supports custom models:** MiniMax BYOK works with EffortlessMetrics/droid-action-safe

**If any assumption proves false, escalate to #droid-migration immediately.**

---

## Troubleshooting & Support

### Common Issues

**Q: Droid action doesn't trigger**
- Verify same-repo guard is correct
- Verify PR is in same repo (not fork)
- Check GitHub Actions permissions: `contents: write`

**Q: MiniMax model not used**
- Verify MINIMAX_API_KEY secret exists in repo
- Check settings.local.json heredoc syntax
- Look for errors in workflow run logs

**Q: Safe action SHA doesn't exist**
- Verify correct SHA: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Verify org: `EffortlessMetrics` (not typo)
- Confirm GitHub repo: `EffortlessMetrics/droid-action-safe`

**Q: Raw debug artifact uploaded**
- Verify `upload_debug_artifacts: false` is set
- Confirm action is safe ref, not Factory-AI upstream
- Check artifact list in workflow run for `droid-review-debug-<run_id>`

### Escalation

- **Phase 0 blockers:** Ops/Security team
- **Workflow syntax:** Review IMPLEMENTATION_[REPO].md templates
- **Model integration:** Check MiniMax dashboard for API errors
- **Unexpected behavior:** Post issue with repo + PR link in #droid-migration

---

## Documentation Handoff

This package includes:

1. **Strategic plan** (`MIGRATION_PLAN.md`) — Timeline, phases, risks
2. **Reference implementation** (`IMPLEMENTATION_OPENRACING.md`) — Deep-dive template
3. **Batch templates** (`IMPLEMENTATION_BATCH1.md`, `IMPLEMENTATION_BATCH2.md`) — Copies to 6 more repos
4. **Execution guide** (this file) — Sign-off, timelines, roles
5. **Phase 3 planning** (below) — Future org-level consolidation

**All documents are in this branch** (`claude/migrate-factory-action-uHHnB`), ready to be merged to main and shared with team.

---

## Phase 3 Planning: Org Reusable Workflows

(After Phase 1 + 2 Sign-Off)

### Architecture

```
EffortlessMetrics/.github/
├── .github/workflows/
│   ├── droid-review-reusable.yml
│   ├── droid-tag-reusable.yml
│   └── droid-security-scan-reusable.yml
└── README.md
    └── Caller workflow templates
```

### Caller Repos Use Pattern

Each target repo has thin wrapper workflows that:

```yaml
jobs:
  droid-review:
    if: <same-repo-guard>
    permissions: <repo-specific>
    uses: EffortlessMetrics/.github/.github/workflows/droid-review-reusable.yml@<sha>
    secrets:
      FACTORY_API_KEY: ${{ secrets.FACTORY_API_KEY }}
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
    with:
      security_severity_threshold: high  # optional
```

### Benefits Over Inline Workflows

| Aspect | Current (Inline) | Phase 3 (Reusable) |
|--------|------------------|-------------------|
| BYOK update | Edit 7 repos | Edit 1 org repo |
| Action SHA update | Edit 7 repos | Edit 1 org repo |
| New repo onboarding | Copy full workflow | Inherit reusable |
| Debugging | 7 copies, drift risk | Single source |

### Reusable Workflow Spec (Draft)

**Inputs:**
- `security_severity_threshold` (string, default: "high")
- `security_block_on_critical` (bool, default: true)
- `security_block_on_high` (bool, default: false)

**Secrets:**
- `FACTORY_API_KEY`
- `MINIMAX_API_KEY`

**Outputs:**
- (None; reviews are posted directly)

**Assumptions:**
- Caller handles `on:` triggers, `if:` guards, `concurrency`, `permissions`
- Caller owns repo-local guidance files (AGENTS.md, .factory/rules/)
- Caller responsible for MINIMAX_API_KEY secret scope

---

## Next Steps

### Immediate (This Week)

1. **Distribute this package** to:
   - Ops/Security team (Phase 0 owners)
   - Repository maintainers (Phase 1–2 owners)
   - QA/Code review team (validation owners)

2. **Schedule Phase 0 completion:** Ops rotates keys, scopes secrets

3. **Prepare Phase 1 launch:** Engineering reviews IMPLEMENTATION_[REPO].md files

### Week 1–4

Execute Phases 1 and 2 per timeline in `MIGRATION_PLAN.md`

### Week 5+ (Phase 3)

Plan org reusable workflows based on Phase 1–2 success

---

## Sign-Off Template

Copy this to #droid-migration or issue when ready:

```
## Factory Action Migration: Phase 1 + 2 Sign-Off

**Date:** [YYYY-MM-DD]

### Phase 0 (Ops)
- [x] Key rotation complete
- [x] Secrets scoped correctly
- **Signatory:** [Name] | **Date:** [YYYY-MM-DD]

### Phase 1 + 2 (Engineering)
- [x] All 7 repos patched (safe action SHA)
- [x] All 7 repos converged (MiniMax BYOK)
- [x] All smoke tests green
- [x] 0 unsafe artifacts
- **Signatory:** [Name] | **Date:** [YYYY-MM-DD]

### Validation (QA)
- [x] Safe action verified
- [x] BYOK convergence verified
- [x] Manual @droid proven in 2+ repos
- **Signatory:** [Name] | **Date:** [YYYY-MM-DD]

### Approval (Lead)
Ready to proceed to Phase 3 (org reusable workflows).
**Signatory:** [Name] | **Date:** [YYYY-MM-DD]
```

---

## Final Notes

This migration is **non-breaking** and **reversible**:

- Phase 1 PRs are pure safety patches; no behavior change
- Phase 2 PRs add MiniMax BYOK and guards; existing workflows continue to work
- Each repo can be reverted independently if issues arise
- Org-level Phase 3 is optional; repos work fine with inline workflows

**The goal is safe, consistent, org-standard Droid review across 7 repos, proven by smoke tests, then extended to new repos via reusable workflows.**

---

## Document Inventory

All files are in the `claude/migrate-factory-action-uHHnB` branch:

- ✅ MIGRATION_PLAN.md (strategic overview)
- ✅ IMPLEMENTATION_OPENRACING.md (reference deep-dive)
- ✅ IMPLEMENTATION_ADZE.md (Batch 1 repo 2)
- ✅ IMPLEMENTATION_BATCH1.md (unified Batch 1 template for 3 more repos)
- ✅ IMPLEMENTATION_BATCH2.md (Batch 2 template for 2 repos)
- ✅ EXECUTION_SUMMARY.md (this file; sign-off + Phase 3)

**Ready to share with team and begin execution.**
