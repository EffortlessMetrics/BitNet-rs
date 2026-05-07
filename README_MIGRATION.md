# EffortlessMetrics Factory Action Safe Migration

## Quick Start

**You are here:** Branch `claude/migrate-factory-action-uHHnB`  
**Status:** Documentation package complete  
**Ready for execution:** YES

**For quick info**, read this file. **For detailed work**, see the doc references below.

---

## What Is This Migration?

Move 7 EffortlessMetrics repositories from unsafe `Factory-AI/droid-action` references to safe `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f` with MiniMax M2.7 BYOK model.

**Why?** Upstream Factory action can upload raw `$HOME/.factory/**` and `droid-prompts/**`, which in BYOK mode includes resolved API keys.

**Safe action:?** `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f` — based on Factory v5, disables raw artifact upload.

---

## Repos in Scope

### Batch 1: High Priority (Mutable Refs)

| Repo | Current Ref | Your Action |
|------|-------------|-------------|
| OpenRacing | @main | See `IMPLEMENTATION_OPENRACING.md` |
| adze | @v5 | See `IMPLEMENTATION_ADZE.md` |
| SwiftMTP-dev | @main | See `IMPLEMENTATION_BATCH1.md` |
| SwiftMailSort | @main | See `IMPLEMENTATION_BATCH1.md` |
| shiplog | @main | See `IMPLEMENTATION_BATCH1.md` |

### Batch 2: Medium Priority (SHA-Pinned Refs)

| Repo | Current Ref | Your Action |
|------|-------------|-------------|
| perl-lsp | @<sha> | See `IMPLEMENTATION_BATCH2.md` |
| pkm-python | @<sha> | See `IMPLEMENTATION_BATCH2.md` |

---

## Document Map

**Read these in order:**

1. **README_MIGRATION.md** ← You are here
2. **MIGRATION_PLAN.md** — Strategic overview, phases, timelines, acceptance criteria
3. **EXECUTION_SUMMARY.md** — Sign-off checklist, roles, Phase 3 planning

**Then, by repo (pick your starting repo):**

4. **IMPLEMENTATION_OPENRACING.md** — Deep reference template (Phase 1 + 2 full details)
5. **IMPLEMENTATION_ADZE.md** — Batch 1 repo 2 specifics
6. **IMPLEMENTATION_BATCH1.md** — Unified template for repos 3–5
7. **IMPLEMENTATION_BATCH2.md** — Template for Batch 2 repos

---

## Roles & Quick Actions

### I'm the Ops/Security Team

**You own:** Phase 0 (key rotation)

**Action:**
1. Read: `MIGRATION_PLAN.md` → "Phase 0: Pre-Rollout Checklist"
2. Read: `EXECUTION_SUMMARY.md` → "Ops/Security Team" section
3. Complete:
   - [ ] Rotate exposed MiniMax Token Plan key
   - [ ] Update MINIMAX_API_KEY secret in GitHub
   - [ ] Confirm FACTORY_API_KEY valid
   - [ ] Scope MINIMAX_API_KEY to 7 repos only

**When done:** Notify Engineering that Phase 1 can begin

---

### I'm an Engineering Lead / Repository Maintainer

**You own:** Phase 1 + 2 (7 repos, ~3 weeks total)

**Action:**

1. **Read first:**
   - `MIGRATION_PLAN.md` → "Rollout Timeline"
   - `EXECUTION_SUMMARY.md` → "Engineering Lead" section

2. **Pick your starting repo:**
   - First repo: OpenRacing (use `IMPLEMENTATION_OPENRACING.md` as reference for all)
   - Next repo: adze (use `IMPLEMENTATION_ADZE.md` or reference OpenRacing)
   - Repos 3–5: SwiftMTP-dev, SwiftMailSort, shiplog (use `IMPLEMENTATION_BATCH1.md`)
   - Repos 6–7: perl-lsp, pkm-python (use `IMPLEMENTATION_BATCH2.md`)

3. **For each repo, do 2 PRs:**
   - **PR 1 (Phase 1):** Replace unsafe action, add safety flag
     - Title: `ci: use safe Droid action`
     - Steps: Read your implementation guide, follow checklist, create PR, merge, smoke test
   - **PR 2 (Phase 2):** Add MiniMax BYOK, guards, guidance
     - Title: `ci: align Droid review baseline`
     - Steps: Read implementation guide, follow checklist, create PR, merge, smoke test

4. **Timeline:**
   - Batch 1: 5 repos × 2 phases = 10 PRs (~2 weeks)
   - Batch 2: 2 repos × 2 phases = 4 PRs (~1 week)
   - Total: ~3 weeks

---

### I'm QA / Code Reviewer

**You own:** Validation of Phase 1 + 2 PRs

**Action:**

1. **Read first:**
   - `EXECUTION_SUMMARY.md` → "QA / Code Review" section

2. **For each Phase 1 PR, verify:**
   - [ ] Safe action SHA is correct: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
   - [ ] `upload_debug_artifacts: false` is set
   - [ ] Checkout pinned to SHA v5
   - [ ] No unrelated changes

3. **For each Phase 2 PR, verify:**
   - [ ] MiniMax BYOK heredoc syntax (single quotes around EOF)
   - [ ] Same-repo guard present in droid-review.yml
   - [ ] Trusted-actor guard present in droid.yml
   - [ ] Model: `custom:MiniMax-M2.7-0`
   - [ ] AGENTS.md and .factory/rules/ files added

---

## Quickest Way to Understand

**Read this order (20 min total):**

1. This file (README_MIGRATION.md) — 3 min
2. MIGRATION_PLAN.md → "Executive Summary" + "Rollout Timeline" — 5 min
3. EXECUTION_SUMMARY.md → "Acceptance Criteria for Phase 1 + 2 Completion" — 2 min
4. IMPLEMENTATION_OPENRACING.md → "Phase 1" section — 5 min
5. IMPLEMENTATION_OPENRACING.md → "Phase 2" section — 5 min

Then jump to the repo-specific implementation doc for your starting repo.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Repos in scope | 7 |
| Batches | 2 (5 + 2) |
| Phases per repo | 2 |
| PRs needed | 14 total (7 repos × 2 PRs) |
| Estimated duration | 3–4 weeks |
| Estimated effort per repo | 1–2 days |

---

## The 30-Second Summary

**Problem:** 7 repos use unsafe `Factory-AI/droid-action` that uploads raw secrets.

**Solution:** Replace with safe action + MiniMax BYOK model.

**Process:**
- Phase 1: Replace action, disable artifacts (1 PR per repo)
- Phase 2: Add MiniMax BYOK, guards, guidance (1 PR per repo)
- Phase 3: Move shared plumbing to org reusable workflows (future)

**Timeline:** ~3 weeks for Phases 1 + 2, starting after key rotation.

**Success:** All 7 repos use safe action, MiniMax BYOK, no artifacts leak.

---

## Validation Checklist: Complete Phase 1 + 2

After all 7 repos are done:

- [ ] All 7 repos use safe action SHA
- [ ] All 7 repos have `upload_debug_artifacts: false`
- [ ] All 7 repos have MiniMax BYOK setup
- [ ] All 7 repos use model: `custom:MiniMax-M2.7-0`
- [ ] All 7 repos have same-repo + trusted-actor guards
- [ ] All 7 repos have AGENTS.md and .factory/rules/droid-review.md
- [ ] 0 repos upload raw `droid-review-debug-<run_id>` artifacts
- [ ] Manual @droid review proven in 2+ repos
- [ ] MiniMax usage visible in provider dashboard
- [ ] No Factory-AI/droid-action direct refs remain in any repo

**When all checked:** Ready for sign-off and Phase 3 planning.

---

## FAQ

### Q: What happens if Phase 0 (key rotation) doesn't finish?

**A:** Phase 1 PRs cannot merge. They'll sit in review until Phase 0 is done. Ops team drives Phase 0.

### Q: Can I start Phase 2 before all Phase 1 repos are done?

**A:** Yes. Phase 2 for repo A can start while Phase 1 for repo B is in progress. But Batch 2 should wait for all Batch 1 Phase 1 to be safe.

### Q: What if a repo has custom Droid config?

**A:** Check your implementation guide for that repo. If unique, document it in AGENTS.md.

### Q: Will Droid stop working during the migration?

**A:** No. Phase 1 is a pure safety patch; behavior doesn't change. Phase 2 adds MiniMax BYOK and guides; reviews still work.

### Q: How do I test my changes before merging?

**A:** Create a draft PR in the target repo, observe Droid runs, check for artifacts. See "Smoke Test" sections in implementation guides.

### Q: What if I find a bug in the safe action during rollout?

**A:** File an issue in `EffortlessMetrics/droid-action-safe`. In the meantime, repos can roll back the Phase 1 PR.

### Q: Is Phase 3 (org reusable workflows) required?

**A:** No. Phase 1 + 2 make repos safe and consistent. Phase 3 is just convenience for future repos.

---

## Getting Unstuck

| Problem | Doc to Read |
|---------|------------|
| Don't know what Phase 1 does | IMPLEMENTATION_OPENRACING.md → "Phase 1" |
| Don't know what Phase 2 does | IMPLEMENTATION_OPENRACING.md → "Phase 2" |
| Droid doesn't trigger | IMPLEMENTATION_BATCH2.md → "Common Issues & Fixes" |
| MiniMax not in logs | IMPLEMENTATION_BATCH2.md → "Common Issues & Fixes" |
| YAML syntax error | IMPLEMENTATION_BATCH2.md → "Common Issues & Fixes" |
| General timeline question | MIGRATION_PLAN.md → "Rollout Timeline" |
| Sign-off requirements | EXECUTION_SUMMARY.md → "Sign-Off Checklist" |

---

## File Manifest

All files in this branch (`claude/migrate-factory-action-uHHnB`):

```
README_MIGRATION.md                   ← Start here
MIGRATION_PLAN.md                     ← Strategic plan, phases, timelines
IMPLEMENTATION_OPENRACING.md          ← Deep-dive reference template
IMPLEMENTATION_ADZE.md                ← Batch 1 repo 2 specifics
IMPLEMENTATION_BATCH1.md              ← Unified template for repos 3–5
IMPLEMENTATION_BATCH2.md              ← Batch 2 template
EXECUTION_SUMMARY.md                  ← Sign-off, roles, Phase 3
```

---

## Next Steps

1. **Share this package** with your team (link to this branch)
2. **Ops completes Phase 0** (key rotation, secret scoping)
3. **Engineering starts Phase 1** with OpenRacing
4. **QA validates each PR** against Phase 1 + 2 checklists
5. **After 3–4 weeks,** all repos are safe and converged
6. **Optional Phase 3:** Plan org reusable workflows

---

## Contact / Support

Questions? Issues? Escalations?

**Channel:** #droid-migration (or equivalent in your org)

**Include:** Repo name + PR link + specific error/blocker

---

## Version & Metadata

| Field | Value |
|-------|-------|
| Created | 2026-05-07 |
| Branch | `claude/migrate-factory-action-uHHnB` |
| Safe action SHA | `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f` |
| Reference impl | `EffortlessMetrics/ripr` PR #467 |
| Status | Ready for execution |
| Estimated completion | 4 weeks from Phase 0 completion |

---

**You are ready to begin. Pick your starting repo and dive into the implementation guide.**
