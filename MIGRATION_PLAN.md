# EffortlessMetrics Factory Action Safe Migration Plan

**Status:** In Progress  
**Branch:** `claude/migrate-factory-action-uHHnB`  
**Date:** 2026-05-07

## Executive Summary

Migrate EffortlessMetrics repositories from unsafe `Factory-AI/droid-action` to `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f` with MiniMax BYOK baseline. Reference implementation: `ripr` (PR #467).

**Safe action SHA:** `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`

---

## Phase 0: Pre-Rollout Checklist

**Status:** ⚠️ NOT YET EXECUTED

- [ ] Rotate exposed MiniMax Token Plan key
- [ ] Update MINIMAX_API_KEY in GitHub org/repo secrets
- [ ] Confirm FACTORY_API_KEY is still valid
- [ ] Verify MINIMAX_API_KEY scoped only to pilot repos

**Blockers:** None identified yet. Assumption: Phase 0 is handled by ops team before phase 1 repos begin.

---

## Repository Inventory

### Batch 1: Mutable Action Refs (Priority)

Highest drift risk — `@main` or `@v5` refs.

| Repo | Current Ref | State | Risk | Phase 1 PR | Phase 2 PR |
|------|-------------|-------|------|-----------|-----------|
| OpenRacing | `Factory-AI/droid-action@main` | ❌ Unsafe | HIGH | Pending | Pending |
| adze | `Factory-AI/droid-action@v5` | ❌ Unsafe | HIGH | Pending | Pending |
| SwiftMTP-dev | `Factory-AI/droid-action@main` | ❌ Unsafe | HIGH | Pending | Pending |
| SwiftMailSort | `Factory-AI/droid-action@main` | ❌ Unsafe | HIGH | Pending | Pending |
| shiplog | `Factory-AI/droid-action@main` | ❌ Unsafe | HIGH | Pending | Pending |

### Batch 2: Pinned Upstream Refs

Lower drift risk (SHA-pinned) but still unsafe BYOK behavior.

| Repo | Current Ref | State | Risk | Phase 1 PR | Phase 2 PR |
|------|-------------|-------|------|-----------|-----------|
| perl-lsp | SHA-pinned Factory-AI | ❌ Unsafe | MEDIUM | Pending | Pending |
| pkm-python | SHA-pinned Factory-AI | ❌ Unsafe | MEDIUM | Pending | Pending |

### Reference: Already Safe

| Repo | Ref | State | Notes |
|------|-----|-------|-------|
| ripr | `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f` | ✅ Safe | PR #467; baseline |

---

## Rollout Timeline

### Phase 1: Emergency Safety Patch (Batch 1 → Batch 2)

**Goal:** Replace direct `Factory-AI/droid-action` refs; disable raw artifact upload.

**Per-repo scope:**
- Replace action with safe SHA
- Add `upload_debug_artifacts: false`
- Pin checkout to SHA
- No behavior change unless safety-critical

**PR title:** `ci: use safe Droid action`

**PR count:** 1 per repo (7 repos × 1 PR = 7 PRs)

**Timeline:** Sequential (one repo per day for safety review)

**Validation:**
- [ ] Repo workflow checks pass
- [ ] Same-repo smoke PR succeeds
- [ ] No `droid-review-debug-<run_id>` artifact uploaded

**Repos in order:**
1. OpenRacing (mutable `@main`)
2. adze (mutable `@v5`)
3. SwiftMTP-dev (mutable `@main`)
4. SwiftMailSort (mutable `@main`)
5. shiplog (mutable `@main`)
6. perl-lsp (pinned SHA)
7. pkm-python (pinned SHA)

---

### Phase 2: Baseline Convergence (Batch 1 → Batch 2)

**Goal:** Align to `ripr` baseline: MiniMax BYOK, model inputs, guards, guidance.

**Per-repo scope:**
- MiniMax BYOK settings.local.json
- Model: `custom:MiniMax-M2.7-0`
- Same-repo guard for auto review
- Trusted-actor guard for @droid
- Security scan workflow (if pilot)
- Repo-local review guidance

**PR title:** `ci: align Droid review baseline`

**PR count:** 1 per repo (may combine with Phase 1 for small repos)

**Timeline:** After Phase 1 is merged

**Validation:**
- [ ] Workflow checks pass
- [ ] Same-repo smoke PR with MiniMax
- [ ] Manual @droid review works
- [ ] No raw debug artifacts

---

### Phase 3: Reusable Workflows (Org-Level)

**Goal:** Move shared plumbing to `EffortlessMetrics/.github`

**Scope:**
- `.github/workflows/droid-review-reusable.yml`
- `.github/workflows/droid-tag-reusable.yml`
- `.github/workflows/droid-security-scan-reusable.yml`

**Timeline:** After 3+ repos prove baseline in Phase 2

**Not in scope yet** — documented for reference.

---

## Static Validation: Current State Audit

### Required Searches (per repo)

```bash
rg "Factory-AI/droid-action|droid-action@main|droid-action@v5|upload_debug_artifacts: true|show_full_output: true" .github docs .factory AGENTS.md
```

**Expected findings:**
- Mutable refs: `@main` or `@v5`
- Missing `upload_debug_artifacts: false`
- Missing MiniMax BYOK or wrong model
- Missing guards/guidance

### Batch 1 Audit Results

✅ **Completed:** All 5 repos confirmed to have Droid workflows.

| Repo | Files | Status | Notes |
|------|-------|--------|-------|
| OpenRacing | droid.yml, droid-review.yml | ❌ Unsafe | Contains Factory-AI/droid-action refs |
| adze | droid.yml, droid-review.yml | ❌ Unsafe | Contains Factory-AI/droid-action refs |
| SwiftMTP-dev | droid.yml, droid-review.yml | ❌ Unsafe | Contains Factory-AI/droid-action refs |
| SwiftMailSort | droid.yml, droid-review.yml | ❌ Unsafe | Contains Factory-AI/droid-action refs |
| shiplog | droid.yml, droid-review.yml | ❌ Unsafe | Contains Factory-AI/droid-action refs |

**Audit method:** GitHub code search confirmed `Factory-AI/droid-action` present in all 5 repos.

**Detailed content audit:** Requires per-repo examination (see implementation packages below).

### Batch 2 Audit Results

✅ **Completed:** Both repos confirmed to have Droid workflows.

| Repo | Files | Status | Notes |
|------|-------|--------|-------|
| perl-lsp | droid.yml, droid-review.yml | ❌ Unsafe | Likely SHA-pinned; requires manual check |
| pkm-python | droid.yml, droid-review.yml | ❌ Unsafe | Likely SHA-pinned; requires manual check |

**Audit method:** GitHub code search confirmed workflow presence.

**Next step:** Manual content inspection for ref pinning and BYOK status.

---

## Phase 1 Implementation: Safety Patch Template

### Checklist per repo

- [ ] Audit current workflows
- [ ] Identify all `Factory-AI/droid-action` uses
- [ ] Clone/checkout target repo
- [ ] Create branch: `ci/safe-droid-action`
- [ ] Replace action refs with safe SHA
- [ ] Add `upload_debug_artifacts: false`
- [ ] Pin checkout SHA
- [ ] Run static checks (e.g., `cargo xtask check-workflows`)
- [ ] Open Phase 1 PR
- [ ] Merge Phase 1 PR
- [ ] Open same-repo draft PR for smoke test
- [ ] Confirm no raw debug artifacts uploaded
- [ ] Close smoke PR
- [ ] Mark repo Phase 1 ✅ Complete

### Phase 1 PR Template

```markdown
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
```

---

## Phase 2 Implementation: Baseline Convergence Template

### Checklist per repo

- [ ] Create branch: `ci/droid-baseline`
- [ ] Add MiniMax BYOK settings.local.json step
- [ ] Add model inputs: `custom:MiniMax-M2.7-0`
- [ ] Add same-repo guard (auto review only)
- [ ] Add trusted-actor guard (@droid only)
- [ ] Add security scan workflow if in pilot
- [ ] Create/update repo-local guidance:
  - `AGENTS.md` (high-level)
  - `.factory/skills/review-guidelines/SKILL.md` (Droid rules)
  - `.factory/rules/droid-review.md` (strict rules)
  - `docs/agent-context/review-invariants.md` (edge cases)
  - `docs/agent-context/droid-smoke-tests.md` (validation)
- [ ] Run static checks
- [ ] Open Phase 2 PR
- [ ] Merge Phase 2 PR
- [ ] Open same-repo draft PR for smoke test (with @droid commands)
- [ ] Confirm MiniMax model used in provider dashboard
- [ ] Confirm clean-review inspection record format
- [ ] Close smoke PR
- [ ] Mark repo Phase 2 ✅ Complete

### Minimal Repo-Local Guidance

For small repos, minimum files:

**AGENTS.md** (100 lines max):
```markdown
# Droid Review Configuration

This repository uses Factory Droid for automated code review with MiniMax M2.7.

## Review Rules

- No naked LGTM comments
- Findings must be repair packets with failure mode, fix direction, validation
- Clean reviews include inspection record with observed/reported/not-verified
- No extra @mentions in Droid-generated bodies
- Evidence split by provenance

## Triggers

- Auto-review: Same-repo PRs, auto-triggered on open/sync
- Manual: `@droid review` or `@droid security` (OWNER/MEMBER/COLLABORATOR only)
- Scheduled: Weekly security scan
```

**`.factory/skills/review-guidelines/SKILL.md`**:
```markdown
# Droid Review Guidelines

Standard Droid review shape:

## Finding Format

[P0|P1|P2] Short title
Failure mode: ...
Why here: ...
Fix direction: ...
Validation: ...
Confidence: ...

## Clean Review Format

No actionable findings emitted.

Inspected surfaces: ...
Checks performed: ...
Why no comments: ...
Residual risk: ...
Validation signal:
  Observed: ...
  Reported: ...
  Not verified: ...
```

---

## Success Metrics

### Phase 1: Safety (Must-Have)

- ✅ All 7 repos use safe action SHA
- ✅ All 7 repos have `upload_debug_artifacts: false`
- ✅ No raw `droid-review-debug-<run_id>` artifacts uploaded
- ✅ All Phase 1 PRs merged

### Phase 2: Baseline (Must-Have for Broad Rollout)

- ✅ 5+ repos have MiniMax BYOK
- ✅ 5+ repos have `custom:MiniMax-M2.7-0` model inputs
- ✅ 5+ repos have same-repo + trusted-actor guards
- ✅ 2+ repos with proven manual @droid review
- ✅ 2+ repos with proven manual @droid security
- ✅ 1+ repo with proven scheduled security scan
- ✅ MiniMax usage visible in provider dashboard
- ✅ No repo uses `Factory-AI/droid-action` directly

### Acceptance Criteria for Full Rollout

Before Phase 3 (org reusable workflows):
- ✅ `ripr` safe action smoke is green post key-rotation
- ✅ 3+ pilot repos use safe action SHA
- ✅ 0 pilot repos upload raw Droid debug artifacts
- ✅ Manual @droid works in 2+ repos
- ✅ Manual @droid security works in 2+ repos
- ✅ Scheduled security scan works in 1+ repo
- ✅ MiniMax visible and expected usage
- ✅ All repos Phase 2 converged
- ✅ No mutable action refs remain

---

## Execution Order (Detailed)

### Round 1: Batch 1 Proof of Concept (Days 1–3)

**Target:** OpenRacing + adze

1. **Day 1: OpenRacing Phase 1**
   - Audit current workflows
   - Create Phase 1 PR
   - Merge after approval
   - Open smoke PR
   - Validate artifact behavior
   - Close smoke PR

2. **Day 2: adze Phase 1**
   - Repeat Day 1 steps

3. **Day 3: OpenRacing Phase 2**
   - Create Phase 2 PR (MiniMax BYOK, guidance)
   - Merge after approval
   - Smoke test with manual @droid
   - Validate MiniMax model used

### Round 2: Batch 1 Remaining (Days 4–7)

Repeat for SwiftMTP-dev, SwiftMailSort, shiplog in sequence.

### Round 3: Batch 2 (Days 8–10)

perl-lsp, pkm-python (Phase 1 + 2 combined if low-risk)

### Round 4: Validation & Sign-Off (Day 11+)

- Verify all repos Phase 1 complete
- Verify 3+ repos Phase 2 complete
- Document successful MiniMax usage
- Prepare Phase 3 planning (org reusable workflows)

---

## Critical Constraints

1. **No Phase 0 assumption:** This plan assumes ops has rotated keys and scoped secrets. If not done, repos must wait.
2. **Same-repo PRs only:** Auto review must guard on `github.event.pull_request.head.repo.full_name == github.repository`.
3. **Trusted actors only:** Manual @droid must check `author_association in [OWNER, MEMBER, COLLABORATOR]`.
4. **Debug artifacts disabled:** `upload_debug_artifacts: false` is non-negotiable.
5. **No mutable refs:** All Droid action uses must pin to the safe SHA.
6. **Model convergence:** All repos Phase 2 must use `custom:MiniMax-M2.7-0`.

---

## Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MiniMax key leaked again | LOW | CRITICAL | Phase 0 rotation + scope; audit key usage logs |
| Action ref mutates upstream | LOW | HIGH | Pin to immutable SHA; not version tag |
| Workflow policy blocks BYOK | MEDIUM | MEDIUM | Repo owner approval; check `cargo xtask check-workflows` |
| Droid stops in middle of PR | LOW | MEDIUM | Timeout is 1h; smoke test validates |
| False negatives in review | MEDIUM | LOW | Manual review catches; Phase 1 preserves behavior |

---

## Rollback Plan

If a Phase 1 or Phase 2 PR causes widespread issues:

1. **Immediate:** Revert PR in affected repo
2. **Communication:** Post issue in #droid-migration
3. **Investigation:** Root cause in safe action or repo config?
4. **Fix:** Either fix safe action (unlikely) or adjust repo workflow
5. **Retry:** Re-open PR with fix

No org-wide rollback needed; repos are independent.

---

## Appendix: Ref Implementation (ripr #467)

Repository: `EffortlessMetrics/ripr`  
PR: #467  
Commit: Post-merge verification by hosted checks  
Artifacts: None uploaded (verified by workflow run history)

Model: `custom:MiniMax-M2.7-0`  
BYOK: ~/.factory/settings.local.json with quoted heredoc  
Action: `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`  
Debug: `upload_debug_artifacts: false`  

---

## Sign-Off

- **Plan approved:** [pending]
- **Phase 0 complete:** [pending]
- **Phase 1 start:** [pending]
- **Phase 1 complete:** [pending]
- **Phase 2 complete:** [pending]
- **Final rollout ready:** [pending]
