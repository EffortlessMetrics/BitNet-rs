# EffortlessMetrics Factory Action Migration: Execution Tracking Sheet

**Start date:** [To be filled when Phase 1 begins]  
**Expected completion:** [Start date + 28 days]

Use this document to track progress across all 7 repos.

---

## Phase 0: Pre-Rollout (Ops/Security)

### Prerequisite Checklist

| Item | Owner | Status | Date | Notes |
|------|-------|--------|------|-------|
| Rotate MiniMax Token Plan key | Ops | ⬜ | — | Exposed key rotation |
| Update MINIMAX_API_KEY secret | Ops | ⬜ | — | Scope to 7 repos only |
| Confirm FACTORY_API_KEY valid | Ops | ⬜ | — | Verify no expiration |
| Document key scoping | Ops | ⬜ | — | Record for audit |
| **Phase 0 Sign-Off** | **Ops** | **⬜** | **—** | **Blocker for Phase 1** |

---

## Phase 1: Safety Patches (All 7 Repos)

### Batch 1: High Priority (Mutable Refs)

#### OpenRacing

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | Find @main or @v5 refs |
| | Create branch ci/safe-droid-action | ⬜ | — | — | Local branch created |
| | Replace action refs in droid.yml | ⬜ | — | — | → safe SHA |
| | Replace action refs in droid-review.yml | ⬜ | — | — | → safe SHA |
| | Add upload_debug_artifacts: false | ⬜ | — | — | Both workflows |
| | Pin checkout to SHA v5 | ⬜ | — | — | Both workflows |
| | Run workflow checks | ⬜ | — | — | Repo-specific validator |
| | Open Phase 1 PR | ⬜ | — | [PR#] | Title: ci: use safe Droid action |
| | PR approved + merged | ⬜ | — | [PR#] | Review + merge |
| | Smoke test: create draft PR | ⬜ | — | [PR#] | Dummy change, title [smoke-test] |
| | Smoke test: Droid triggers | ⬜ | — | — | Check workflow logs |
| | Smoke test: no debug artifacts | ⬜ | — | — | Check artifact list in run |
| | Smoke test: close PR | ⬜ | — | — | No merge needed |
| **Phase 1 Complete** | **OpenRacing** | **⬜** | **—** | — | **Ready for Phase 2** |

#### adze

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | Likely @v5 ref |
| | Create branch ci/safe-droid-action | ⬜ | — | — | |
| | Replace action refs + add safety flag | ⬜ | — | — | Same as OpenRacing |
| | Open Phase 1 PR | ⬜ | — | [PR#] | |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test complete | ⬜ | — | — | No artifacts |
| **Phase 1 Complete** | **adze** | **⬜** | **—** | — | **Ready for Phase 2** |

#### SwiftMTP-dev

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | @main ref expected |
| | Create branch ci/safe-droid-action | ⬜ | — | — | |
| | Replace action refs + add safety flag | ⬜ | — | — | Same pattern as OpenRacing |
| | Open Phase 1 PR | ⬜ | — | [PR#] | |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test complete | ⬜ | — | — | No artifacts |
| **Phase 1 Complete** | **SwiftMTP-dev** | **⬜** | **—** | — | **Ready for Phase 2** |

#### SwiftMailSort

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | @main ref expected |
| | Create branch ci/safe-droid-action | ⬜ | — | — | |
| | Replace action refs + add safety flag | ⬜ | — | — | Same pattern |
| | Open Phase 1 PR | ⬜ | — | [PR#] | |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test complete | ⬜ | — | — | No artifacts |
| **Phase 1 Complete** | **SwiftMailSort** | **⬜** | **—** | — | **Ready for Phase 2** |

#### shiplog

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | @main ref expected |
| | Create branch ci/safe-droid-action | ⬜ | — | — | |
| | Replace action refs + add safety flag | ⬜ | — | — | Same pattern |
| | Open Phase 1 PR | ⬜ | — | [PR#] | |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test complete | ⬜ | — | — | No artifacts |
| **Phase 1 Complete** | **shiplog** | **⬜** | **—** | — | **Ready for Phase 2** |

### Batch 2: Medium Priority (SHA-Pinned Refs)

#### perl-lsp

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | SHA-pinned Factory-AI |
| | Create branch ci/safe-droid-action | ⬜ | — | — | |
| | Replace action refs + add safety flag | ⬜ | — | — | Same pattern |
| | Open Phase 1 PR | ⬜ | — | [PR#] | |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test complete | ⬜ | — | — | No artifacts |
| **Phase 1 Complete** | **perl-lsp** | **⬜** | **—** | — | **Ready for Phase 2** |

#### pkm-python

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 1 | Audit current workflows | ⬜ | — | — | SHA-pinned Factory-AI |
| | Create branch ci/safe-droid-action | ⬜ | — | — | |
| | Replace action refs + add safety flag | ⬜ | — | — | Same pattern |
| | Open Phase 1 PR | ⬜ | — | [PR#] | |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test complete | ⬜ | — | — | No artifacts |
| **Phase 1 Complete** | **pkm-python** | **⬜** | **—** | — | **Ready for Phase 2** |

### Phase 1 Summary

**Target:** All 7 repos safe action SHA + disable artifacts

| Repo | Phase 1 Status | PR# | Date | Lead |
|------|----------------|----|------|------|
| OpenRacing | ⬜ Pending | — | — | — |
| adze | ⬜ Pending | — | — | — |
| SwiftMTP-dev | ⬜ Pending | — | — | — |
| SwiftMailSort | ⬜ Pending | — | — | — |
| shiplog | ⬜ Pending | — | — | — |
| perl-lsp | ⬜ Pending | — | — | — |
| pkm-python | ⬜ Pending | — | — | — |

**Phase 1 Completion Criteria:**
- [ ] All 7 repos have Phase 1 PR merged
- [ ] All 7 repos have smoke test green
- [ ] 0 repos have raw `droid-review-debug-<run_id>` artifacts

---

## Phase 2: Baseline Convergence (All 7 Repos)

### Batch 1: High Priority

#### OpenRacing

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | Create branch ci/droid-baseline | ⬜ | — | — | |
| | Add MiniMax BYOK step (droid-review.yml) | ⬜ | — | — | Heredoc with quotes |
| | Add same-repo guard (droid-review.yml) | ⬜ | — | — | github.event.pull_request... |
| | Add MINIMAX_API_KEY env var | ⬜ | — | — | Both workflows |
| | Update action inputs (model, review_depth) | ⬜ | — | — | custom:MiniMax-M2.7-0 |
| | Update droid.yml trusted-actor guard | ⬜ | — | — | OWNER/MEMBER/COLLABORATOR only |
| | Change droid.yml permissions to read | ⬜ | — | — | contents: read |
| | Create AGENTS.md | ⬜ | — | — | High-level config |
| | Create .factory/rules/droid-review.md | ⬜ | — | — | Droid-specific rules |
| | Run workflow checks | ⬜ | — | — | |
| | Open Phase 2 PR | ⬜ | — | [PR#] | Title: ci: align Droid review baseline |
| | PR approved + merged | ⬜ | — | [PR#] | |
| | Smoke test: Droid with MiniMax | ⬜ | — | — | Check logs for custom:MiniMax-M2.7-0 |
| | Smoke test: Manual @droid review | ⬜ | — | — | Comment as OWNER/MEMBER |
| | Smoke test: no artifacts | ⬜ | — | — | No debug artifacts |
| | Smoke test: close PR | ⬜ | — | — | |
| **Phase 2 Complete** | **OpenRacing** | **⬜** | **—** | — | **All done** |

#### adze

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | (Same tasks as OpenRacing) | ⬜ | — | — | Copy checklist above |
| | ... | ⬜ | — | — | |
| **Phase 2 Complete** | **adze** | **⬜** | **—** | — | **All done** |

#### SwiftMTP-dev

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | (Same tasks as OpenRacing) | ⬜ | — | — | Copy checklist above |
| **Phase 2 Complete** | **SwiftMTP-dev** | **⬜** | **—** | — | **All done** |

#### SwiftMailSort

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | (Same tasks as OpenRacing) | ⬜ | — | — | Copy checklist above |
| **Phase 2 Complete** | **SwiftMailSort** | **⬜** | **—** | — | **All done** |

#### shiplog

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | (Same tasks as OpenRacing) | ⬜ | — | — | Copy checklist above |
| **Phase 2 Complete** | **shiplog** | **⬜** | **—** | — | **All done** |

### Batch 2: Medium Priority

#### perl-lsp

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | (Same tasks as OpenRacing) | ⬜ | — | — | Copy checklist above |
| **Phase 2 Complete** | **perl-lsp** | **⬜** | **—** | — | **All done** |

#### pkm-python

| Stage | Task | Status | Date | Link | Notes |
|-------|------|--------|------|------|-------|
| Phase 2 | (Same tasks as OpenRacing) | ⬜ | — | — | Copy checklist above |
| **Phase 2 Complete** | **pkm-python** | **⬜** | **—** | — | **All done** |

### Phase 2 Summary

**Target:** All 7 repos have MiniMax BYOK, guards, guidance

| Repo | Phase 2 Status | PR# | Date | Lead |
|------|----------------|----|------|------|
| OpenRacing | ⬜ Pending | — | — | — |
| adze | ⬜ Pending | — | — | — |
| SwiftMTP-dev | ⬜ Pending | — | — | — |
| SwiftMailSort | ⬜ Pending | — | — | — |
| shiplog | ⬜ Pending | — | — | — |
| perl-lsp | ⬜ Pending | — | — | — |
| pkm-python | ⬜ Pending | — | — | — |

**Phase 2 Completion Criteria:**
- [ ] All 7 repos have Phase 2 PR merged
- [ ] All 7 repos have smoke test with MiniMax
- [ ] Manual @droid proven in 2+ repos
- [ ] All 7 repos have AGENTS.md and .factory/rules/droid-review.md

---

## Final Validation & Sign-Off

### Acceptance Criteria

| Criterion | Status | Verified By | Date |
|-----------|--------|-------------|------|
| All 7 repos use safe action SHA | ⬜ | Eng Lead | — |
| All 7 repos have upload_debug_artifacts: false | ⬜ | QA | — |
| All 7 repos have MiniMax BYOK step | ⬜ | QA | — |
| All 7 repos use custom:MiniMax-M2.7-0 | ⬜ | QA | — |
| All 7 repos have same-repo guard | ⬜ | QA | — |
| All 7 repos have trusted-actor guard | ⬜ | QA | — |
| All 7 repos have AGENTS.md | ⬜ | QA | — |
| All 7 repos have .factory/rules/droid-review.md | ⬜ | QA | — |
| 0 repos upload raw droid-review-debug artifacts | ⬜ | QA | — |
| Manual @droid proven in 2+ repos | ⬜ | Eng | — |
| MiniMax visible in provider dashboard | ⬜ | Eng | — |
| No Factory-AI/droid-action refs remain | ⬜ | QA | — |

### Sign-Off

**Phase 0 (Ops):**
- [ ] Completed by: _________________ Date: _______
- [ ] Verified by: _________________ Date: _______

**Phase 1 (Eng Lead):**
- [ ] Completed by: _________________ Date: _______
- [ ] Verified by: _________________ Date: _______

**Phase 2 (Eng Lead):**
- [ ] Completed by: _________________ Date: _______
- [ ] Verified by: _________________ Date: _______

**Validation (QA):**
- [ ] Completed by: _________________ Date: _______

**Final Approval (Lead):**
- [ ] Ready for Phase 3: YES / NO
- [ ] Approved by: _________________ Date: _______
- [ ] Notes: _________________________________________

---

## Notes & Issues

### Week 1

- [ ] Phase 0 complete by: ___________
- [ ] OpenRacing Phase 1 complete by: ___________
- [ ] Notes: ___________________________________________

### Week 2

- [ ] adze Phase 1 complete by: ___________
- [ ] SwiftMTP-dev Phase 1 complete by: ___________
- [ ] Notes: ___________________________________________

### Week 3

- [ ] SwiftMailSort Phase 1 complete by: ___________
- [ ] shiplog Phase 1 complete by: ___________
- [ ] OpenRacing Phase 2 complete by: ___________
- [ ] Notes: ___________________________________________

### Week 4

- [ ] Batch 1 Phase 2 complete by: ___________
- [ ] Batch 2 Phase 1 complete by: ___________
- [ ] Notes: ___________________________________________

### Week 5+

- [ ] Batch 2 Phase 2 complete by: ___________
- [ ] All validation complete by: ___________
- [ ] Ready for Phase 3 by: ___________
- [ ] Notes: ___________________________________________

---

## Escalations

**Issue:** ___________________________________________

**Repo:** ___________________________________________

**PR Link:** ___________________________________________

**Error Message:** ___________________________________________

**Action Taken:** ___________________________________________

**Resolution:** ___________________________________________

**Date Resolved:** ___________________________________________

---

## Key Contacts

| Role | Name | Contact | Notes |
|------|------|---------|-------|
| Project Lead | — | — | — |
| Ops/Security Owner | — | — | Phase 0 |
| Engineering Lead | — | — | Phases 1–2 |
| QA/Code Review | — | — | Validation |
| Repository Owners | — | — | OpenRacing, adze, etc. |

---

**Print this document and update it regularly. Share progress weekly in #droid-migration.**
