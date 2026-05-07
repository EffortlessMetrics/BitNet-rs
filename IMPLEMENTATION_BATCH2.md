# Batch 2 Implementation: perl-lsp, pkm-python

**Repos:** perl-lsp, pkm-python  
**Priority:** MEDIUM  
**Ref Type:** SHA-pinned (not mutable, but still unsafe BYOK)  
**Status:** Phase 1 and Phase 2 templates ready

---

## Batch 2 Characteristics

### Difference from Batch 1

| Aspect | Batch 1 | Batch 2 |
|--------|---------|---------|
| Ref type | Mutable (`@main`, `@v5`) | Pinned SHA (safer, but still Factory-AI) |
| Drift risk | Very high | Low (SHA-pinned) |
| Urgency | CRITICAL | Medium |
| Implementation | Same as Batch 1 | Same as Batch 1 |

**Key:** Even though Batch 2 uses SHA pins, they still reference upstream Factory-AI directly, which means:
- Upstream behavior changes affect them
- Raw debug artifact upload can still leak BYOK secrets
- No MiniMax BYOK convergence

Both must be migrated.

---

## perl-lsp: Implementation Summary

**Current Ref:** SHA-pinned Factory-AI/droid-action (requires verification)  
**Sequence:** 6th in rollout

### Phase 1: Safety Patch

**Files:**
- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

**Changes:**
1. Replace pinned Factory-AI ref → `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
2. Add `upload_debug_artifacts: false`
3. Pin checkout to SHA v5

**Current state check:**
```bash
# Before starting, verify current ref:
rg "Factory-AI/droid-action" .github/workflows/
# Expected: Factory-AI/droid-action@<40-char-sha>
```

**PR:** `ci: use safe Droid action`

**Smoke test:**
- Create draft PR
- Verify Droid triggers, no debug artifacts
- Close PR

### Phase 2: Baseline Convergence

**Changes:**
1. Add MiniMax BYOK heredoc step
2. Add `MINIMAX_API_KEY` env var
3. Update action inputs with model: `custom:MiniMax-M2.7-0`
4. Add same-repo guard in droid-review.yml
5. Update droid.yml with trusted-actor guard and contents: read
6. Add AGENTS.md and .factory/rules/droid-review.md

**PR:** `ci: align Droid review baseline`

**Smoke test:**
- Create draft PR
- Verify MiniMax model used
- Test manual @droid review
- Close PR

**Special note:** If perl-lsp is Perl-focused, ensure:
- Review guidance mentions Perl patterns if applicable
- AGENTS.md includes language-specific context

---

## pkm-python: Implementation Summary

**Current Ref:** SHA-pinned Factory-AI/droid-action (requires verification)  
**Sequence:** 7th in rollout

### Phase 1: Safety Patch

**Files:** Same as perl-lsp
- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

**Changes:** Identical to perl-lsp Phase 1

**PR:** `ci: use safe Droid action`

**Smoke test:** Same format

### Phase 2: Baseline Convergence

**Changes:** Identical to perl-lsp Phase 2

**PR:** `ci: align Droid review baseline`

**Smoke test:** Same format

**Special note:** If pkm-python is Python-focused, ensure:
- Review guidance mentions Python patterns (type hints, etc.)
- AGENTS.md includes language-specific context

---

## Batch 2 Unified Template

### Pre-Implementation Checklist

For each repo:

- [ ] Examine `.github/workflows/droid.yml` for current Factory-AI ref
- [ ] Examine `.github/workflows/droid-review.yml` for current Factory-AI ref
- [ ] Document exact current SHA (e.g., `Factory-AI/droid-action@a1b2c3d...`)
- [ ] Verify both files exist
- [ ] Check for any custom Droid config (might differ from standard)

### Phase 1 Implementation

**Scope:** Replace refs, add safety flag, pin checkout

**Commands to run (per repo):**

```bash
# In repo root
cd /path/to/perl-lsp  # or pkm-python

# Find current refs
rg "Factory-AI/droid-action" .github/workflows/

# Create branch
git checkout -b ci/safe-droid-action

# Edit .github/workflows/droid.yml
# Edit .github/workflows/droid-review.yml

# Validate YAML
yamllint .github/workflows/droid*.yml || true

# Commit and push
git add .github/workflows/
git commit -m "ci: use safe Droid action"
git push origin ci/safe-droid-action
```

**PR steps:**
1. Open PR on GitHub
2. Use Phase 1 PR template (see below)
3. Wait for review
4. Merge
5. Smoke test (create draft PR, verify Droid runs, close)

### Phase 2 Implementation

**Scope:** Add MiniMax BYOK, guards, model inputs, guidance

**Additions to both workflow files:**

```yaml
# At job level:
  droid-review:
    env:
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

# Add if guard (droid-review.yml):
    if: |
      github.event.pull_request.head.repo.full_name == github.repository &&
      !contains(github.event.pull_request.title, '[skip-review]')

# Add step before action:
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

# Update action inputs:
        with:
          factory_api_key: ${{ secrets.FACTORY_API_KEY }}
          upload_debug_artifacts: false

          automatic_review: true
          automatic_security_review: true

          review_depth: shallow
          review_model: "custom:MiniMax-M2.7-0"
          security_model: "custom:MiniMax-M2.7-0"

          security_severity_threshold: high
          security_block_on_critical: true
          security_block_on_high: false

          include_suggestions: true
          show_full_output: false
```

**Droid.yml updates:**

```yaml
  droid:
    # Add trusted-actor guard (see OpenRacing for full template)
    if: |
      (github.event_name == 'issue_comment' && ... ) ||
      ...

    permissions:
      contents: read
      pull-requests: write
      issues: write
      id-token: write
      actions: read

    env:
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

    steps:
      # Add same BYOK step
      # Update action inputs with model: custom:MiniMax-M2.7-0
```

**Add guidance files:**
- `AGENTS.md` (reference: OpenRacing)
- `.factory/rules/droid-review.md` (reference: OpenRacing)

**Commands:**

```bash
git checkout -b ci/droid-baseline

# Edit .github/workflows/droid.yml
# Edit .github/workflows/droid-review.yml
# Create AGENTS.md
# Create .factory/rules/droid-review.md

git add .github/workflows/ AGENTS.md .factory/
git commit -m "ci: align Droid review baseline"
git push origin ci/droid-baseline
```

### Phase 1 PR Template

**Title:** `ci: use safe Droid action`

**Body:**

```markdown
## Summary

- Switch Droid workflows to `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`.
- Add `upload_debug_artifacts: false`.
- Preserve existing Droid behavior except for disabling raw debug artifact upload.

## Why

The upstream Factory action can upload raw `$HOME/.factory/**` and `droid-prompts/**`. In BYOK mode that can include resolved provider credentials. Normal Droid runs should not upload raw debug artifacts.

## Changed workflows

- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

## Validation

- [x] Repo workflow checks pass.
- [ ] Same-repo PR smoke run succeeds.
- [ ] No raw artifact named `droid-review-debug-<run_id>` is uploaded.

## Non-goals

- No permission reduction.
- No model/provider change except MiniMax BYOK convergence if already intended.
- No `review_depth: deep`.
- No `pull_request_target`.
```

### Phase 2 PR Template

**Title:** `ci: align Droid review baseline`

**Body:**

```markdown
## Summary

- Add MiniMax BYOK through `~/.factory/settings.local.json`
- Set review model to `custom:MiniMax-M2.7-0`
- Add same-repo guard for auto review
- Add trusted-actor guard for manual @droid
- Add minimal repo-local guidance

## Why

Convergence to org baseline reduces review variance and ensures safe, consistent BYOK model usage.

## Changes

- `.github/workflows/droid-review.yml` — BYOK step, model inputs, same-repo guard
- `.github/workflows/droid.yml` — Trusted-actor guard, model inputs
- `AGENTS.md` — High-level review config
- `.factory/rules/droid-review.md` — Droid-specific rules

## Validation

- [x] Repo workflow checks pass.
- [ ] Same-repo smoke PR succeeds with MiniMax model.
- [ ] Manual `@droid review` works (OWNER/MEMBER comment).
- [ ] No raw artifacts uploaded.
```

---

## Execution Order: Batch 2

1. **perl-lsp** — Phase 1, merge, smoke, Phase 2, merge, smoke
2. **pkm-python** — Phase 1, merge, smoke, Phase 2, merge, smoke

**Timeline:** 2 repos × 2 PRs each = 4 PRs, ~4 days

**Prerequisite:** All Batch 1 (5 repos) must be complete Phase 1 before starting Batch 2

---

## Validation Checklist: Batch 2

### Phase 1 Validation

- [ ] perl-lsp Phase 1 ✅ (safe action, no debug artifacts)
- [ ] pkm-python Phase 1 ✅ (safe action, no debug artifacts)
- [ ] Both smoke tests green
- [ ] 0 repos have `droid-review-debug-<run_id>` artifacts

### Phase 2 Validation

- [ ] perl-lsp Phase 2 ✅ (MiniMax BYOK, guards, guidance)
- [ ] pkm-python Phase 2 ✅ (MiniMax BYOK, guards, guidance)
- [ ] Both smoke tests with MiniMax
- [ ] Manual @droid works in at least 1 repo
- [ ] Both repos have AGENTS.md and .factory/rules/

---

## Full Batch 1 + 2 Completion Criteria

After all 7 repos complete Phases 1 + 2:

- ✅ 7 repos use safe action SHA
- ✅ 7 repos have `upload_debug_artifacts: false`
- ✅ 7 repos have MiniMax BYOK configured
- ✅ 7 repos use model: `custom:MiniMax-M2.7-0`
- ✅ 7 repos have same-repo/trusted-actor guards
- ✅ 7 repos have AGENTS.md and .factory/rules/
- ✅ 0 repos upload raw droid-review-debug artifacts
- ✅ MiniMax usage visible in provider dashboard
- ✅ Manual @droid review proven in 2+ repos
- ✅ No Factory-AI/droid-action direct refs remain

---

## References

- Safe action: `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Reference impl: `EffortlessMetrics/ripr` PR #467
- Batch 1 templates: `IMPLEMENTATION_BATCH1.md`
- OpenRacing deep dive: `IMPLEMENTATION_OPENRACING.md`
- Migration plan: `MIGRATION_PLAN.md`
