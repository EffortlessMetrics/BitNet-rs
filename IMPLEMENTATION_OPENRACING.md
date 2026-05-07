# OpenRacing: Droid Migration Implementation Package

**Batch:** 1 (Mutable ref)  
**Current Ref:** `Factory-AI/droid-action@main` (assumed, requires verification)  
**Priority:** HIGH  
**Sequence:** 1st in rollout

---

## Current State Assessment

### Workflows to Patch

- `.github/workflows/droid.yml` — Manual @droid trigger
- `.github/workflows/droid-review.yml` — Auto PR review

### Expected Issues

- Direct `Factory-AI/droid-action@main` (or @v5) reference
- No `upload_debug_artifacts: false` (or defaults to true)
- No MiniMax BYOK settings.local.json
- Missing same-repo/trusted-actor guards
- No repo-local review guidance

---

## Phase 1: Safety Patch Implementation

### Changes Required

**In `.github/workflows/droid.yml`:**

1. Find this line:
   ```yaml
   uses: Factory-AI/droid-action@main
   ```
   (or `@v5` or other mutable ref)

2. Replace with:
   ```yaml
   uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5; raw debug artifact upload disabled
   ```

3. In the `with:` section, add:
   ```yaml
   upload_debug_artifacts: false
   ```

4. For checkout action, ensure pinned to SHA:
   ```yaml
   uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5
   ```

**In `.github/workflows/droid-review.yml`:**

Repeat steps 1–4 above.

### Detailed Diff Template

```diff
  - name: Run Droid [Auto Review|Tag|Security]
-   uses: Factory-AI/droid-action@main
+   uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f
    with:
      factory_api_key: ${{ secrets.FACTORY_API_KEY }}
+     upload_debug_artifacts: false
      # ... rest of with block
```

### Validation: Phase 1 PR

**Title:** `ci: use safe Droid action`

**Body:**
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

### Smoke Test: Phase 1

1. **Merge Phase 1 PR** (or mark ready for merge)
2. **Create a test PR** in OpenRacing:
   - Draft PR with dummy change (e.g., update README)
   - Title: `[smoke-test] Droid phase 1`
   - Body: "Smoke test for Phase 1 safe action deployment"
3. **Observe Droid Auto Review:**
   - Does it trigger?
   - Does it complete successfully?
   - No errors in workflow run?
4. **Check artifacts:**
   - Go to workflow run details
   - Expand "Artifacts" section
   - **Expected:** No artifact named `droid-review-debug-<run_id>`
   - **Allowed:** No artifacts at all, or `summary-<run_id>` only
5. **Validation result:** ✅ or ❌ Record in PR checklist
6. **Close test PR** (no merge needed)

---

## Phase 2: Baseline Convergence Implementation

### Changes Required

**Add MiniMax BYOK step** to `.github/workflows/droid-review.yml` (before action):

```yaml
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
```

**Update Droid action inputs:**

```yaml
      - name: Run Droid Auto Review with MiniMax M2.7 BYOK
        uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f
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

**Add environment variable to job:**

```yaml
jobs:
  droid-review:
    env:
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
```

**Add same-repo guard** (in `if:` block):

```yaml
jobs:
  droid-review:
    if: |
      github.event.pull_request.head.repo.full_name == github.repository &&
      !contains(github.event.pull_request.title, '[skip-review]')
```

**Update `.github/workflows/droid.yml`** (manual @droid) similarly, but with `contents: read`:

```yaml
permissions:
  contents: read
  pull-requests: write
  issues: write
  id-token: write
  actions: read
```

And add trusted-actor guard:

```yaml
jobs:
  droid:
    if: |
      (
        github.event_name == 'issue_comment' &&
        contains(github.event.comment.body, '@droid') &&
        contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'), github.event.comment.author_association)
      ) ||
      (
        github.event_name == 'pull_request_review_comment' &&
        contains(github.event.comment.body, '@droid') &&
        contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'), github.event.comment.author_association)
      ) ||
      [... additional event type checks as in template ...]
```

### Add Minimal Repo-Local Guidance

**File: `AGENTS.md`**

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

- **Auto-review:** Same-repo PRs, auto-triggered on open/sync/ready-for-review
- **Manual:** Comment `@droid review` or `@droid security` (OWNER/MEMBER/COLLABORATOR only)

## Model

MiniMax M2.7 via BYOK (custom:MiniMax-M2.7-0)

## For Reviewers

- Expect shallow review (priority on correctness, security, maintainability)
- Droid reviews are repair-packet format; see `.factory/rules/droid-review.md`
- Manual @droid follow-up for deep dives if needed
```

**File: `.factory/rules/droid-review.md`**

```markdown
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
```

### Validation: Phase 2 PR

**Title:** `ci: align Droid review baseline`

**Body:**
```markdown
## Summary

- Add MiniMax BYOK through `~/.factory/settings.local.json`
- Set review model to `custom:MiniMax-M2.7-0`
- Add same-repo guard for auto review
- Add trusted-actor guard for manual @droid
- Add minimal repo-local guidance (AGENTS.md, .factory/rules/)

## Why

Convergence to org baseline reduces review variance and ensures safe, consistent BYOK model usage across the rollout batch.

## Changes

- `.github/workflows/droid-review.yml` — BYOK step, model inputs, same-repo guard
- `.github/workflows/droid.yml` — Trusted-actor guard, model inputs
- `AGENTS.md` — High-level review config
- `.factory/rules/droid-review.md` — Droid-specific rules

## Validation

- [x] Repo workflow/static checks pass.
- [ ] Same-repo smoke PR succeeds with MiniMax model.
- [ ] Manual `@droid review` works (OWNER/MEMBER comment).
- [ ] No raw artifacts uploaded.
```

### Smoke Test: Phase 2

1. **Merge Phase 2 PR**
2. **Create test PR:**
   - Draft PR with dummy change
   - Title: `[smoke-test] Droid phase 2 baseline`
   - Body: "Smoke test for Phase 2 MiniMax BYOK and guards"
3. **Observe auto review:**
   - Does Droid trigger?
   - Does it use MiniMax (check action run logs for "custom:MiniMax-M2.7-0")?
   - Does it complete successfully?
4. **Manual @droid trigger:**
   - Comment on PR: `@droid review`
   - As OWNER/MEMBER/COLLABORATOR account
   - Does Droid respond?
   - Does it use MiniMax?
5. **Check artifacts:**
   - No `droid-review-debug-<run_id>` uploaded
6. **Validation result:** ✅ or ❌ Record
7. **Close test PR**

---

## Checklist: Full OpenRacing Rollout

### Phase 1

- [ ] Current workflows examined
- [ ] Phase 1 PR opened
- [ ] Phase 1 PR merged
- [ ] Smoke test run completed
- [ ] No unsafe artifacts found
- [ ] Phase 1 ✅ Complete

### Phase 2

- [ ] Phase 2 PR opened
- [ ] Phase 2 PR merged
- [ ] Smoke test with MiniMax run completed
- [ ] Manual @droid review works
- [ ] Phase 2 ✅ Complete

### Sign-Off

- [ ] OpenRacing repo ready for broad rollout
- [ ] All artifacts clean
- [ ] MiniMax integration confirmed
- [ ] Documentation present

---

## Key Files to Examine

Before starting Phase 1:

1. **`.github/workflows/droid-review.yml`** — Current auto-review workflow
2. **`.github/workflows/droid.yml`** — Current manual @droid workflow
3. **`AGENTS.md`** — Existing review config (if present)
4. **`.factory/` directory** — Existing Factory rules/skills

## Repo Owner Notes

- Phase 1 is a **safety-first** operation; no behavior change
- Phase 2 introduces **MiniMax BYOK**; requires `MINIMAX_API_KEY` secret
- Both phases are **non-breaking**; existing PRs continue to review normally
- Rollback is simple: revert PR, action reverts to Factory-AI

---

## References

- Safe action: `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Reference impl: `EffortlessMetrics/ripr` PR #467
- Org migration plan: `MIGRATION_PLAN.md` (this branch)
