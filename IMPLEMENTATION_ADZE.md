# adze: Droid Migration Implementation Package

**Batch:** 1 (Mutable ref)  
**Current Ref:** `Factory-AI/droid-action@v5` (assumed, requires verification)  
**Priority:** HIGH  
**Sequence:** 2nd in rollout

---

## Current State Assessment

### Workflows to Patch

- `.github/workflows/droid.yml` — Manual @droid trigger
- `.github/workflows/droid-review.yml` — Auto PR review

### Expected Issues

- Direct `Factory-AI/droid-action@v5` (mutable tag)
- No `upload_debug_artifacts: false`
- No MiniMax BYOK settings.local.json
- Missing same-repo/trusted-actor guards
- No repo-local review guidance

### Differences from OpenRacing

**adze may use `@v5` tag instead of `@main`** — both are mutable and must be replaced.

---

## Phase 1: Safety Patch Implementation

### Changes Required

**Replace action ref in both workflow files:**

```yaml
# OLD:
uses: Factory-AI/droid-action@v5

# NEW:
uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f # based on Factory-AI/droid-action v5; raw debug artifact upload disabled
```

**Add `upload_debug_artifacts: false`** in all Droid action `with:` blocks.

**Pin checkout action:**

```yaml
uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5
```

### Files to Modify

- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

### PR Template: Phase 1

**Title:** `ci: use safe Droid action`

**Body:** (Same as OpenRacing Phase 1)

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

1. Merge Phase 1 PR
2. Create draft PR: `[smoke-test] Droid phase 1 safety patch`
3. Verify:
   - Droid Auto Review triggers
   - No errors in run logs
   - No `droid-review-debug-<run_id>` artifact
4. Close test PR

---

## Phase 2: Baseline Convergence Implementation

### Add MiniMax BYOK Step

In `.github/workflows/droid-review.yml`, add before the Droid action:

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

### Update Droid Action Inputs

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

### Add Environment Variable

```yaml
jobs:
  droid-review:
    env:
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
```

### Add Same-Repo Guard

```yaml
jobs:
  droid-review:
    if: |
      github.event.pull_request.head.repo.full_name == github.repository &&
      !contains(github.event.pull_request.title, '[skip-review]')
```

### Update Manual Workflow (`.github/workflows/droid.yml`)

- Add same MiniMax BYOK step
- Add trusted-actor guard (see OpenRacing implementation)
- Set `contents: read` instead of `contents: write`
- Add model inputs

### Add Repo-Local Guidance

**`AGENTS.md`** (same as OpenRacing template)

**`.factory/rules/droid-review.md`** (same as OpenRacing template)

### PR Template: Phase 2

**Title:** `ci: align Droid review baseline`

(Same structure as OpenRacing Phase 2)

### Smoke Test: Phase 2

1. Merge Phase 2 PR
2. Create draft PR: `[smoke-test] Droid phase 2 baseline`
3. Verify:
   - Auto review triggers with MiniMax
   - Manual `@droid review` works (as OWNER/MEMBER)
   - No unsafe artifacts
4. Close test PR

---

## Key Differences from OpenRacing

| Aspect | OpenRacing | adze |
|--------|-----------|------|
| Current ref | `@main` | `@v5` |
| Drift risk | Very high | High |
| Implementation | Same phases | Same phases |

---

## Checklist: Full adze Rollout

### Phase 1

- [ ] Current workflows examined
- [ ] Phase 1 PR opened
- [ ] Phase 1 PR merged
- [ ] Smoke test completed
- [ ] No unsafe artifacts

### Phase 2

- [ ] Phase 2 PR opened
- [ ] Phase 2 PR merged
- [ ] Smoke test with MiniMax
- [ ] Manual @droid works
- [ ] Phase 2 ✅ Complete

---

## References

- Safe action: `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Reference impl: `EffortlessMetrics/ripr` PR #467
- OpenRacing impl: `IMPLEMENTATION_OPENRACING.md` (use as template)
