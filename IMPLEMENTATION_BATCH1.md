# Batch 1 Implementation: SwiftMTP-dev, SwiftMailSort, shiplog

**Repos:** SwiftMTP-dev, SwiftMailSort, shiplog  
**Priority:** HIGH  
**Status:** Phase 1 and Phase 2 templates ready

---

## SwiftMTP-dev: Implementation Summary

**Current Ref:** `Factory-AI/droid-action@main` (assumed)  
**Sequence:** 3rd in rollout

### Phase 1: Safety Patch

**Files:**
- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

**Changes:**
1. Replace `Factory-AI/droid-action@main` → `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
2. Add `upload_debug_artifacts: false`
3. Pin checkout to SHA v5

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

---

## SwiftMailSort: Implementation Summary

**Current Ref:** `Factory-AI/droid-action@main` (assumed)  
**Sequence:** 4th in rollout

### Phase 1: Safety Patch

**Files:** Same as SwiftMTP-dev
- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

**Changes:** Identical to SwiftMTP-dev Phase 1

**PR:** `ci: use safe Droid action`

**Smoke test:** Same format

### Phase 2: Baseline Convergence

**Changes:** Identical to SwiftMTP-dev Phase 2

**PR:** `ci: align Droid review baseline`

**Smoke test:** Same format

---

## shiplog: Implementation Summary

**Current Ref:** `Factory-AI/droid-action@main` (assumed)  
**Sequence:** 5th in rollout

### Phase 1: Safety Patch

**Files:** Same as previous repos
- `.github/workflows/droid.yml`
- `.github/workflows/droid-review.yml`

**Changes:** Identical to SwiftMTP-dev Phase 1

**PR:** `ci: use safe Droid action`

**Smoke test:** Same format

### Phase 2: Baseline Convergence

**Changes:** Identical to SwiftMTP-dev Phase 2

**PR:** `ci: align Droid review baseline`

**Smoke test:** Same format

---

## Batch 1 Unified Template

All three repos (SwiftMTP-dev, SwiftMailSort, shiplog) follow the same pattern:

### Phase 1 Diff

```diff
  - name: Run Droid [Auto Review|Tag]
-   uses: Factory-AI/droid-action@main
+   uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f
    with:
      factory_api_key: ${{ secrets.FACTORY_API_KEY }}
+     upload_debug_artifacts: false
      # ... rest
```

### Phase 2 Additions

**To droid-review.yml:**

```yaml
  droid-review:
    env:
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

    if: |
      github.event.pull_request.head.repo.full_name == github.repository &&
      !contains(github.event.pull_request.title, '[skip-review]')

    steps:
      - name: Checkout repository
        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5

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

**To droid.yml:**

```yaml
  droid:
    if: |
      (
        github.event_name == 'issue_comment' &&
        contains(github.event.comment.body, '@droid') &&
        contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'), github.event.comment.author_association)
      ) ||
      (... other event types ...)

    permissions:
      contents: read
      pull-requests: write
      issues: write
      id-token: write
      actions: read

    env:
      MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5

      - name: Configure MiniMax BYOK for Factory Droid
        shell: bash
        run: |
          mkdir -p "$HOME/.factory"
          cat > "$HOME/.factory/settings.local.json" <<'JSON'
          { ... same JSON as above ... }
          JSON

      - name: Run Droid Exec with MiniMax M2.7 BYOK
        uses: EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f
        with:
          factory_api_key: ${{ secrets.FACTORY_API_KEY }}
          upload_debug_artifacts: false

          review_depth: shallow
          review_model: "custom:MiniMax-M2.7-0"
          security_model: "custom:MiniMax-M2.7-0"
          show_full_output: false
```

**Add repo-local guidance:**
- `AGENTS.md`
- `.factory/rules/droid-review.md`

(Use OpenRacing templates)

---

## Execution Order: Batch 1

1. **OpenRacing** — Phase 1, merge, smoke, Phase 2, merge, smoke
2. **adze** — Phase 1, merge, smoke, Phase 2, merge, smoke
3. **SwiftMTP-dev** — Phase 1, merge, smoke, Phase 2, merge, smoke
4. **SwiftMailSort** — Phase 1, merge, smoke, Phase 2, merge, smoke
5. **shiplog** — Phase 1, merge, smoke, Phase 2, merge, smoke

**Timeline:** 5 repos × 2 PRs each = 10 PRs, ~1 per day = ~10 days

---

## Validation Checklist: Batch 1

### Phase 1 Validation (All 5 repos)

- [ ] OpenRacing Phase 1 ✅
- [ ] adze Phase 1 ✅
- [ ] SwiftMTP-dev Phase 1 ✅
- [ ] SwiftMailSort Phase 1 ✅
- [ ] shiplog Phase 1 ✅
- [ ] All smoke tests green
- [ ] 0 repos have `droid-review-debug-<run_id>` artifacts

### Phase 2 Validation (All 5 repos)

- [ ] OpenRacing Phase 2 ✅
- [ ] adze Phase 2 ✅
- [ ] SwiftMTP-dev Phase 2 ✅
- [ ] SwiftMailSort Phase 2 ✅
- [ ] shiplog Phase 2 ✅
- [ ] All smoke tests with MiniMax
- [ ] Manual @droid works in at least 2 repos
- [ ] MiniMax usage visible in provider dashboard
- [ ] All repos have AGENTS.md and .factory/rules/

---

## Common Issues & Fixes

### Issue: Workflow validation fails

**Cause:** YAML syntax error in BYOK heredoc or action inputs

**Fix:**
- Ensure quoted heredoc: `cat > file <<'JSON'` (single quotes)
- Validate YAML: `yamllint .github/workflows/`
- Check indentation (2 spaces)

### Issue: Droid action doesn't trigger

**Cause:** Same-repo guard or permissions issue

**Fix:**
- Verify `github.event.pull_request.head.repo.full_name == github.repository`
- Ensure PR is in same repo (not fork)
- Check permissions: `contents: write` for auto-review

### Issue: No MiniMax usage in logs

**Cause:** Settings file not written or API key empty

**Fix:**
- Verify heredoc runs successfully
- Check `MINIMAX_API_KEY` secret exists in repo
- Run workflow manually to inspect logs

### Issue: Safe action doesn't exist

**Cause:** Typo in SHA or org name

**Fix:**
- Verify SHA: `01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Verify org: `EffortlessMetrics` (not Effort, not Metrics)
- Check GitHub repo: `github.com/EffortlessMetrics/droid-action-safe`

---

## References

- Safe action: `EffortlessMetrics/droid-action-safe@01e76b659e4b1e5f23feedc8cfabf8dc14c7485f`
- Reference impl: `EffortlessMetrics/ripr` PR #467
- OpenRacing template: `IMPLEMENTATION_OPENRACING.md`
- adze template: `IMPLEMENTATION_ADZE.md`
- Migration plan: `MIGRATION_PLAN.md`
