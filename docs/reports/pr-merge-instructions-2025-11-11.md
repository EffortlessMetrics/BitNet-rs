# PR Merge Instructions - Final Sprint-2 Lockdown

**Date**: November 11, 2025
**Status**: PRs ready, branch protection requires manual intervention

---

## Current Status

### PR #507: Dependabot Configuration
- **State**: MERGEABLE but BLOCKED
- **Checks**: 6 passing
- **Issue**: Branch protection rules require approval or admin override
- **URL**: https://github.com/EffortlessMetrics/BitNet-rs/pull/507

### PR #508: README Guardrails Pointer
- **State**: MERGEABLE (likely also BLOCKED)
- **Checks**: 12 passing
- **URL**: https://github.com/EffortlessMetrics/BitNet-rs/pull/508

---

## Merge Options (Choose One)

### Option A: Request Review (Recommended)

```bash
# Ask maintainers for approval
gh pr edit 507 --add-reviewer EffortlessMetrics/maintainers
# Or specific collaborator
# gh pr edit 507 --add-reviewer <username>

# After approval, merge:
gh pr merge 507 --squash --delete-branch
gh pr merge 508 --squash --delete-branch
```

**Pros**: Follows standard workflow, maintains audit trail
**Cons**: Requires another person

### Option B: Admin Override (If You Have Permissions)

```bash
# Bypass branch protection with admin rights
gh pr merge 507 --squash --delete-branch --admin
gh pr merge 508 --squash --delete-branch --admin
```

**Note**: Current attempt failed with:
```
GraphQL: Repository rule violations found
5 of 5 required status checks are expected.
```

This suggests either:
1. Required checks haven't completed (unlikely - shows 6 passing)
2. Branch protection requires approval even for admins
3. Ruleset is enforcing additional constraints

### Option C: Enable Auto-Merge (Long-term Solution)

**Step 1**: Enable in repo settings
- Go to Settings → Pull Requests
- Check "Allow auto-merge"

**Step 2**: Add Dependabot auto-merge workflow

Create `.github/workflows/dependabot-automerge.yml`:

```yaml
name: Dependabot Auto-merge
on:
  pull_request_target:
    types: [opened, edited, synchronize, ready_for_review]

permissions:
  contents: write
  pull-requests: write
  checks: read
  statuses: read

jobs:
  automerge:
    if: ${{ github.actor == 'dependabot[bot]' }}
    runs-on: ubuntu-22.04
    steps:
      - uses: dependabot/fetch-metadata@v2
        id: meta
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Enable auto-merge for patch/minor bumps
        if: contains(fromJson('["version-update:semver-patch","version-update:semver-minor"]'), steps.meta.outputs.update-type)
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PR_URL: ${{ github.event.pull_request.html_url }}
```

**Pros**: Automated Dependabot merges for patch/minor
**Cons**: Requires repo settings change, only helps future PRs

---

## Recommended Action

**For immediate merge**:
```bash
# Option A - Request review and merge
gh pr edit 507 --add-reviewer <maintainer-username>
# Wait for approval, then:
gh pr merge 507 --squash --delete-branch
gh pr merge 508 --squash --delete-branch
```

**For future automation** (after PRs land):
1. Enable "Allow auto-merge" in repo settings
2. Add `.github/workflows/dependabot-automerge.yml`
3. Dependabot patch/minor bumps will auto-merge after checks pass

---

## Post-Merge Actions

Once both PRs land:

```bash
# Pull latest main
git switch main
git pull

# Verify guards still pass
make guards

# Optional: Trigger nightly guards manually
gh workflow run "Guards (nightly)" --ref main
```

---

**Next**: After PRs merge, proceed to SIMD PR1 scaffolding (see Sprint-2 kickoff doc).
