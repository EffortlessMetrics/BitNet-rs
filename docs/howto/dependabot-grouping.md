# Dependabot Dependency Grouping

**Status**: Optional enhancement for future consideration
**Impact**: Reduces PR noise by grouping related dependency updates

---

## Current Configuration

`.github/dependabot.yml` (from PR #507):
- Weekly Cargo updates (max 5 PRs)
- Weekly GitHub Actions updates (max 5 PRs)
- No grouping (each dependency = 1 PR)

---

## Why Add Grouping?

**Problem**: 5 individual PRs per week can create noise and CI cost.

**Solution**: Group related dependencies into single PRs:
- `tokio-ecosystem`: tokio*, bytes, hyper*, h2, mio
- `serde-ecosystem`: serde*
- `github-actions`: actions/*, dtolnay/*, Swatinem/*

**Result**: ~2-3 grouped PRs/week instead of 5 individual PRs.

---

## How to Enable Grouping

Edit `.github/dependabot.yml` and add `groups:` sections:

```yaml
version: 2
updates:
  - package-ecosystem: cargo
    directory: /
    schedule:
      interval: weekly
    open-pull-requests-limit: 5
    labels:
      - automation
      - dependencies
    # Add these groups:
    groups:
      tokio-ecosystem:
        patterns:
          - "tokio*"
          - "bytes"
          - "hyper*"
          - "h2"
          - "mio"
      serde-ecosystem:
        patterns:
          - "serde*"
      tracing-ecosystem:
        patterns:
          - "tracing*"
      test-dependencies:
        patterns:
          - "proptest*"
          - "criterion*"
          - "quickcheck*"
    # Optional: conventional commit prefixes
    commit-message:
      prefix: "deps"
      include: "scope"

  - package-ecosystem: github-actions
    directory: /
    schedule:
      interval: weekly
    open-pull-requests-limit: 5
    labels:
      - automation
      - dependencies
    # Add this group:
    groups:
      github-actions:
        patterns:
          - "actions/*"
          - "Swatinem/*"
          - "dtolnay/*"
          - "taiki-e/*"
    commit-message:
      prefix: "ci"
```

---

## When to Apply

**Wait 2-3 weeks** to observe Dependabot PR volume:

- **Low volume** (<3 PRs/week): Grouping not needed, adds complexity
- **Medium volume** (3-5 PRs/week): Consider grouping
- **High volume** (5+ PRs/week): Definitely group

---

## Testing the Config

After editing `.github/dependabot.yml`:

1. Commit changes to a branch
2. Open a PR
3. Wait for next Dependabot cycle (weekly)
4. Observe grouped PRs

---

## Auto-Merge Integration

If you later enable auto-merge (see `pr-merge-instructions-2025-11-11.md`), grouped PRs work the same way:
- Patch/minor grouped updates auto-merge
- Major updates require manual review

---

**Reference**: PR #507 added the base Dependabot config. This doc describes optional grouping for later.
