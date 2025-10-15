# ADR-002: Manual Branch Protection Configuration

**Status**: ACCEPTED
**Date**: 2025-10-15
**Context**: Issue #465 (v0.1.0-mvp Release Polish)
**Related**: AC5 (GitHub Branch Protection Configuration)

---

## Context

Issue #465 requires GitHub branch protection to enforce Model Gates (CPU) status checks before merging to `main`. This ensures honest compute receipts are validated in CI, blocking PRs with mocked or empty receipts.

Two implementation approaches exist for branch protection configuration:

### Option 1: Manual Configuration (GitHub UI)
- **Method**: Repository admin configures protection rules via GitHub Settings UI
- **Pros**: Simple, immediate, no development overhead, visual confirmation
- **Cons**: Manual operation, not scriptable, requires admin coordination

### Option 2: Automated Configuration (GitHub API)
- **Method**: Create `xtask configure-branch-protection` command using GitHub API
- **Pros**: Repeatable, scriptable, self-service for admins
- **Cons**: Development overhead (~2 hours), requires admin token with `repo` scope, security considerations

---

## Decision

**Manual configuration for MVP (Option 1), with automated command as future enhancement.**

---

## Rationale

### 1. MVP Timeline Prioritization
- **Manual Setup Time**: ~5 minutes for admin to configure via GitHub UI
- **Automated Development Time**: ~2 hours for xtask command + testing + error handling
- **MVP Focus**: Speed to release, not infrastructure automation
- **Cost-Benefit**: 2 hours development for 5-minute operation is not justified for MVP

### 2. One-Time Operation Characteristic
- **Frequency**: Branch protection is configured once per repository/branch
- **Repeatability**: Not a frequent operation requiring automation
- **Maintenance**: Changes to protection rules are rare (governance updates)
- **Post-MVP**: Automated command can be added if multi-repo management needed

### 3. Admin Access Security
- **Token Scope**: Automated configuration requires GitHub token with `repo` scope (write access)
- **Secret Management**: Storing admin tokens introduces security risk
- **Manual Verification**: Admin can visually confirm protection rules in GitHub UI
- **Audit Trail**: GitHub audit log captures manual configuration changes

### 4. Simplicity and Error Handling
- **GitHub UI**: Built-in validation, error messages, visual feedback
- **API Complexity**: Requires error handling for rate limits, permission errors, invalid inputs
- **Debugging**: Manual configuration easier to troubleshoot than API failures
- **Documentation**: GitHub UI steps are self-documenting with screenshots

### 5. Deferrable Automation
- **Not Blocking**: Manual configuration does not block v0.1.0-mvp release
- **Future Enhancement**: Automated command can be added in v0.2.0 if needed
- **Use Case**: Automation valuable for multi-repo organizations (not single-repo MVP)
- **Incremental Value**: Manual process adequate for BitNet.rs current scale

---

## Consequences

### Positive
- ✅ **Fast Setup**: Admin configures in ~5 minutes via GitHub UI
- ✅ **No Development Overhead**: 2 hours saved on automation code
- ✅ **Visual Confirmation**: Admin sees protection rules directly in GitHub UI
- ✅ **Simple Troubleshooting**: GitHub UI provides built-in error messages
- ✅ **Security**: No admin token storage or management required

### Negative
- ⚠️ **Admin Dependency**: Requires repository admin to manually configure
- ⚠️ **Not Scriptable**: Manual steps cannot be automated in CI/CD
- ⚠️ **Timeline Risk**: Admin coordination may delay MVP release (mitigated with documentation)
- ⚠️ **Multi-Repo Limitation**: Manual process does not scale to multiple repositories

### Mitigation Strategies
1. **Admin Documentation**: Provide step-by-step GitHub UI configuration guide with screenshots
2. **Timeline Coordination**: Request admin setup early in MVP timeline (AC5)
3. **Verification Commands**: Use `gh api` to verify protection rules programmatically
4. **Future Automation**: Document automated command design for post-MVP implementation

---

## Alternatives Considered

### Alternative 1: Automated Configuration (xtask command)
**Deferred**: Development overhead not justified for one-time operation. Can be added post-MVP if needed.

**Implementation Sketch** (for future reference):
```rust
// xtask/src/branch_protection.rs (future work)

use anyhow::{Context, Result};
use octocrab::Octocrab;

pub async fn configure_branch_protection(
    owner: &str,
    repo: &str,
    branch: &str,
    required_checks: &[&str],
    token: &str,
) -> Result<()> {
    let client = Octocrab::builder()
        .personal_token(token.to_string())
        .build()?;

    client
        .repos(owner, repo)
        .update_branch_protection(branch)
        .required_status_checks(required_checks)
        .enforce_admins(false)
        .required_approving_review_count(1)
        .dismiss_stale_reviews(true)
        .send()
        .await
        .context("Failed to configure branch protection")?;

    println!("✅ Branch protection configured for {}/{}/{}", owner, repo, branch);
    Ok(())
}

// Usage:
// cargo run -p xtask -- configure-branch-protection \
//   --owner EffortlessMetrics \
//   --repo BitNet-rs \
//   --branch main \
//   --require-check "Model Gates (CPU) / cpu-receipt-gate" \
//   --require-check "Model Gates (CPU) / gate-summary" \
//   --token $GITHUB_TOKEN
```

**Effort**: ~2 hours (command implementation, error handling, testing, documentation)

### Alternative 2: Manual with Shell Script
**Rejected**: Similar complexity to xtask command, still requires admin token, no advantage over GitHub UI.

### Alternative 3: Terraform/Infrastructure-as-Code
**Rejected**: Overkill for single-repo configuration, adds external dependency, no benefit for MVP.

---

## Implementation Details

### Manual Configuration Steps (Admin)

1. **Navigate to Branch Protection Settings**:
   - URL: `https://github.com/EffortlessMetrics/BitNet-rs/settings/branches`
   - Click "Add rule" or edit existing "main" branch rule

2. **Configure Protection Rules**:
   - Branch name pattern: `main`
   - ☑️ Require status checks to pass before merging
   - ☑️ Require branches to be up to date before merging
   - Search for: "Model Gates (CPU)"
   - Select:
     - ☑️ `Model Gates (CPU) / cpu-receipt-gate`
     - ☑️ `Model Gates (CPU) / gate-summary`
   - ☑️ Require approval before merging (1 reviewer)
   - ☐ Allow force pushes (disabled)
   - ☐ Allow deletions (disabled)

3. **Save Configuration**:
   - Click "Create" or "Save changes"

### Verification

```bash
# Check branch protection status (requires GitHub CLI authentication)
gh api repos/EffortlessMetrics/BitNet-rs/branches/main/protection | \
  jq '.required_status_checks.contexts'

# Expected output:
# [
#   "Model Gates (CPU) / cpu-receipt-gate",
#   "Model Gates (CPU) / gate-summary"
# ]
```

---

## Documentation

### Admin Guide
- **Location**: `docs/ci/branch-protection.md`
- **Content**: Step-by-step GitHub UI configuration with screenshots
- **Verification**: `gh api` command to check protection status
- **Troubleshooting**: Common issues (status checks not appearing, merge button enabled)

### Smoke Test
- **Location**: `docs/ci/branch-protection.md` (smoke test section)
- **Purpose**: Verify mocked receipts are blocked by branch protection
- **Procedure**: Create PR with mocked receipt, verify CI fails, verify merge blocked

---

## References

- **Issue #465**: CPU Path Followup (v0.1.0-mvp Release Polish)
- **AC5**: GitHub Branch Protection Configuration
- **AC6**: Smoke Test with Mocked Receipt
- **GitHub API**: [Branch Protection Endpoint](https://docs.github.com/en/rest/branches/branch-protection)
- **Model Gates Workflow**: `.github/workflows/model-gates.yml`

---

## Changelog

- **2025-10-15**: Initial decision for v0.1.0-mvp manual configuration
