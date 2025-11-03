#!/bin/bash
# Patch Policy Enforcement Script
# This script enforces the patch policy in CI/CD

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if patches directory exists and has content
check_patches_directory() {
    local patches_dir="patches"

    if [[ ! -d "$patches_dir" ]]; then
        log_info "✅ No patches directory found - policy compliant"
        return 0
    fi

    # Count patch files
    local patch_count=$(find "$patches_dir" -name "*.patch" | wc -l)

    if [[ $patch_count -eq 0 ]]; then
        log_info "✅ Patches directory is empty - policy compliant"
        return 0
    fi

    log_warn "Found $patch_count patch file(s) - checking policy compliance..."
    return 1
}

# Validate individual patch files
validate_patch_files() {
    local patches_dir="patches"
    local policy_violations=0

    for patch_file in "$patches_dir"/*.patch; do
        if [[ ! -f "$patch_file" ]]; then
            continue
        fi

        local patch_name=$(basename "$patch_file")
        log_info "Checking patch: $patch_name"

        # Check if patch has upstream issue reference
        if ! grep -qi "issue\|bug\|upstream\|microsoft/bitnet" "$patch_file"; then
            log_error "❌ Patch '$patch_name' does not reference an upstream issue"
            log_error "   All patches must reference an upstream issue in Microsoft/BitNet repository"
            policy_violations=$((policy_violations + 1))
        fi

        # Check patch age (warn if older than 90 days)
        local patch_age_days=$(( ($(date +%s) - $(stat -c %Y "$patch_file" 2>/dev/null || stat -f %m "$patch_file" 2>/dev/null || echo 0)) / 86400 ))
        if [[ $patch_age_days -gt 90 ]]; then
            log_warn "⚠️  Patch '$patch_name' is $patch_age_days days old"
            log_warn "   Consider checking if upstream issue has been resolved"
        fi

        # Check if patch is minimal (basic heuristic)
        local patch_lines=$(wc -l < "$patch_file")
        if [[ $patch_lines -gt 100 ]]; then
            log_warn "⚠️  Patch '$patch_name' is large ($patch_lines lines)"
            log_warn "   Consider if this change should be contributed upstream instead"
        fi
    done

    return $policy_violations
}

# Check README for patch documentation
check_patch_documentation() {
    local patches_readme="patches/README.md"

    if [[ ! -f "$patches_readme" ]]; then
        log_error "❌ patches/README.md not found"
        log_error "   Patches directory must have documentation"
        return 1
    fi

    # Check if README documents current patches
    local documented_patches=0
    local actual_patches=$(find patches -name "*.patch" | wc -l)

    if [[ $actual_patches -gt 0 ]]; then
        # Look for patch documentation in README
        while IFS= read -r line; do
            if [[ "$line" =~ \.patch ]]; then
                documented_patches=$((documented_patches + 1))
            fi
        done < "$patches_readme"

        if [[ $documented_patches -lt $actual_patches ]]; then
            log_warn "⚠️  Not all patches are documented in README"
            log_warn "   Found $actual_patches patches but only $documented_patches documented"
        fi
    fi

    log_info "✅ Patch documentation check complete"
    return 0
}

# Create GitHub issue for patch tracking
create_patch_tracking_issue() {
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log_warn "GITHUB_TOKEN not set - cannot create tracking issue"
        return 0
    fi

    local patch_count=$(find patches -name "*.patch" | wc -l)
    if [[ $patch_count -eq 0 ]]; then
        return 0
    fi

    log_info "Creating patch tracking issue..."

    # Create issue body
    local issue_body="# Patch Policy Violation Detected

This issue was automatically created because patches were found in the repository.

## Current Patches

"

    for patch_file in patches/*.patch; do
        if [[ -f "$patch_file" ]]; then
            local patch_name=$(basename "$patch_file")
            issue_body+="- \`$patch_name\`
"
        fi
    done

    issue_body+="
## Required Actions

For each patch above:

1. **Verify upstream issue exists**: Each patch must reference an issue in the Microsoft/BitNet repository
2. **Check if upstream is fixed**: Patches should be removed when upstream fixes are available
3. **Document rationale**: Update patches/README.md with clear justification
4. **Plan removal**: Set timeline for patch removal

## Patch Policy

Our policy strongly prefers **no patches**. Patches should only exist for:

- Critical FFI compatibility issues
- Urgent bug fixes not yet available upstream
- Minimal build system integration needs

## Next Steps

- [ ] Review each patch for necessity
- [ ] Check upstream repository for fixes
- [ ] Update patch documentation
- [ ] Remove unnecessary patches
- [ ] Close this issue when patches are resolved

---
*This issue was created automatically by the patch policy enforcement system.*"

    # Use GitHub CLI or API to create issue (placeholder)
    log_info "Issue body prepared (GitHub integration would create issue here)"

    return 0
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Enforce patch policy for BitNet.rs repository.

This script checks:
1. Whether patches directory exists and contains patches
2. If patches reference upstream issues
3. If patches are documented properly
4. If patches are older than policy limits

OPTIONS:
    --strict            Fail CI if any patches exist
    --create-issue      Create GitHub issue for patch tracking
    --help              Show this help message

EXIT CODES:
    0   No policy violations
    1   Policy violations found
    2   Critical policy violations (with --strict)

EXAMPLES:
    $0                  # Check policy compliance
    $0 --strict         # Fail if any patches exist
    $0 --create-issue   # Create tracking issue for violations
EOF
}

# Parse command line arguments
STRICT_MODE=false
CREATE_ISSUE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --create-issue)
            CREATE_ISSUE=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "🔍 Checking patch policy compliance..."

    local violations=0

    # Check if patches exist
    if check_patches_directory; then
        log_info "✅ Patch policy compliant - no patches found"
        return 0
    fi

    # If we get here, patches exist - validate them
    if ! validate_patch_files; then
        violations=$((violations + $?))
    fi

    # Check documentation
    if ! check_patch_documentation; then
        violations=$((violations + 1))
    fi

    # Create tracking issue if requested
    if [[ "$CREATE_ISSUE" == true ]]; then
        create_patch_tracking_issue
    fi

    # Report results
    if [[ $violations -gt 0 ]]; then
        log_error "❌ Found $violations patch policy violation(s)"

        if [[ "$STRICT_MODE" == true ]]; then
            log_error "💥 STRICT MODE: Failing CI due to patch policy violations"
            log_error ""
            log_error "Our policy strongly discourages patches. Consider:"
            log_error "  1. Contributing fixes upstream to Microsoft/BitNet"
            log_error "  2. Using wrapper functions instead of patches"
            log_error "  3. Adapting to existing C++ API in Rust code"
            log_error ""
            exit 2
        else
            log_warn "⚠️  Patch policy violations found but not failing CI"
            log_warn "   Please address these violations as soon as possible"
            exit 1
        fi
    else
        log_info "✅ All patch policy checks passed"
        exit 0
    fi
}

# Run main function
main "$@"
