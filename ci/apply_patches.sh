#!/bin/bash
# Apply patches to the external BitNet C++ implementation
# This script is called by ci/fetch_bitnet_cpp.sh after downloading the C++ code

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCHES_DIR="$REPO_ROOT/patches"

# Default C++ path
CPP_PATH="${BITNET_CPP_PATH:-$HOME/.cache/bitnet_cpp}"

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

# Check if C++ implementation exists
if [[ ! -d "$CPP_PATH" ]]; then
    log_error "BitNet C++ implementation not found at: $CPP_PATH"
    log_error "Run ci/fetch_bitnet_cpp.sh first"
    exit 1
fi

# Check if patches directory exists
if [[ ! -d "$PATCHES_DIR" ]]; then
    log_info "No patches directory found - nothing to apply"
    exit 0
fi

# Count patches
PATCH_COUNT=$(find "$PATCHES_DIR" -name "*.patch" | wc -l)

if [[ $PATCH_COUNT -eq 0 ]]; then
    log_info "No patches found - C++ implementation will be used as-is"
    exit 0
fi

log_warn "Found $PATCH_COUNT patches to apply"
log_warn "Remember: patches should be avoided when possible"
log_warn "See patches/README.md for patch policy"

# Change to C++ directory
cd "$CPP_PATH"

# Check if we're in a git repository
if [[ ! -d ".git" ]]; then
    log_error "C++ implementation is not a git repository"
    log_error "Cannot apply patches safely"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet; then
    log_error "C++ implementation has uncommitted changes"
    log_error "Cannot apply patches safely"
    exit 1
fi

# Apply patches in order
APPLIED_COUNT=0
FAILED_COUNT=0

for patch_file in "$PATCHES_DIR"/*.patch; do
    if [[ ! -f "$patch_file" ]]; then
        continue
    fi
    
    patch_name=$(basename "$patch_file")
    log_info "Applying patch: $patch_name"
    
    # Check if patch has upstream issue reference
    if ! grep -q "issue" "$patch_file" && ! grep -q "Issue" "$patch_file"; then
        log_error "Patch $patch_name does not reference an upstream issue"
        log_error "All patches must reference upstream issues (see patches/README.md)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi
    
    # Try to apply the patch
    if git apply --check "$patch_file" 2>/dev/null; then
        git apply "$patch_file"
        log_info "Successfully applied: $patch_name"
        APPLIED_COUNT=$((APPLIED_COUNT + 1))
    else
        log_error "Failed to apply patch: $patch_name"
        log_error "Patch may be outdated or conflict with current C++ version"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        
        # Show what failed
        echo "Patch application error:"
        git apply "$patch_file" 2>&1 || true
    fi
done

# Summary
echo
log_info "Patch application summary:"
log_info "  Applied: $APPLIED_COUNT"
if [[ $FAILED_COUNT -gt 0 ]]; then
    log_error "  Failed: $FAILED_COUNT"
else
    log_info "  Failed: $FAILED_COUNT"
fi

# Exit with error if any patches failed
if [[ $FAILED_COUNT -gt 0 ]]; then
    log_error "Some patches failed to apply"
    log_error "Check patch compatibility with current C++ version"
    exit 1
fi

if [[ $APPLIED_COUNT -gt 0 ]]; then
    log_warn "Applied $APPLIED_COUNT patches to C++ implementation"
    log_warn "Consider contributing these changes upstream"
fi

log_info "Patch application completed successfully"