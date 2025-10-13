#!/bin/bash
# Manual release script for BitNet.rs
# Handles version bumping, tagging, and release preparation

set -euo pipefail

echo "🚀 BitNet.rs Release Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
DRY_RUN=false
SKIP_TESTS=false
SKIP_VALIDATION=false
RELEASE_TYPE=""
VERSION=""

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <VERSION|RELEASE_TYPE>

Release BitNet.rs with proper version management and validation.

Arguments:
  VERSION        Specific version to release (e.g., 1.0.0)
  RELEASE_TYPE   Type of release: major, minor, patch, pre

Options:
  -d, --dry-run           Perform a dry run without making changes
  -s, --skip-tests        Skip running tests
  -v, --skip-validation   Skip pre-release validation
  -h, --help             Show this help message

Examples:
  $0 1.0.0                # Release version 1.0.0
  $0 patch                # Bump patch version
  $0 minor                # Bump minor version
  $0 --dry-run 1.0.1      # Dry run for version 1.0.1

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -v|--skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$VERSION" ]]; then
                    if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
                        VERSION="$1"
                    elif [[ "$1" =~ ^(major|minor|patch|pre)$ ]]; then
                        RELEASE_TYPE="$1"
                    else
                        print_error "Invalid version or release type: $1"
                        show_usage
                        exit 1
                    fi
                else
                    print_error "Multiple versions specified"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$VERSION" && -z "$RELEASE_TYPE" ]]; then
        print_error "Version or release type is required"
        show_usage
        exit 1
    fi
}

# Get current version from Cargo.toml
get_current_version() {
    grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'
}

# Calculate next version based on release type
calculate_next_version() {
    local current_version="$1"
    local release_type="$2"

    # Parse current version
    local major minor patch pre
    if [[ "$current_version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-(.+))?$ ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        patch="${BASH_REMATCH[3]}"
        pre="${BASH_REMATCH[5]}"
    else
        print_error "Invalid current version format: $current_version"
        exit 1
    fi

    # Calculate next version
    case "$release_type" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "${major}.$((minor + 1)).0"
            ;;
        patch)
            echo "${major}.${minor}.$((patch + 1))"
            ;;
        pre)
            if [[ -n "$pre" ]]; then
                # Increment pre-release
                if [[ "$pre" =~ ^(.+)\.([0-9]+)$ ]]; then
                    echo "${major}.${minor}.${patch}-${BASH_REMATCH[1]}.$((BASH_REMATCH[2] + 1))"
                else
                    echo "${major}.${minor}.${patch}-${pre}.1"
                fi
            else
                echo "${major}.${minor}.${patch}-pre.1"
            fi
            ;;
        *)
            print_error "Invalid release type: $release_type"
            exit 1
            ;;
    esac
}

# Validate git repository state
validate_git_state() {
    print_status "Validating git repository state..."

    # Check if we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_error "Uncommitted changes detected. Please commit or stash them."
        exit 1
    fi

    # Check if we're on main/master branch
    local current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
        print_warning "Not on main/master branch (current: $current_branch)"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check if we're up to date with remote
    if git remote get-url origin >/dev/null 2>&1; then
        git fetch origin
        local local_commit=$(git rev-parse HEAD)
        local remote_commit=$(git rev-parse origin/"$current_branch" 2>/dev/null || echo "")

        if [[ -n "$remote_commit" && "$local_commit" != "$remote_commit" ]]; then
            print_error "Local branch is not up to date with remote"
            exit 1
        fi
    fi

    print_success "Git repository state is valid"
}

# Run pre-release validation
run_validation() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        print_warning "Skipping pre-release validation"
        return
    fi

    print_status "Running pre-release validation..."

    # Run quality checks
    if [[ -x "scripts/quality-check.sh" ]]; then
        ./scripts/quality-check.sh
    else
        print_warning "Quality check script not found or not executable"
    fi

    # Run security audit
    if [[ -x "scripts/security-audit.sh" ]]; then
        ./scripts/security-audit.sh
    else
        print_warning "Security audit script not found or not executable"
    fi

    # Run documentation validation
    if [[ -x "scripts/docs-validation.sh" ]]; then
        ./scripts/docs-validation.sh
    else
        print_warning "Documentation validation script not found or not executable"
    fi

    print_success "Pre-release validation completed"
}

# Update version in all Cargo.toml files
update_version() {
    local new_version="$1"

    print_status "Updating version to $new_version..."

    # Update workspace Cargo.toml
    sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" Cargo.toml

    # Update all crate Cargo.toml files
    find crates -name Cargo.toml -exec sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" {} \;

    # Update version references in dependencies
    find . -name Cargo.toml -exec sed -i.bak "s/bitnet-[a-z-]* = { path = \"[^\"]*\", version = \"[^\"]*\"/bitnet-&, version = \"$new_version\"/g" {} \;

    # Clean up backup files
    find . -name "*.bak" -delete

    print_success "Version updated to $new_version"
}

# Update CHANGELOG.md
update_changelog() {
    local new_version="$1"
    local release_date=$(date +%Y-%m-%d)

    print_status "Updating CHANGELOG.md..."

    if [[ -f "CHANGELOG.md" ]]; then
        # Replace [Unreleased] with version and date
        sed -i.bak "s/## \[Unreleased\]/## [Unreleased]\n\n## [$new_version] - $release_date/" CHANGELOG.md
        rm -f CHANGELOG.md.bak
        print_success "CHANGELOG.md updated"
    else
        print_warning "CHANGELOG.md not found"
    fi
}

# Create git tag
create_git_tag() {
    local version="$1"
    local tag="v$version"

    print_status "Creating git tag $tag..."

    if $DRY_RUN; then
        print_status "DRY RUN: Would create tag $tag"
        return
    fi

    # Commit version changes
    git add .
    git commit -m "chore: bump version to $version"

    # Create annotated tag
    git tag -a "$tag" -m "Release $version"

    print_success "Git tag $tag created"
}

# Build release artifacts
build_artifacts() {
    local version="$1"

    print_status "Building release artifacts..."

    if $DRY_RUN; then
        print_status "DRY RUN: Would build release artifacts"
        return
    fi

    # Build release binaries
    cargo build --release --all-features

    # Run tests if not skipped
    if [[ "$SKIP_TESTS" != "true" ]]; then
        cargo test --release --all-features
    fi

    # Create release directory
    local release_dir="release-$version"
    mkdir -p "$release_dir"

    # Copy binaries
    cp target/release/bitnet "$release_dir/" 2>/dev/null || true
    cp target/release/server "$release_dir/bitnet-server" 2>/dev/null || true

    # Copy documentation
    cp README.md CHANGELOG.md "$release_dir/"
    cp LICENSE* "$release_dir/" 2>/dev/null || true

    # Create archive
    tar czf "$release_dir.tar.gz" "$release_dir"

    # Generate checksums
    sha256sum "$release_dir.tar.gz" > "$release_dir.tar.gz.sha256"

    print_success "Release artifacts built: $release_dir.tar.gz"
}

# Push changes and trigger CI
push_release() {
    local version="$1"
    local tag="v$version"

    if $DRY_RUN; then
        print_status "DRY RUN: Would push tag $tag to trigger release"
        return
    fi

    print_status "Pushing tag $tag to trigger automated release..."

    # Push commits and tags
    git push origin HEAD
    git push origin "$tag"

    print_success "Tag pushed. Automated release pipeline should start shortly."
    print_status "Monitor the release at: https://github.com/microsoft/BitNet/actions"
}

# Main release function
main() {
    parse_args "$@"

    print_status "Starting release process..."

    # Determine version
    local current_version=$(get_current_version)
    local target_version

    if [[ -n "$VERSION" ]]; then
        target_version="$VERSION"
    else
        target_version=$(calculate_next_version "$current_version" "$RELEASE_TYPE")
    fi

    print_status "Current version: $current_version"
    print_status "Target version: $target_version"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY RUN MODE - No changes will be made"
    fi

    # Confirm release
    if [[ "$DRY_RUN" != "true" ]]; then
        echo
        read -p "Proceed with release $target_version? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Release cancelled"
            exit 0
        fi
    fi

    # Execute release steps
    validate_git_state
    run_validation

    if [[ "$DRY_RUN" != "true" ]]; then
        update_version "$target_version"
        update_changelog "$target_version"
    fi

    create_git_tag "$target_version"
    build_artifacts "$target_version"
    push_release "$target_version"

    print_success "🎉 Release $target_version completed successfully!"

    if [[ "$DRY_RUN" != "true" ]]; then
        echo
        print_status "Next steps:"
        print_status "1. Monitor the automated release pipeline"
        print_status "2. Verify packages are published to registries"
        print_status "3. Update documentation and announce the release"
        print_status "4. Create post-release tasks if needed"
    fi
}

# Run main function with all arguments
main "$@"
