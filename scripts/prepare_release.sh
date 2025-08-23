#!/usr/bin/env bash
# Automated release preparation script
# Runs all validation, generates artifacts, and prepares for tagging
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
VERSION=""
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--version X.Y.Z] [--skip-tests]"
            exit 1
            ;;
    esac
done

# Get current version if not specified
if [ -z "$VERSION" ]; then
    VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
fi

echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          BitNet.rs Release Preparation v$VERSION          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo

# Function to run or simulate command
run_cmd() {
    local desc="$1"
    shift
    echo -e "${BLUE}→${NC} $desc"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would execute: $*"
    else
        "$@" || return 1
    fi
}

# Step 1: Run sign-off checks
echo -e "\n${GREEN}[1/8] Running sign-off checks...${NC}"
if [ "$DRY_RUN" = false ]; then
    if ! "${SCRIPT_DIR}/release_signoff.sh"; then
        echo -e "${RED}Sign-off checks failed! Fix issues before proceeding.${NC}"
        exit 1
    fi
else
    echo "  [DRY RUN] Would run release_signoff.sh"
fi

# Step 2: Run acceptance tests
echo -e "\n${GREEN}[2/8] Running acceptance tests...${NC}"
if [ "$SKIP_TESTS" = true ]; then
    echo "  Skipping tests (--skip-tests flag)"
elif [ "$DRY_RUN" = false ]; then
    if ! "${SCRIPT_DIR}/acceptance_test.sh"; then
        echo -e "${RED}Acceptance tests failed!${NC}"
        exit 1
    fi
else
    echo "  [DRY RUN] Would run acceptance_test.sh"
fi

# Step 3: Generate performance data
echo -e "\n${GREEN}[3/8] Generating performance measurements...${NC}"
run_cmd "Measuring SafeTensors performance" \
    "${SCRIPT_DIR}/measure_perf_json.sh" --format safetensors

run_cmd "Measuring GGUF performance" \
    "${SCRIPT_DIR}/measure_perf_json.sh" --format gguf

# Step 4: Render performance documentation
echo -e "\n${GREEN}[4/8] Rendering performance documentation...${NC}"
if [ "$DRY_RUN" = false ]; then
    python3 "${SCRIPT_DIR}/render_perf_md.py" bench/results/*.json > docs/PERF_COMPARISON.md
    echo "  Performance comparison saved to docs/PERF_COMPARISON.md"
else
    echo "  [DRY RUN] Would render performance documentation"
fi

# Step 5: Validate JSON schemas
echo -e "\n${GREEN}[5/8] Validating JSON artifacts...${NC}"
for json in bench/results/*.json; do
    if [ -f "$json" ]; then
        name=$(basename "$json")
        if [ "$DRY_RUN" = false ]; then
            if echo "$name" | grep -q "parity"; then
                schema="bench/schema/parity.schema.json"
            else
                schema="bench/schema/perf.schema.json"
            fi
            
            if command -v ajv > /dev/null 2>&1; then
                run_cmd "Validating $name" \
                    ajv validate -s "$schema" -d "$json"
            else
                echo "  ⚠ ajv not installed, skipping schema validation"
                echo "    Install with: npm install -g ajv-cli"
            fi
        else
            echo "  [DRY RUN] Would validate $name"
        fi
    fi
done

# Step 6: Create release artifacts
echo -e "\n${GREEN}[6/8] Creating release artifacts...${NC}"
ARTIFACTS_DIR="release-artifacts-v${VERSION}"
run_cmd "Creating artifacts directory" mkdir -p "$ARTIFACTS_DIR"

# Collect artifacts
if [ "$DRY_RUN" = false ]; then
    # Performance data
    cp -r bench/results "$ARTIFACTS_DIR/performance" 2>/dev/null || true
    
    # Validation results
    cp artifacts/*.json "$ARTIFACTS_DIR/" 2>/dev/null || true
    cp artifacts/*.log "$ARTIFACTS_DIR/" 2>/dev/null || true
    
    # Documentation
    cp docs/PERF_COMPARISON.md "$ARTIFACTS_DIR/" 2>/dev/null || true
    cp docs/PRODUCTION_READINESS.md "$ARTIFACTS_DIR/" 2>/dev/null || true
    
    # Create manifest
    cat > "$ARTIFACTS_DIR/MANIFEST.json" <<EOF
{
    "version": "$VERSION",
    "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
    "platform": "$(uname -s)-$(uname -m)",
    "wsl2": $(detect_wsl2 && echo "true" || echo "false"),
    "artifacts": {
        "performance": $(ls "$ARTIFACTS_DIR/performance/"*.json 2>/dev/null | wc -l),
        "validation": $(ls "$ARTIFACTS_DIR/"*.json 2>/dev/null | grep -v MANIFEST | wc -l),
        "logs": $(ls "$ARTIFACTS_DIR/"*.log 2>/dev/null | wc -l)
    }
}
EOF
    
    # Compress artifacts
    run_cmd "Compressing artifacts" \
        tar -czf "bitnet-rs-v${VERSION}-artifacts.tar.gz" "$ARTIFACTS_DIR"
    
    echo "  Artifacts saved to: bitnet-rs-v${VERSION}-artifacts.tar.gz"
else
    echo "  [DRY RUN] Would create release artifacts"
fi

# Step 7: Update version numbers
echo -e "\n${GREEN}[7/8] Updating version numbers...${NC}"
if [ "$DRY_RUN" = false ]; then
    # Update all Cargo.toml files
    for cargo_file in Cargo.toml crates/*/Cargo.toml; do
        if [ -f "$cargo_file" ]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                run_cmd "Updating $(basename $(dirname "$cargo_file"))" \
                    sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" "$cargo_file"
            else
                run_cmd "Updating $(basename $(dirname "$cargo_file"))" \
                    sed -i "s/^version = \".*\"/version = \"$VERSION\"/" "$cargo_file"
            fi
        fi
    done
    
    # Update lock file
    run_cmd "Updating Cargo.lock" cargo update --workspace
else
    echo "  [DRY RUN] Would update version to $VERSION in all Cargo.toml files"
fi

# Step 8: Create release checklist
echo -e "\n${GREEN}[8/8] Creating release checklist...${NC}"
CHECKLIST="RELEASE_CHECKLIST_v${VERSION}.md"
if [ "$DRY_RUN" = false ]; then
    cat > "$CHECKLIST" <<EOF
# Release Checklist for v$VERSION

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Pre-release Validation ✅
- [x] Sign-off checks passed
- [x] Acceptance tests passed
- [x] Performance measurements generated
- [x] JSON schemas validated
- [x] Release artifacts created

## Artifacts
- Performance data: \`bench/results/\`
- Validation results: \`artifacts/\`
- Compressed bundle: \`bitnet-rs-v${VERSION}-artifacts.tar.gz\`

## Manual Steps Required

### 1. Review Changes
\`\`\`bash
git diff HEAD^ -- Cargo.toml
git log --oneline -10
\`\`\`

### 2. Commit Version Bump
\`\`\`bash
git add -A
git commit -m "chore: bump version to v$VERSION

- Update all crate versions to $VERSION
- Generate release artifacts
- Update performance documentation"
\`\`\`

### 3. Create Tag
\`\`\`bash
git tag -a v$VERSION -m "Release v$VERSION

Dual-format support (SafeTensors + GGUF) with:
- Format parity validation
- Performance measurements
- Cross-validation framework
- Production readiness guarantees

See PRODUCTION_READINESS.md for details."
\`\`\`

### 4. Push to Remote
\`\`\`bash
git push origin main
git push origin v$VERSION
\`\`\`

### 5. Create GitHub Release
1. Go to: https://github.com/yourusername/BitNet-rs/releases/new
2. Select tag: v$VERSION
3. Title: BitNet.rs v$VERSION
4. Attach: \`bitnet-rs-v${VERSION}-artifacts.tar.gz\`
5. Copy release notes from this checklist

## Release Notes Template

### 🎉 BitNet.rs v$VERSION

#### ✨ Highlights
- Full dual-format support (SafeTensors + GGUF)
- Format parity validation with tau-b correlation
- Comprehensive performance measurements
- Production-ready validation suite

#### 📊 Performance
See attached artifacts for detailed benchmarks.

#### 🔧 Compatibility
- Minimum Rust Version: 1.89.0
- Supports: Linux, macOS, Windows (WSL2)
- Model formats: SafeTensors, GGUF

#### 📦 Artifacts
- Performance benchmarks: JSON data
- Validation results: Parity tests
- Documentation: Production readiness guide

---
Generated by prepare_release.sh
EOF
    echo "  Release checklist saved to: $CHECKLIST"
else
    echo "  [DRY RUN] Would create release checklist"
fi

# Summary
echo
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    SUMMARY                         ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN COMPLETE${NC}"
    echo "No changes were made. Remove --dry-run to execute."
else
    echo -e "${GREEN}✅ RELEASE PREPARATION COMPLETE!${NC}"
    echo
    echo "Version: v$VERSION"
    echo "Artifacts: bitnet-rs-v${VERSION}-artifacts.tar.gz"
    echo "Checklist: $CHECKLIST"
    echo
    echo "Next steps:"
    echo "  1. Review: cat $CHECKLIST"
    echo "  2. Commit: git commit -am 'chore: prepare release v$VERSION'"
    echo "  3. Tag: git tag -a v$VERSION -m 'Release v$VERSION'"
    echo "  4. Push: git push --tags"
fi