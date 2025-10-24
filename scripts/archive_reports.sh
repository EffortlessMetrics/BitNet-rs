#!/usr/bin/env bash
# scripts/archive_reports.sh - Archive historical docs/reports/ to docs/archive/reports/
#
# Usage:
#   ./scripts/archive_reports.sh [OPTIONS]
#
# Options:
#   --dry-run       Preview changes without executing
#   --rollback      Restore archived reports to docs/reports/
#   --skip-banner   Skip banner injection (migration only)
#   --help          Show this help message
#
# Examples:
#   # Preview migration
#   ./scripts/archive_reports.sh --dry-run
#
#   # Execute migration with banners
#   ./scripts/archive_reports.sh
#
#   # Rollback migration
#   ./scripts/archive_reports.sh --rollback

set -euo pipefail

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="${REPO_ROOT}/docs/reports"
ARCHIVE_DIR="${REPO_ROOT}/docs/archive/reports"
BANNER_TEMPLATE="${REPO_ROOT}/scripts/templates/archive_banner.md"
ARCHIVE_DATE="2025-10-23"

# Cross-reference files to update
CROSS_REF_FILES=(
    "COMPREHENSIVE_IMPLEMENTATION_REPORT.md"
    "DOCS_LINK_VALIDATION_REPORT.md"
    "ISSUE_254_SHAPE_MISMATCH_RESEARCH_REPORT.md"
    "CARGO_FEATURE_FLAG_AUDIT.md"
)

# Command-line flags
DRY_RUN=false
ROLLBACK=false
SKIP_BANNER=false

# Function: Show help
show_help() {
    sed -n '2,22p' "$0" | sed 's/^# //'
    exit 0
}

# Function: Get category and current doc for file
get_file_category() {
    local filename="$1"
    local category=""
    local current_doc=""
    local current_doc_name=""

    case "${filename}" in
        PR_*|PR422_*)
            category="PR Review Report"
            current_doc="../../PR_475_FINAL_SUCCESS_REPORT.md"
            current_doc_name="PR #475 Final Report"
            ;;
        ISSUE_*)
            category="Issue Resolution Report"
            current_doc="../explanation/specs/GITHUB_ISSUES_P1_SPECIFICATIONS.md"
            current_doc_name="Current Issue Specifications"
            ;;
        ALPHA_*|LAUNCH_*|SPRINT_*|VALIDATION_STATUS*)
            category="Status Report"
            current_doc="../../CLAUDE.md"
            current_doc_name="CLAUDE.md Project Reference"
            ;;
        TEST_*|COVERAGE_*|SECURITY_*|CROSSVAL_*)
            category="Validation Report"
            current_doc="../development/test-suite.md"
            current_doc_name="Current Test Suite Documentation"
            ;;
        DOCUMENTATION_*)
            category="Documentation Report"
            current_doc="../CONTRIBUTING-DOCS.md"
            current_doc_name="Documentation Contributing Guide"
            ;;
        FIXTURE_*|BENCHMARKING_*|INFRASTRUCTURE_*)
            category="Implementation Report"
            current_doc="../../PR_475_FINAL_SUCCESS_REPORT.md"
            current_doc_name="PR #475 Final Report"
            ;;
        *)
            category="Project Report"
            current_doc="../../CLAUDE.md"
            current_doc_name="CLAUDE.md Project Reference"
            ;;
    esac

    echo "${category}|${current_doc}|${current_doc_name}"
}

# Function: Generate banner for file
generate_banner() {
    local filename="$1"
    local metadata
    metadata="$(get_file_category "${filename}")"

    local category="${metadata%%|*}"
    local rest="${metadata#*|}"
    local current_doc="${rest%%|*}"
    local current_doc_name="${rest#*|}"

    # Read template and substitute variables
    sed -e "s|{ARCHIVE_DATE}|${ARCHIVE_DATE}|g" \
        -e "s|{REPORT_CATEGORY}|${category}|g" \
        -e "s|{CURRENT_DOC}|${current_doc}|g" \
        -e "s|{CURRENT_DOC_NAME}|${current_doc_name}|g" \
        "${BANNER_TEMPLATE}"
}

# Function: Inject banner into file
inject_banner() {
    local filepath="$1"
    local filename
    filename="$(basename "${filepath}")"

    if ${DRY_RUN}; then
        echo "  Would add banner to: ${filename}"
        return
    fi

    # Generate banner
    local banner
    banner="$(generate_banner "${filename}")"

    # Create temporary file with banner + original content
    local tmpfile
    tmpfile="$(mktemp)"
    echo "${banner}" > "${tmpfile}"
    cat "${filepath}" >> "${tmpfile}"

    # Replace original
    mv "${tmpfile}" "${filepath}"

    echo "  ✅ Injected banner: ${filename}"
}

# Function: Update cross-references
update_cross_references() {
    echo ""
    echo "Updating cross-references in root files..."

    for file in "${CROSS_REF_FILES[@]}"; do
        local filepath="${REPO_ROOT}/${file}"
        if [[ -f "${filepath}" ]]; then
            if ${DRY_RUN}; then
                echo "  Would update: ${file}"
                continue
            fi

            echo "  Updating: ${file}"

            # Replace docs/reports/ → docs/archive/reports/
            sed -i 's|docs/reports/|docs/archive/reports/|g' "${filepath}"

            # Add archive context note if not present
            if ! grep -q "Historical Archive" "${filepath}"; then
                # Insert note after first heading
                sed -i '0,/^#/a \\n> **Note**: References to `docs/archive/reports/` point to historical archived documentation.\n> For current status, see [CLAUDE.md](CLAUDE.md) and [PR #475](PR_475_FINAL_SUCCESS_REPORT.md).\n' "${filepath}"
            fi

            echo "  ✅ Updated: ${file}"
        fi
    done
}

# Function: Update lychee config
update_lychee_config() {
    local lychee_config="${REPO_ROOT}/.lychee.toml"

    echo ""
    echo "Updating .lychee.toml to exclude docs/archive/..."

    if ${DRY_RUN}; then
        echo "  Would add exclusion: docs/archive/"
        return
    fi

    # Add docs/archive/ to exclude list if not present
    if ! grep -q '"docs/archive/"' "${lychee_config}"; then
        # Find the exclude array and insert before the closing bracket
        sed -i '/exclude = \[/,/\]/{
            /\]/i\    "docs/archive/",  # Historical documentation - not maintained (archived 2025-10-23)
        }' "${lychee_config}"
        echo "  ✅ Added archive exclusion to .lychee.toml"
    else
        echo "  ⏭️  Archive already excluded in .lychee.toml"
    fi
}

# Function: Rollback migration
rollback_migration() {
    echo "Rolling back archive migration..."
    echo ""

    # Verify archive exists
    if [[ ! -d "${ARCHIVE_DIR}" ]]; then
        echo "❌ ERROR: Archive directory not found (nothing to rollback)"
        exit 1
    fi

    if ${DRY_RUN}; then
        echo "Would rollback migration:"
        echo "  - Move files from ${ARCHIVE_DIR} to ${REPORTS_DIR}"
        echo "  - Remove banners from files"
        echo "  - Restore cross-references"
        echo "  - Restore lychee config"
        echo "  - Remove empty archive directory"
        return
    fi

    # 1. Create reports directory if needed
    mkdir -p "${REPORTS_DIR}"

    # 2. Move files back (preserve git history)
    echo "Moving files back to docs/reports/..."
    for file in "${ARCHIVE_DIR}"/*.md; do
        if [[ -f "${file}" ]]; then
            git mv "${file}" "${REPORTS_DIR}/"
        fi
    done

    # 3. Remove banners from restored files
    echo ""
    echo "Removing banners from restored files..."
    cd "${REPORTS_DIR}"
    for file in *.md; do
        if [[ -f "${file}" ]]; then
            # Remove banner (everything from first > line to first --- line after it)
            sed -i '1,/^---$/{ /^> \*\*ARCHIVED DOCUMENT\*\*/,/^---$/d }' "${file}"
            echo "  ✅ Removed banner: ${file}"
        fi
    done
    cd "${REPO_ROOT}"

    # 4. Restore cross-references
    echo ""
    echo "Restoring cross-references..."
    for file in "${CROSS_REF_FILES[@]}"; do
        local filepath="${REPO_ROOT}/${file}"
        if [[ -f "${filepath}" ]]; then
            # Reverse the path change
            sed -i 's|docs/archive/reports/|docs/reports/|g' "${filepath}"
            # Remove archive context note (matches "References to `docs/archive/reports/`")
            sed -i '/^> \*\*Note\*\*: References to `docs\/\(archive\/\)\?reports\//,+2d' "${filepath}"
            echo "  ✅ Restored: ${file}"
        fi
    done

    # 5. Restore lychee config
    echo ""
    echo "Restoring .lychee.toml..."
    sed -i '/docs\/archive\//d' "${REPO_ROOT}/.lychee.toml"

    # 6. Remove empty archive directory
    rmdir "${ARCHIVE_DIR}" 2>/dev/null || true
    rmdir "${REPO_ROOT}/docs/archive" 2>/dev/null || true

    echo ""
    echo "✅ Rollback complete!"
    echo ""
    echo "Verify rollback:"
    echo "  git status"
    echo "  ls -la docs/reports/"
}

# Function: Execute migration
execute_migration() {
    echo "Archive Migration Script"
    echo "========================"
    echo ""
    echo "Source: ${REPORTS_DIR}"
    echo "Target: ${ARCHIVE_DIR}"
    echo "Dry-run: ${DRY_RUN}"
    echo ""

    # Pre-migration validation
    if [[ ! -d "${REPORTS_DIR}" ]]; then
        echo "❌ ERROR: docs/reports/ directory not found"
        exit 1
    fi

    if [[ ! -f "${BANNER_TEMPLATE}" ]]; then
        echo "❌ ERROR: Banner template not found: ${BANNER_TEMPLATE}"
        exit 1
    fi

    # Count files
    local file_count
    file_count=$(find "${REPORTS_DIR}" -maxdepth 1 -name "*.md" | wc -l)
    echo "Found ${file_count} markdown files to migrate"
    echo ""

    if [[ ${file_count} -eq 0 ]]; then
        echo "❌ ERROR: No markdown files found in docs/reports/"
        exit 1
    fi

    # Phase 1: Create archive directory
    echo "Phase 1: Creating archive directory..."
    if ${DRY_RUN}; then
        echo "  Would create: ${ARCHIVE_DIR}"
    else
        mkdir -p "${ARCHIVE_DIR}"
        echo "  ✅ Created: ${ARCHIVE_DIR}"
    fi

    # Phase 2: Migrate files
    echo ""
    echo "Phase 2: Migrating files (preserving git history)..."
    local migrated=0
    for file in "${REPORTS_DIR}"/*.md; do
        if [[ -f "${file}" ]]; then
            local filename
            filename="$(basename "${file}")"

            if ${DRY_RUN}; then
                echo "  Would move: ${filename}"
            else
                git mv "${file}" "${ARCHIVE_DIR}/"
                echo "  ✅ Moved: ${filename}"
            fi
            migrated=$((migrated + 1))
        fi
    done

    echo ""
    echo "Migrated ${migrated} files"

    # Phase 3: Inject banners
    if ! ${SKIP_BANNER}; then
        echo ""
        echo "Phase 3: Injecting tombstone banners..."
        for file in "${ARCHIVE_DIR}"/*.md; do
            if [[ -f "${file}" ]]; then
                inject_banner "${file}"
            fi
        done
    fi

    # Phase 4: Update cross-references
    update_cross_references

    # Phase 5: Update lychee config
    update_lychee_config

    # Phase 6: Remove empty reports directory
    if ! ${DRY_RUN}; then
        echo ""
        echo "Removing empty docs/reports/ directory..."
        rmdir "${REPORTS_DIR}" 2>/dev/null && echo "  ✅ Removed: docs/reports/" || echo "  ⏭️  Directory not empty or already removed"
    fi

    # Summary
    echo ""
    echo "========================================="
    echo "Migration Summary"
    echo "========================================="
    echo "Files migrated: ${migrated}"
    echo "Banners injected: ${SKIP_BANNER:-${migrated}}"
    echo "Cross-refs updated: ${#CROSS_REF_FILES[@]}"
    echo "Lychee config updated: Yes"
    echo ""

    if ${DRY_RUN}; then
        echo "✅ Dry-run complete (no changes made)"
        echo ""
        echo "To execute migration, run:"
        echo "  ./scripts/archive_reports.sh"
    else
        echo "✅ Migration complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Verify git status: git status"
        echo "  2. Review changes: git diff"
        echo "  3. Verify archive: ls -la docs/archive/reports/"
        echo "  4. Run link validation: lychee docs/ --exclude 'docs/archive/' --offline"
        echo ""
        echo "To rollback:"
        echo "  ./scripts/archive_reports.sh --rollback"
    fi
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --skip-banner)
            SKIP_BANNER=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Execute requested operation
if ${ROLLBACK}; then
    rollback_migration
else
    execute_migration
fi
