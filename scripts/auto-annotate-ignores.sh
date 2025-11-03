#!/usr/bin/env bash
set -euo pipefail

# Auto-annotation tool for bulk migration
# Usage: ./scripts/auto-annotate-ignores.sh [--dry-run] [--file FILE]

DRY_RUN="${DRY_RUN:-true}"
TARGET_FILE="${TARGET_FILE:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --file)
            TARGET_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--file FILE]"
            exit 1
            ;;
    esac
done

# Define helper functions inline
extract_context() {
    local file="$1"
    local line="$2"
    local context_lines=10
    sed -n "$((line - context_lines)),$((line + 5))p" "$file" 2>/dev/null || echo ""
}

categorize_ignore() {
    local file="$1"
    local line="$2"
    local context="$3"

    local categories=()
    local confidence=0
    local suggested_reason=""
    local primary_reason=""

    # Issue-blocked detection (highest priority)
    if echo "$context" | grep -qEi 'issue\s*#?[0-9]+'; then
        local issue_num=$(echo "$context" | grep -oEi 'issue\s*#?[0-9]+' | head -1 | grep -oE '[0-9]+')
        categories+=("issue-blocked")
        confidence=$((confidence + 30))
        primary_reason="Issue #${issue_num}: "
    fi

    # Flaky detection (high priority, distinct marker)
    if echo "$context" | grep -qEi '(flaky|non-deterministic|race|timeout|intermittent)'; then
        categories+=("flaky")
        confidence=$((confidence + 25))
        if [ -z "$primary_reason" ]; then
            primary_reason="FLAKY: "
        fi
    fi

    # GPU detection
    if echo "$file" | grep -qE 'gpu_' || echo "$context" | grep -qEi '(gpu|cuda|device)'; then
        categories+=("gpu")
        confidence=$((confidence + 25))
        if [ -z "$primary_reason" ]; then
            primary_reason="gpu: "
        fi
    fi

    # Slow/performance detection
    if echo "$context" | grep -qEi '(slow|performance|benchmark|timing)'; then
        categories+=("slow")
        confidence=$((confidence + 20))
        if [ -z "$primary_reason" ]; then
            primary_reason="slow: "
        fi
    fi

    # Model/fixture detection
    if echo "$context" | grep -qE '(BITNET_GGUF|CROSSVAL_GGUF|\.gguf|models?/|fixture)'; then
        categories+=("requires-model")
        confidence=$((confidence + 20))
        if [ -z "$primary_reason" ]; then
            primary_reason="requires: "
        fi
    fi

    # Network detection
    if echo "$context" | grep -qEi '(network|download|fetch|api|huggingface)'; then
        categories+=("network")
        confidence=$((confidence + 20))
        if [ -z "$primary_reason" ]; then
            primary_reason="network: "
        fi
    fi

    # Quantization detection
    if echo "$context" | grep -qEi '(i2s|tl1|tl2|qk256|quantization)'; then
        categories+=("quantization")
        confidence=$((confidence + 20))
        if [ -z "$primary_reason" ]; then
            primary_reason="quantization: "
        fi
    fi

    # Parity/crossval detection
    if echo "$context" | grep -qEi '(parity|crossval|reference|accuracy)'; then
        categories+=("parity")
        confidence=$((confidence + 20))
        if [ -z "$primary_reason" ]; then
            primary_reason="parity: "
        fi
    fi

    # TODO/placeholder detection
    if echo "$context" | grep -qEi '(todo|fixme|wip|placeholder|unimplemented)'; then
        categories+=("todo")
        confidence=$((confidence + 15))
        if [ -z "$primary_reason" ]; then
            primary_reason="TODO: "
        fi
    fi

    # Use primary reason if set, otherwise use first category
    if [ -n "$primary_reason" ]; then
        suggested_reason="$primary_reason"
    else
        # Default to unknown if no matches
        categories+=("unknown")
        confidence=0
        suggested_reason="FIXME: add reason - "
    fi

    # Emit result
    echo "${file}:${line}|$(IFS=,; echo "${categories[*]}")|${confidence}|${suggested_reason}"
}

find_bare_ignores() {
    rg '#\[ignore\]' --type rust crates/ tests/ xtask/ -n --color never 2>/dev/null || true
}

annotate_file() {
    local file="$1"
    local temp_file="${file}.ignore-tmp"

    echo "Processing: $file"

    # Find bare ignores and annotate
    local modified=false
    local changes=0

    while IFS= read -r grep_line; do
        # Parse grep output: line_number:content
        local line="${grep_line%%:*}"
        local content="${grep_line#*:}"

        # Check if line has bare #[ignore] (without = "reason" syntax)
        if echo "$content" | grep -qE '#\[ignore\]' && ! echo "$content" | grep -qE '#\[ignore\s*=\s*"'; then
            modified=true

            # Get suggested annotation
            local context=$(extract_context "$file" "$line")
            local result=$(categorize_ignore "$file" "$line" "$context")
            IFS='|' read -r location categories confidence suggested_reason <<< "$result"

            # Replace line if confidence >= 70%
            if [ "$confidence" -ge 70 ]; then
                if [ "$DRY_RUN" = "true" ]; then
                    echo "  Line ${line}: Would annotate with \"${suggested_reason}...\" (confidence: ${confidence}%)"
                else
                    echo "  Line ${line}: Annotating with \"${suggested_reason}...\" (confidence: ${confidence}%)"
                    # Perform replacement - replace #[ignore] with #[ignore = "reason"], preserving any comment
                    sed -i "${line}s|#\[ignore\]|#[ignore = \"${suggested_reason}...\"]|" "$file"
                fi
                changes=$((changes + 1))
            else
                echo "  Line ${line}: LOW CONFIDENCE (${confidence}%) - manual review needed"
            fi
        fi
    done < <(grep -n '#\[ignore\]' "$file" 2>/dev/null || true)

    if [ "$changes" -gt 0 ]; then
        if [ "$DRY_RUN" = "false" ]; then
            # Format with rustfmt
            rustfmt "$file" 2>/dev/null || true
            echo "  ✅ Annotated ${changes} ignores and formatted"
        else
            echo "  📋 Would annotate ${changes} ignores"
        fi
    else
        echo "  ℹ️  No high-confidence annotations found"
    fi
}

# Main execution
if [ -n "$TARGET_FILE" ]; then
    annotate_file "$TARGET_FILE"
else
    # Process all files with bare ignores
    while IFS=: read -r file _; do
        annotate_file "$file"
    done < <(find_bare_ignores | awk -F: '{print $1}' | sort -u)
fi

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "DRY RUN complete. Run with DRY_RUN=false to apply changes."
fi
