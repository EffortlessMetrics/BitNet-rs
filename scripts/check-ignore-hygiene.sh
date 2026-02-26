#!/usr/bin/env bash
set -euo pipefail

# BitNet-rs #[ignore] Annotation Hygiene Checker
# Detects bare #[ignore] annotations and suggests categorized reasons
# Usage: MODE=full|diff|suggest|enforce ./scripts/check-ignore-hygiene.sh

# Color output helpers
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }

# Configuration
TAXONOMY_FILE="${TAXONOMY_FILE:-scripts/ignore-taxonomy.json}"
MODE="${MODE:-full}"  # full | diff | suggest | enforce
DRY_RUN="${DRY_RUN:-true}"
FAIL_ON_BARE="${FAIL_ON_BARE:-false}"
MAX_BARE_PERCENT="${MAX_BARE_PERCENT:-5}"

# Output formatting
print_header() {
    echo ""
    blue "═══════════════════════════════════════════════════════"
    blue "$1"
    blue "═══════════════════════════════════════════════════════"
    echo ""
}

# Count functions
count_total_ignores() {
    rg '#\[ignore' --type rust crates/ tests/ xtask/ -c 2>/dev/null | \
        awk -F: '{sum += $2} END {print sum+0}'
}

count_bare_ignores() {
    rg '#\[ignore\]' --type rust crates/ tests/ xtask/ -c 2>/dev/null | \
        awk -F: '{sum += $2} END {print sum+0}'
}

count_annotated_ignores() {
    rg '#\[ignore\s*=\s*"' --type rust crates/ tests/ xtask/ -c 2>/dev/null | \
        awk -F: '{sum += $2} END {print sum+0}'
}

# Detection functions
find_bare_ignores() {
    # Find all bare #[ignore] without reasons
    rg '#\[ignore\]' --type rust crates/ tests/ xtask/ -n --color never 2>/dev/null || true
}

extract_context() {
    local file="$1"
    local line="$2"
    local context_lines=10

    # Extract surrounding context for categorization
    sed -n "$((line - context_lines)),$((line + 5))p" "$file" 2>/dev/null || echo ""
}

# Categorization engine
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

# Suggestion generator
suggest_annotation() {
    local file="$1"
    local line="$2"
    local categories="$3"
    local confidence="$4"
    local suggested_reason="$5"

    yellow "  📍 ${file}:${line}"
    yellow "     Categories: ${categories}"
    yellow "     Confidence: ${confidence}%"
    yellow "     Suggested: #[ignore = \"${suggested_reason}...\"]"
    echo ""
}

# Main workflow
main() {
    print_header "BitNet-rs #[ignore] Hygiene Check"

    # Gather statistics
    local total=$(count_total_ignores)
    local bare=$(count_bare_ignores)
    local annotated=$(count_annotated_ignores)
    local bare_percent=0

    if [ "$total" -gt 0 ]; then
        bare_percent=$((bare * 100 / total))
    fi

    echo "Total #[ignore] annotations: ${total}"
    echo "Annotated (with reason):     ${annotated} ($((annotated * 100 / total))%)"
    echo "Bare (no reason):            ${bare} (${bare_percent}%)"
    echo ""

    # Check threshold
    if [ "$bare_percent" -gt "$MAX_BARE_PERCENT" ]; then
        red "❌ Bare ignore percentage (${bare_percent}%) exceeds threshold (${MAX_BARE_PERCENT}%)"
        if [ "$FAIL_ON_BARE" = "true" ]; then
            echo ""
            red "FAILED: Fix bare #[ignore] annotations before merging"
            exit 1
        fi
    else
        green "✅ Bare ignore percentage (${bare_percent}%) within threshold (${MAX_BARE_PERCENT}%)"
    fi

    # Mode-specific execution
    case "$MODE" in
        full)
            print_header "Full Scan: All Bare Ignores"
            scan_and_categorize
            ;;
        diff)
            print_header "Diff Mode: PR Changes Only"
            scan_diff
            ;;
        suggest)
            print_header "Suggestion Mode: Auto-Annotation Preview"
            generate_suggestions
            ;;
        enforce)
            print_header "Enforce Mode: CI Guard"
            enforce_hygiene
            ;;
        *)
            red "Unknown mode: $MODE"
            exit 1
            ;;
    esac
}

scan_and_categorize() {
    local count=0

    while IFS=: read -r file line _; do
        count=$((count + 1))
        local context=$(extract_context "$file" "$line")
        local result=$(categorize_ignore "$file" "$line" "$context")

        # Parse result
        IFS='|' read -r location categories confidence suggested_reason <<< "$result"

        suggest_annotation "$file" "$line" "$categories" "$confidence" "$suggested_reason"
    done < <(find_bare_ignores)

    echo ""
    if [ "$count" -gt 0 ]; then
        yellow "Found ${count} bare #[ignore] annotations requiring attention"
    else
        green "✅ All #[ignore] annotations have explicit reasons!"
    fi
}

scan_diff() {
    # Check only lines changed in current git diff
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        red "Not in a git repository - diff mode unavailable"
        exit 1
    fi

    local new_bare_ignores=0

    # Get diff for Rust files
    while IFS=: read -r file line content; do
        # Check if line contains bare #[ignore]
        if echo "$content" | grep -qE '^\+.*#\[ignore\]$'; then
            new_bare_ignores=$((new_bare_ignores + 1))

            red "❌ New bare #[ignore] found:"
            red "   ${file}:${line}"
            echo ""

            # Suggest annotation
            local context=$(extract_context "$file" "$line")
            local result=$(categorize_ignore "$file" "$line" "$context")
            IFS='|' read -r location categories confidence suggested_reason <<< "$result"

            yellow "   Quick-fix suggestion:"
            yellow "   #[ignore = \"${suggested_reason}...\"]"
            echo ""
        fi
    done < <(git diff HEAD --unified=0 --diff-filter=AM -- '*.rs' | grep -E '^\+.*#\[ignore\]' || true)

    if [ "$new_bare_ignores" -gt 0 ]; then
        red "❌ Found ${new_bare_ignores} new bare #[ignore] annotation(s) in this PR"
        echo ""
        red "Please add explicit reasons using the taxonomy:"
        echo "  - Issue #NNN: <description>     (for blocked tests)"
        echo "  - slow: <reason>                 (for performance tests)"
        echo "  - requires: <resource>           (for external dependencies)"
        echo "  - gpu: <requirement>             (for GPU-specific tests)"
        echo "  - network: <dependency>          (for network tests)"
        echo "  - TODO: <task>                   (for placeholders)"
        echo ""
        exit 1
    else
        green "✅ No new bare #[ignore] annotations in this PR"
    fi
}

generate_suggestions() {
    # Generate suggested annotations in batch
    local output_file="ignore-suggestions.txt"

    echo "# Auto-Generated #[ignore] Annotation Suggestions" > "$output_file"
    echo "# Generated: $(date)" >> "$output_file"
    echo "" >> "$output_file"

    while IFS=: read -r file line _; do
        local context=$(extract_context "$file" "$line")
        local result=$(categorize_ignore "$file" "$line" "$context")

        IFS='|' read -r location categories confidence suggested_reason <<< "$result"

        echo "# ${file}:${line}" >> "$output_file"
        echo "# Categories: ${categories} (confidence: ${confidence}%)" >> "$output_file"
        echo "#[ignore = \"${suggested_reason}...\"]" >> "$output_file"
        echo "" >> "$output_file"
    done < <(find_bare_ignores)

    green "✅ Suggestions written to ${output_file}"
}

enforce_hygiene() {
    # CI mode: strict enforcement
    export FAIL_ON_BARE=true
    export MODE=diff

    scan_diff
}

# Execute main workflow
main "$@"
