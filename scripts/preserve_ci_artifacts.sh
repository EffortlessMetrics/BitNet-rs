#!/usr/bin/env bash
# Preserve CI artifacts for debugging and audit trail
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Configuration
ARTIFACTS_DIR="${ARTIFACTS_DIR:-ci-artifacts}"
PRESERVE_DAYS="${PRESERVE_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_ID="${GITHUB_RUN_ID:-local-$TIMESTAMP}"

# Create archive directory structure
ARCHIVE_DIR="$ARTIFACTS_DIR/$RUN_ID"
mkdir -p "$ARCHIVE_DIR"/{logs,perf,validation,models}

log_info() {
    echo "[PRESERVE] $*"
}

# Preserve validation artifacts
preserve_validation() {
    log_info "Preserving validation artifacts..."
    
    # Copy parity results
    if [ -f "/tmp/parity_results.json" ]; then
        cp /tmp/parity_results.json "$ARCHIVE_DIR/validation/"
        log_info "  • Parity results preserved"
    fi
    
    # Copy tokenizer test results
    if [ -f "artifacts/parity_failures.jsonl" ]; then
        cp artifacts/parity_failures.jsonl "$ARCHIVE_DIR/validation/"
        log_info "  • Parity failures preserved"
    fi
    
    # Copy validation logs
    for log in validation*.log test-results/*/validation.log; do
        if [ -f "$log" ]; then
            cp "$log" "$ARCHIVE_DIR/logs/"
            log_info "  • $(basename $log) preserved"
        fi
    done
}

# Preserve performance data
preserve_performance() {
    log_info "Preserving performance data..."
    
    # Copy all performance JSONs
    if [ -d "bench/results" ]; then
        cp bench/results/*.json "$ARCHIVE_DIR/perf/" 2>/dev/null || true
        local count=$(ls -1 "$ARCHIVE_DIR/perf/"*.json 2>/dev/null | wc -l)
        log_info "  • $count performance JSONs preserved"
    fi
    
    # Copy rendered markdown
    for md in docs/PERF_*.md; do
        if [ -f "$md" ]; then
            cp "$md" "$ARCHIVE_DIR/perf/"
            log_info "  • $(basename $md) preserved"
        fi
    done
}

# Preserve model info
preserve_model_info() {
    log_info "Preserving model information..."
    
    # Copy model info JSONs
    for info in test-results/*/info-*.json; do
        if [ -f "$info" ]; then
            cp "$info" "$ARCHIVE_DIR/models/"
            log_info "  • $(basename "$info") preserved"
        fi
    done
    
    # Save model checksums
    if [ -d "models" ]; then
        find models -type f \( -name "*.gguf" -o -name "*.safetensors" \) -exec sha256sum {} \; \
            > "$ARCHIVE_DIR/models/checksums.txt" 2>/dev/null || true
        log_info "  • Model checksums preserved"
    fi
}

# Create metadata file
create_metadata() {
    log_info "Creating metadata..."
    
    cat > "$ARCHIVE_DIR/metadata.json" <<EOF
{
    "run_id": "$RUN_ID",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "platform": "$(get_platform_name)",
    "wsl2": $(detect_wsl2 && echo "true" || echo "false"),
    "github_event": "${GITHUB_EVENT_NAME:-local}",
    "github_ref": "${GITHUB_REF:-local}",
    "github_sha": "${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}",
    "github_actor": "${GITHUB_ACTOR:-$USER}",
    "preserve_days": $PRESERVE_DAYS,
    "artifacts": {
        "validation": $(ls -1 "$ARCHIVE_DIR/validation/"* 2>/dev/null | wc -l),
        "performance": $(ls -1 "$ARCHIVE_DIR/perf/"* 2>/dev/null | wc -l),
        "logs": $(ls -1 "$ARCHIVE_DIR/logs/"* 2>/dev/null | wc -l),
        "models": $(ls -1 "$ARCHIVE_DIR/models/"* 2>/dev/null | wc -l)
    }
}
EOF
    
    log_info "  • Metadata created"
}

# Create compressed archive
create_archive() {
    log_info "Creating compressed archive..."
    
    local archive_name="bitnet-artifacts-$RUN_ID.tar.gz"
    tar -czf "$ARTIFACTS_DIR/$archive_name" -C "$ARTIFACTS_DIR" "$RUN_ID"
    
    local size=$(du -h "$ARTIFACTS_DIR/$archive_name" | cut -f1)
    log_info "  • Archive created: $archive_name ($size)"
    
    # Clean up uncompressed directory
    rm -rf "$ARCHIVE_DIR"
}

# Clean old artifacts
clean_old_artifacts() {
    log_info "Cleaning old artifacts (> $PRESERVE_DAYS days)..."
    
    find "$ARTIFACTS_DIR" -name "bitnet-artifacts-*.tar.gz" -mtime +$PRESERVE_DAYS -delete 2>/dev/null || true
    
    local remaining=$(ls -1 "$ARTIFACTS_DIR"/bitnet-artifacts-*.tar.gz 2>/dev/null | wc -l)
    log_info "  • $remaining archives remaining"
}

# Generate summary for GitHub Actions
generate_github_summary() {
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        cat >> "$GITHUB_STEP_SUMMARY" <<EOF
## Artifacts Preserved

- **Run ID**: $RUN_ID
- **Platform**: $(get_platform_name)
- **Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)

### Preserved Files
- Validation: $(ls -1 "$ARCHIVE_DIR/validation/"* 2>/dev/null | wc -l) files
- Performance: $(ls -1 "$ARCHIVE_DIR/perf/"* 2>/dev/null | wc -l) files
- Logs: $(ls -1 "$ARCHIVE_DIR/logs/"* 2>/dev/null | wc -l) files
- Models: $(ls -1 "$ARCHIVE_DIR/models/"* 2>/dev/null | wc -l) files

Archive available for $PRESERVE_DAYS days.
EOF
    fi
}

# Main execution
main() {
    echo "====================================================="
    echo "    CI Artifact Preservation"
    echo "====================================================="
    echo ""
    log_info "Run ID: $RUN_ID"
    log_info "Archive directory: $ARTIFACTS_DIR"
    echo ""
    
    # Preserve all artifacts
    preserve_validation
    preserve_performance
    preserve_model_info
    create_metadata
    
    # Create archive
    create_archive
    
    # Clean old artifacts
    clean_old_artifacts
    
    # Generate GitHub summary if in CI
    generate_github_summary
    
    echo ""
    log_info "✅ Artifacts preserved successfully"
    log_info "Archive: $ARTIFACTS_DIR/bitnet-artifacts-$RUN_ID.tar.gz"
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi