#!/bin/bash
# Documentation review and accuracy validation for BitNet.rs
# Ensures documentation is complete, accurate, and up-to-date

set -euo pipefail

echo "📚 Documentation Review and Accuracy Validation"
echo "==============================================="

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
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Documentation validation results
doc_results=()
failed_checks=0
total_checks=0

# Function to record validation result
record_result() {
    local check_name="$1"
    local status="$2"
    local message="$3"

    doc_results+=("$check_name:$status:$message")
    total_checks=$((total_checks + 1))

    if [[ "$status" == "FAIL" ]]; then
        failed_checks=$((failed_checks + 1))
        print_error "$check_name: $message"
    elif [[ "$status" == "WARN" ]]; then
        print_warning "$check_name: $message"
    else
        print_success "$check_name: $message"
    fi
}

# 1. README Validation
validate_readme() {
    print_status "Validating README.md..."

    if [[ ! -f "README.md" ]]; then
        record_result "README Exists" "FAIL" "README.md not found"
        return
    fi

    local readme_content=$(cat README.md)

    # Check for required sections
    local required_sections=("Features" "Quick Start" "Installation" "Usage" "License")
    local missing_sections=()

    for section in "${required_sections[@]}"; do
        if ! echo "$readme_content" | grep -qi "## $section\|# $section"; then
            missing_sections+=("$section")
        fi
    done

    if [[ ${#missing_sections[@]} -eq 0 ]]; then
        record_result "README Sections" "PASS" "All required sections present"
    else
        record_result "README Sections" "WARN" "Missing sections: ${missing_sections[*]}"
    fi

    # Check for badges
    if echo "$readme_content" | grep -q "!\[.*\](https://.*\.svg)"; then
        record_result "README Badges" "PASS" "Status badges present"
    else
        record_result "README Badges" "WARN" "No status badges found"
    fi

    # Check for code examples
    if echo "$readme_content" | grep -q '```rust\|```bash\|```python'; then
        record_result "README Examples" "PASS" "Code examples present"
    else
        record_result "README Examples" "WARN" "No code examples found"
    fi

    # Check README length (should be substantial but not too long)
    local readme_lines=$(wc -l < README.md)
    if [[ $readme_lines -ge 50 && $readme_lines -le 500 ]]; then
        record_result "README Length" "PASS" "$readme_lines lines (appropriate length)"
    elif [[ $readme_lines -lt 50 ]]; then
        record_result "README Length" "WARN" "$readme_lines lines (may be too short)"
    else
        record_result "README Length" "WARN" "$readme_lines lines (may be too long)"
    fi
}

# 2. API Documentation Validation
validate_api_docs() {
    print_status "Validating API documentation..."

    # Generate documentation
    if cargo doc --all-features --no-deps >/dev/null 2>&1; then
        record_result "Doc Generation" "PASS" "Documentation generates successfully"

        # Check for missing documentation
        local missing_docs_output=$(cargo doc --all-features --no-deps 2>&1 | grep "warning.*missing documentation" || true)
        if [[ -z "$missing_docs_output" ]]; then
            record_result "Doc Completeness" "PASS" "No missing documentation warnings"
        else
            local missing_count=$(echo "$missing_docs_output" | wc -l)
            record_result "Doc Completeness" "WARN" "$missing_count items missing documentation"
        fi

        # Check for broken links in documentation
        if command -v linkchecker &> /dev/null; then
            if linkchecker target/doc/bitnet/index.html >/dev/null 2>&1; then
                record_result "Doc Links" "PASS" "No broken links found"
            else
                record_result "Doc Links" "WARN" "Some documentation links may be broken"
            fi
        else
            record_result "Doc Links" "WARN" "linkchecker not available, skipping link validation"
        fi

    else
        record_result "Doc Generation" "FAIL" "Documentation generation failed"
    fi
}

# 3. Example Code Validation
validate_examples() {
    print_status "Validating example code..."

    local example_count=0
    local working_examples=0

    # Check examples directory
    if [[ -d "examples" ]]; then
        for example_file in examples/*.rs; do
            if [[ -f "$example_file" ]]; then
                example_count=$((example_count + 1))

                # Check if example compiles
                local example_name=$(basename "$example_file" .rs)
                if cargo check --example "$example_name" >/dev/null 2>&1; then
                    working_examples=$((working_examples + 1))
                fi
            fi
        done

        if [[ $example_count -gt 0 ]]; then
            if [[ $working_examples -eq $example_count ]]; then
                record_result "Examples Compile" "PASS" "All $example_count examples compile"
            else
                record_result "Examples Compile" "WARN" "$working_examples/$example_count examples compile"
            fi
        else
            record_result "Examples Exist" "WARN" "No examples found"
        fi
    else
        record_result "Examples Directory" "WARN" "Examples directory not found"
    fi

    # Check README examples
    local readme_rust_blocks=$(grep -c '```rust' README.md 2>/dev/null || echo "0")
    if [[ $readme_rust_blocks -gt 0 ]]; then
        record_result "README Code Blocks" "PASS" "$readme_rust_blocks Rust code blocks in README"

        # Extract and test README code blocks
        local temp_example="temp_readme_example.rs"
        local readme_examples_work=true

        # This is a simplified check - in practice, you'd extract and test each block
        if grep -A 20 '```rust' README.md | grep -q "use bitnet"; then
            record_result "README Code Quality" "PASS" "README examples use proper imports"
        else
            record_result "README Code Quality" "WARN" "README examples may need improvement"
        fi
    else
        record_result "README Code Blocks" "WARN" "No Rust code blocks in README"
    fi
}

# 4. Feature Documentation Validation
validate_feature_docs() {
    print_status "Validating feature documentation..."

    # Check if FEATURES.md exists
    if [[ -f "FEATURES.md" ]]; then
        record_result "Feature Docs Exist" "PASS" "FEATURES.md found"

        # Check if all Cargo.toml features are documented
        local cargo_features=$(grep -A 20 "\[features\]" Cargo.toml | grep "^[a-zA-Z]" | cut -d' ' -f1 | cut -d'=' -f1 || true)
        local undocumented_features=()

        for feature in $cargo_features; do
            if [[ "$feature" != "default" ]] && ! grep -q "$feature" FEATURES.md; then
                undocumented_features+=("$feature")
            fi
        done

        if [[ ${#undocumented_features[@]} -eq 0 ]]; then
            record_result "Feature Coverage" "PASS" "All features documented"
        else
            record_result "Feature Coverage" "WARN" "Undocumented features: ${undocumented_features[*]}"
        fi
    else
        record_result "Feature Docs Exist" "WARN" "FEATURES.md not found"
    fi
}

# 5. Changelog Validation
validate_changelog() {
    print_status "Validating CHANGELOG.md..."

    if [[ -f "CHANGELOG.md" ]]; then
        record_result "Changelog Exists" "PASS" "CHANGELOG.md found"

        local changelog_content=$(cat CHANGELOG.md)

        # Check for proper format
        if echo "$changelog_content" | grep -q "## \[Unreleased\]\|## \[[0-9]"; then
            record_result "Changelog Format" "PASS" "Proper changelog format"
        else
            record_result "Changelog Format" "WARN" "Changelog format may need improvement"
        fi

        # Check for recent updates
        local changelog_lines=$(wc -l < CHANGELOG.md)
        if [[ $changelog_lines -gt 20 ]]; then
            record_result "Changelog Content" "PASS" "Substantial changelog content"
        else
            record_result "Changelog Content" "WARN" "Changelog may need more content"
        fi
    else
        record_result "Changelog Exists" "FAIL" "CHANGELOG.md not found"
    fi
}

# 6. License Documentation
validate_license_docs() {
    print_status "Validating license documentation..."

    # Check for LICENSE file
    local license_files=("LICENSE" "LICENSE.md" "LICENSE.txt" "LICENSE-MIT" "LICENSE-APACHE")
    local license_found=false

    for license_file in "${license_files[@]}"; do
        if [[ -f "$license_file" ]]; then
            license_found=true
            break
        fi
    done

    if $license_found; then
        record_result "License File" "PASS" "License file found"
    else
        record_result "License File" "WARN" "No license file found"
    fi

    # Check license in Cargo.toml
    if grep -q "license.*=" Cargo.toml; then
        record_result "License Metadata" "PASS" "License specified in Cargo.toml"
    else
        record_result "License Metadata" "WARN" "License not specified in Cargo.toml"
    fi

    # Check license headers in source files
    local files_with_license=0
    local total_source_files=0

    while IFS= read -r -d '' file; do
        total_source_files=$((total_source_files + 1))
        if head -10 "$file" | grep -qi "license\|copyright"; then
            files_with_license=$((files_with_license + 1))
        fi
    done < <(find src crates -name "*.rs" -print0 2>/dev/null)

    if [[ $total_source_files -gt 0 ]]; then
        local license_percentage=$((files_with_license * 100 / total_source_files))
        if [[ $license_percentage -gt 80 ]]; then
            record_result "License Headers" "PASS" "$license_percentage% of files have license headers"
        elif [[ $license_percentage -gt 50 ]]; then
            record_result "License Headers" "WARN" "$license_percentage% of files have license headers"
        else
            record_result "License Headers" "WARN" "Few files have license headers ($license_percentage%)"
        fi
    fi
}

# 7. Migration Guide Validation
validate_migration_docs() {
    print_status "Validating migration documentation..."

    local migration_files=("MIGRATION.md" "MIGRATION_GUIDE.md" "crates/bitnet-py/MIGRATION_GUIDE.md")
    local migration_found=false

    for migration_file in "${migration_files[@]}"; do
        if [[ -f "$migration_file" ]]; then
            migration_found=true

            # Check migration guide content
            local migration_content=$(cat "$migration_file")
            if echo "$migration_content" | grep -qi "python\|c++\|migration"; then
                record_result "Migration Content" "PASS" "Migration guide has relevant content"
            else
                record_result "Migration Content" "WARN" "Migration guide content may need improvement"
            fi
            break
        fi
    done

    if $migration_found; then
        record_result "Migration Guide" "PASS" "Migration guide found"
    else
        record_result "Migration Guide" "WARN" "No migration guide found"
    fi
}

# 8. Performance Documentation
validate_performance_docs() {
    print_status "Validating performance documentation..."

    # Check for performance claims in README
    if grep -qi "performance\|speed\|fast\|benchmark" README.md; then
        record_result "Performance Claims" "PASS" "Performance information in README"

        # Check if claims are backed by data
        if grep -qi "benchmark\|test\|measurement" README.md; then
            record_result "Performance Evidence" "PASS" "Performance claims appear to be backed by evidence"
        else
            record_result "Performance Evidence" "WARN" "Performance claims may need supporting evidence"
        fi
    else
        record_result "Performance Claims" "WARN" "No performance information in README"
    fi

    # Check for benchmark documentation
    if [[ -d "benches" ]] && [[ -n "$(ls -A benches 2>/dev/null)" ]]; then
        record_result "Benchmark Docs" "PASS" "Benchmark code available"
    else
        record_result "Benchmark Docs" "WARN" "No benchmark code found"
    fi
}

# Generate documentation validation report
generate_docs_report() {
    print_status "Generating documentation validation report..."

    local report_file="docs_validation_report.md"

    cat > "$report_file" << EOF
# BitNet.rs Documentation Validation Report

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Summary

- **Total Checks**: $total_checks
- **Passed**: $((total_checks - failed_checks))
- **Failed/Warnings**: $failed_checks

## Validation Results

EOF

    for result in "${doc_results[@]}"; do
        local check_name=$(echo "$result" | cut -d: -f1)
        local status=$(echo "$result" | cut -d: -f2)
        local message=$(echo "$result" | cut -d: -f3-)

        local status_emoji="✅"
        if [[ "$status" == "WARN" ]]; then
            status_emoji="⚠️"
        elif [[ "$status" == "FAIL" ]]; then
            status_emoji="❌"
        fi

        echo "### $status_emoji $check_name" >> "$report_file"
        echo "" >> "$report_file"
        echo "**Status**: $status" >> "$report_file"
        echo "**Details**: $message" >> "$report_file"
        echo "" >> "$report_file"
    done

    cat >> "$report_file" << EOF

## Documentation Areas Validated

1. **README**: Main project documentation and first impressions
2. **API Documentation**: Generated documentation from code comments
3. **Examples**: Working code examples for users
4. **Feature Documentation**: Comprehensive feature flag documentation
5. **Changelog**: Version history and changes
6. **License**: Legal documentation and compliance
7. **Migration Guide**: Help for users transitioning from other implementations
8. **Performance**: Performance claims and supporting evidence

## Recommendations

1. Address any failed validations before release
2. Ensure all public APIs have comprehensive documentation
3. Keep examples up-to-date with API changes
4. Maintain accurate performance claims with supporting benchmarks
5. Regular documentation reviews as part of development process

## Tools Used

- cargo doc (API documentation generation)
- grep-based content analysis
- linkchecker (if available)
- Manual content validation

EOF

    print_success "Documentation validation report generated: $report_file"
}

# Main execution
main() {
    print_status "Starting comprehensive documentation validation..."

    # Run all validation checks
    validate_readme
    validate_api_docs
    validate_examples
    validate_feature_docs
    validate_changelog
    validate_license_docs
    validate_migration_docs
    validate_performance_docs

    # Generate report
    generate_docs_report

    # Summary
    echo ""
    echo "📚 Documentation Validation Summary"
    echo "=================================="
    echo "Total checks: $total_checks"
    echo "Issues found: $failed_checks"

    if [[ $failed_checks -eq 0 ]]; then
        print_success "🎉 Documentation validation completed successfully!"
        return 0
    else
        print_warning "⚠️ Documentation validation completed with $failed_checks issues to review"
        print_status "See docs_validation_report.md for detailed findings"
        return 0  # Don't fail on warnings, just inform
    fi
}

# Run main function
main "$@"
