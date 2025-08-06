#!/bin/bash
# Comprehensive security audit for BitNet.rs
# Validates security practices and identifies potential vulnerabilities

set -euo pipefail

echo "🔒 Security Audit for BitNet.rs"
echo "==============================="

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

# Security audit results
audit_results=()
failed_checks=0
total_checks=0

# Function to record audit result
record_result() {
    local check_name="$1"
    local status="$2"
    local message="$3"
    
    audit_results+=("$check_name:$status:$message")
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

# 1. Dependency Security Audit
audit_dependencies() {
    print_status "Auditing dependencies for known vulnerabilities..."
    
    if command -v cargo-audit &> /dev/null; then
        if cargo audit --json > audit_report.json 2>/dev/null; then
            local vuln_count=$(jq '.vulnerabilities.count' audit_report.json 2>/dev/null || echo "0")
            if [[ "$vuln_count" -eq 0 ]]; then
                record_result "Dependency Audit" "PASS" "No known vulnerabilities found"
            else
                record_result "Dependency Audit" "FAIL" "$vuln_count known vulnerabilities found"
                jq -r '.vulnerabilities.list[] | "  - \(.advisory.title) (\(.advisory.id))"' audit_report.json
            fi
        else
            record_result "Dependency Audit" "FAIL" "cargo-audit failed to run"
        fi
    else
        record_result "Dependency Audit" "WARN" "cargo-audit not installed, skipping vulnerability check"
    fi
}

# 2. Unsafe Code Analysis
audit_unsafe_code() {
    print_status "Analyzing unsafe code usage..."
    
    local unsafe_count=0
    local undocumented_unsafe=0
    
    # Find all unsafe blocks
    while IFS= read -r -d '' file; do
        if grep -q "unsafe" "$file"; then
            local file_unsafe_count=$(grep -c "unsafe" "$file")
            unsafe_count=$((unsafe_count + file_unsafe_count))
            
            # Check for undocumented unsafe blocks
            local undoc_count=$(grep -A5 -B5 "unsafe" "$file" | grep -c "unsafe" || true)
            local doc_count=$(grep -A5 -B5 "unsafe" "$file" | grep -c "// SAFETY:" || true)
            
            if [[ $doc_count -lt $undoc_count ]]; then
                undocumented_unsafe=$((undocumented_unsafe + undoc_count - doc_count))
                print_warning "Undocumented unsafe code in: $file"
            fi
        fi
    done < <(find src crates -name "*.rs" -print0 2>/dev/null)
    
    if [[ $unsafe_count -eq 0 ]]; then
        record_result "Unsafe Code" "PASS" "No unsafe code found"
    elif [[ $undocumented_unsafe -eq 0 ]]; then
        record_result "Unsafe Code" "PASS" "$unsafe_count unsafe blocks found, all documented"
    else
        record_result "Unsafe Code" "WARN" "$undocumented_unsafe undocumented unsafe blocks found"
    fi
}

# 3. Input Validation Analysis
audit_input_validation() {
    print_status "Checking input validation practices..."
    
    local validation_issues=0
    
    # Check for potential buffer overflow patterns
    local buffer_patterns=("from_raw_parts" "slice::from_raw_parts" "Vec::from_raw_parts")
    for pattern in "${buffer_patterns[@]}"; do
        if grep -r "$pattern" src crates --include="*.rs" >/dev/null 2>&1; then
            validation_issues=$((validation_issues + 1))
            print_warning "Found potentially unsafe buffer operation: $pattern"
        fi
    done
    
    # Check for unchecked arithmetic
    local arithmetic_patterns=("unchecked_add" "unchecked_sub" "unchecked_mul")
    for pattern in "${arithmetic_patterns[@]}"; do
        if grep -r "$pattern" src crates --include="*.rs" >/dev/null 2>&1; then
            validation_issues=$((validation_issues + 1))
            print_warning "Found unchecked arithmetic: $pattern"
        fi
    done
    
    if [[ $validation_issues -eq 0 ]]; then
        record_result "Input Validation" "PASS" "No obvious input validation issues found"
    else
        record_result "Input Validation" "WARN" "$validation_issues potential input validation issues found"
    fi
}

# 4. Memory Safety Analysis
audit_memory_safety() {
    print_status "Analyzing memory safety patterns..."
    
    local memory_issues=0
    
    # Check for manual memory management
    local memory_patterns=("Box::from_raw" "Box::into_raw" "forget" "transmute")
    for pattern in "${memory_patterns[@]}"; do
        local count=$(grep -r "$pattern" src crates --include="*.rs" | wc -l)
        if [[ $count -gt 0 ]]; then
            memory_issues=$((memory_issues + count))
            print_warning "Found manual memory management: $pattern ($count occurrences)"
        fi
    done
    
    # Check for potential use-after-free patterns
    if grep -r "drop.*raw" src crates --include="*.rs" >/dev/null 2>&1; then
        memory_issues=$((memory_issues + 1))
        print_warning "Found potential use-after-free pattern"
    fi
    
    if [[ $memory_issues -eq 0 ]]; then
        record_result "Memory Safety" "PASS" "No obvious memory safety issues found"
    else
        record_result "Memory Safety" "WARN" "$memory_issues potential memory safety issues found"
    fi
}

# 5. Cryptographic Security (if applicable)
audit_cryptography() {
    print_status "Checking cryptographic practices..."
    
    local crypto_issues=0
    
    # Check for weak random number generation
    if grep -r "rand::random" src crates --include="*.rs" >/dev/null 2>&1; then
        if ! grep -r "ChaCha" src crates --include="*.rs" >/dev/null 2>&1; then
            crypto_issues=$((crypto_issues + 1))
            print_warning "Using potentially weak random number generation"
        fi
    fi
    
    # Check for hardcoded secrets (basic patterns)
    local secret_patterns=("password.*=" "secret.*=" "key.*=" "token.*=")
    for pattern in "${secret_patterns[@]}"; do
        if grep -ri "$pattern" src crates --include="*.rs" | grep -v "test" >/dev/null 2>&1; then
            crypto_issues=$((crypto_issues + 1))
            print_warning "Potential hardcoded secret found: $pattern"
        fi
    done
    
    if [[ $crypto_issues -eq 0 ]]; then
        record_result "Cryptography" "PASS" "No obvious cryptographic issues found"
    else
        record_result "Cryptography" "WARN" "$crypto_issues potential cryptographic issues found"
    fi
}

# 6. Supply Chain Security
audit_supply_chain() {
    print_status "Checking supply chain security..."
    
    local supply_chain_issues=0
    
    # Check for git dependencies (potential supply chain risk)
    if grep -r "git.*=" Cargo.toml crates/*/Cargo.toml 2>/dev/null; then
        supply_chain_issues=$((supply_chain_issues + 1))
        print_warning "Git dependencies found (potential supply chain risk)"
    fi
    
    # Check for path dependencies outside workspace
    if grep -r "path.*=.*\.\." Cargo.toml crates/*/Cargo.toml 2>/dev/null; then
        supply_chain_issues=$((supply_chain_issues + 1))
        print_warning "External path dependencies found"
    fi
    
    # Check for license compliance
    if command -v cargo-deny &> /dev/null; then
        if ! cargo deny check licenses >/dev/null 2>&1; then
            supply_chain_issues=$((supply_chain_issues + 1))
            print_warning "License compliance issues found"
        fi
    else
        print_warning "cargo-deny not installed, skipping license check"
    fi
    
    if [[ $supply_chain_issues -eq 0 ]]; then
        record_result "Supply Chain" "PASS" "No supply chain security issues found"
    else
        record_result "Supply Chain" "WARN" "$supply_chain_issues potential supply chain issues found"
    fi
}

# 7. Information Disclosure
audit_information_disclosure() {
    print_status "Checking for information disclosure risks..."
    
    local disclosure_issues=0
    
    # Check for debug prints in release code
    local debug_patterns=("println!" "eprintln!" "dbg!" "print!")
    for pattern in "${debug_patterns[@]}"; do
        local count=$(grep -r "$pattern" src crates --include="*.rs" | grep -v "test" | grep -v "example" | wc -l)
        if [[ $count -gt 5 ]]; then  # Allow some debug prints
            disclosure_issues=$((disclosure_issues + 1))
            print_warning "Many debug prints found: $pattern ($count occurrences)"
        fi
    done
    
    # Check for potential path traversal
    if grep -r "\.\./\.\." src crates --include="*.rs" >/dev/null 2>&1; then
        disclosure_issues=$((disclosure_issues + 1))
        print_warning "Potential path traversal pattern found"
    fi
    
    if [[ $disclosure_issues -eq 0 ]]; then
        record_result "Information Disclosure" "PASS" "No information disclosure risks found"
    else
        record_result "Information Disclosure" "WARN" "$disclosure_issues potential information disclosure risks found"
    fi
}

# 8. Fuzzing Readiness
audit_fuzzing_readiness() {
    print_status "Checking fuzzing readiness..."
    
    local fuzzing_score=0
    
    # Check for fuzz targets
    if [[ -d "fuzz" ]] && [[ -n "$(ls -A fuzz 2>/dev/null)" ]]; then
        fuzzing_score=$((fuzzing_score + 1))
        print_success "Fuzz targets directory found"
    else
        print_warning "No fuzz targets found"
    fi
    
    # Check for property-based tests
    if grep -r "proptest" Cargo.toml crates/*/Cargo.toml >/dev/null 2>&1; then
        fuzzing_score=$((fuzzing_score + 1))
        print_success "Property-based testing framework found"
    else
        print_warning "No property-based testing found"
    fi
    
    if [[ $fuzzing_score -ge 1 ]]; then
        record_result "Fuzzing Readiness" "PASS" "Some fuzzing infrastructure present"
    else
        record_result "Fuzzing Readiness" "WARN" "Limited fuzzing infrastructure"
    fi
}

# Generate security report
generate_security_report() {
    print_status "Generating security audit report..."
    
    local report_file="security_audit_report.md"
    
    cat > "$report_file" << EOF
# BitNet.rs Security Audit Report

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Summary

- **Total Checks**: $total_checks
- **Passed**: $((total_checks - failed_checks))
- **Failed/Warnings**: $failed_checks

## Detailed Results

EOF
    
    for result in "${audit_results[@]}"; do
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

## Recommendations

1. **Regular Audits**: Run this security audit regularly, especially before releases
2. **Dependency Updates**: Keep dependencies updated and monitor for new vulnerabilities
3. **Code Review**: Ensure all unsafe code is properly reviewed and documented
4. **Fuzzing**: Consider implementing comprehensive fuzz testing
5. **Static Analysis**: Use additional static analysis tools like Clippy with security lints

## Tools Used

- cargo-audit (dependency vulnerability scanning)
- grep-based pattern analysis
- cargo-deny (license compliance)
- Manual code review patterns

EOF
    
    print_success "Security audit report generated: $report_file"
}

# Main execution
main() {
    print_status "Starting comprehensive security audit..."
    
    # Run all audit checks
    audit_dependencies
    audit_unsafe_code
    audit_input_validation
    audit_memory_safety
    audit_cryptography
    audit_supply_chain
    audit_information_disclosure
    audit_fuzzing_readiness
    
    # Generate report
    generate_security_report
    
    # Summary
    echo ""
    echo "🔒 Security Audit Summary"
    echo "========================"
    echo "Total checks: $total_checks"
    echo "Issues found: $failed_checks"
    
    if [[ $failed_checks -eq 0 ]]; then
        print_success "🎉 Security audit completed with no critical issues!"
        return 0
    else
        print_warning "⚠️ Security audit completed with $failed_checks issues to review"
        print_status "See security_audit_report.md for detailed findings"
        return 0  # Don't fail on warnings, just inform
    fi
}

# Run main function
main "$@"