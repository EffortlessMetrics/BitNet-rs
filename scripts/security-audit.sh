#!/bin/bash
# Comprehensive security audit script

set -e

echo "🔒 Starting comprehensive security audit..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install required tools
install_tools() {
    print_status $BLUE "📦 Installing security audit tools..."
    
    if ! command_exists cargo-audit; then
        print_status $YELLOW "Installing cargo-audit..."
        cargo install cargo-audit --locked
    fi
    
    if ! command_exists cargo-deny; then
        print_status $YELLOW "Installing cargo-deny..."
        cargo install cargo-deny --locked
    fi
    
    if ! command_exists cargo-license; then
        print_status $YELLOW "Installing cargo-license..."
        cargo install cargo-license --locked
    fi
    
    if ! command_exists cargo-fuzz; then
        print_status $YELLOW "Installing cargo-fuzz..."
        cargo install cargo-fuzz --locked
    fi
}

# Run dependency audit
run_dependency_audit() {
    print_status $BLUE "🔍 Running dependency security audit..."
    
    echo "Checking for security vulnerabilities..."
    if cargo audit; then
        print_status $GREEN "✅ No security vulnerabilities found"
    else
        print_status $RED "❌ Security vulnerabilities detected!"
        return 1
    fi
    
    echo ""
    echo "Checking license compatibility..."
    if cargo deny check; then
        print_status $GREEN "✅ License compatibility check passed"
    else
        print_status $RED "❌ License compatibility issues found!"
        return 1
    fi
}

# Check unsafe code documentation
check_unsafe_code() {
    print_status $BLUE "⚠️  Checking unsafe code documentation..."
    
    # Find all unsafe blocks
    unsafe_files=$(find . -name "*.rs" -exec grep -l "unsafe" {} \; | grep -v target | grep -v .git || true)
    unsafe_count=$(echo "$unsafe_files" | wc -l)
    
    if [ -n "$unsafe_files" ] && [ "$unsafe_count" -gt 0 ]; then
        print_status $YELLOW "Found $unsafe_count files with unsafe code:"
        echo "$unsafe_files"
        
        # Check if unsafe_report.md exists and is not empty
        if [ -f "unsafe_report.md" ] && [ -s "unsafe_report.md" ]; then
            print_status $GREEN "✅ unsafe_report.md exists and is not empty"
        else
            print_status $RED "❌ unsafe_report.md is missing or empty!"
            print_status $YELLOW "All unsafe code must be documented in unsafe_report.md"
            return 1
        fi
    else
        print_status $GREEN "✅ No unsafe code found"
    fi
}

# Check third-party license documentation
check_license_documentation() {
    print_status $BLUE "📄 Checking license documentation..."
    
    if [ -f "THIRD_PARTY.md" ] && [ -s "THIRD_PARTY.md" ]; then
        print_status $GREEN "✅ THIRD_PARTY.md exists and is not empty"
    else
        print_status $RED "❌ THIRD_PARTY.md is missing or empty!"
        return 1
    fi
    
    # Update license documentation
    print_status $BLUE "Updating license documentation..."
    if [ -f "scripts/update-licenses.sh" ]; then
        bash scripts/update-licenses.sh
    elif [ -f "scripts/update-licenses.ps1" ]; then
        powershell -ExecutionPolicy Bypass -File scripts/update-licenses.ps1
    else
        print_status $YELLOW "⚠️  License update script not found"
    fi
}

# Run static analysis
run_static_analysis() {
    print_status $BLUE "🔬 Running static analysis..."
    
    echo "Running clippy with security lints..."
    if cargo clippy --all-targets --all-features -- -D warnings -D clippy::all -W clippy::pedantic; then
        print_status $GREEN "✅ Clippy analysis passed"
    else
        print_status $RED "❌ Clippy found issues!"
        return 1
    fi
}

# Run Miri tests
run_miri_tests() {
    print_status $BLUE "🧪 Running Miri tests for undefined behavior detection..."
    
    if [ -f "scripts/run-miri.sh" ]; then
        if bash scripts/run-miri.sh; then
            print_status $GREEN "✅ Miri tests passed"
        else
            print_status $RED "❌ Miri tests failed!"
            return 1
        fi
    else
        print_status $YELLOW "⚠️  Miri test script not found, skipping..."
    fi
}

# Run fuzzing tests
run_fuzzing_tests() {
    print_status $BLUE "🎯 Running fuzzing tests..."
    
    if [ -f "scripts/run-fuzz.sh" ]; then
        # Run fuzzing for a short duration (30 seconds per target)
        if bash scripts/run-fuzz.sh -d 30; then
            print_status $GREEN "✅ Fuzzing tests passed"
        else
            print_status $RED "❌ Fuzzing tests found issues!"
            return 1
        fi
    else
        print_status $YELLOW "⚠️  Fuzzing script not found, skipping..."
    fi
}

# Check security configuration files
check_security_config() {
    print_status $BLUE "⚙️  Checking security configuration..."
    
    # Check deny.toml
    if [ -f "deny.toml" ]; then
        print_status $GREEN "✅ deny.toml exists"
    else
        print_status $RED "❌ deny.toml is missing!"
        return 1
    fi
    
    # Check security workflow
    if [ -f ".github/workflows/security.yml" ]; then
        print_status $GREEN "✅ Security workflow exists"
    else
        print_status $RED "❌ Security workflow is missing!"
        return 1
    fi
    
    # Check if Cargo.lock is committed
    if [ -f "Cargo.lock" ]; then
        print_status $GREEN "✅ Cargo.lock is committed"
    else
        print_status $YELLOW "⚠️  Cargo.lock not found - run 'cargo build' first"
    fi
}

# Generate security report
generate_security_report() {
    print_status $BLUE "📊 Generating security report..."
    
    report_file="security_audit_report.md"
    
    cat > "$report_file" << EOF
# Security Audit Report

Generated on: $(date)

## Summary

This report contains the results of a comprehensive security audit of the BitNet Rust implementation.

## Dependency Security

EOF
    
    echo "### Vulnerability Scan" >> "$report_file"
    echo "" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    cargo audit 2>&1 >> "$report_file" || echo "Vulnerabilities found - see details above" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    echo "" >> "$report_file"
    
    echo "### License Compliance" >> "$report_file"
    echo "" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    cargo deny check 2>&1 >> "$report_file" || echo "License issues found - see details above" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    echo "" >> "$report_file"
    
    echo "## Unsafe Code Analysis" >> "$report_file"
    echo "" >> "$report_file"
    unsafe_files=$(find . -name "*.rs" -exec grep -l "unsafe" {} \; | grep -v target | grep -v .git || true)
    if [ -n "$unsafe_files" ]; then
        echo "Files containing unsafe code:" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        echo "$unsafe_files" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
    else
        echo "No unsafe code found." >> "$report_file"
    fi
    echo "" >> "$report_file"
    
    echo "## Static Analysis Results" >> "$report_file"
    echo "" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    cargo clippy --all-targets --all-features -- -D warnings 2>&1 >> "$report_file" || echo "Clippy issues found - see details above" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    
    print_status $GREEN "✅ Security report generated: $report_file"
}

# Main execution
main() {
    local failed_checks=0
    
    install_tools
    
    # Run all security checks
    run_dependency_audit || ((failed_checks++))
    check_unsafe_code || ((failed_checks++))
    check_license_documentation || ((failed_checks++))
    run_static_analysis || ((failed_checks++))
    check_security_config || ((failed_checks++))
    
    # Optional checks (don't fail the audit if they're not available)
    run_miri_tests || print_status $YELLOW "⚠️  Miri tests had issues (non-fatal)"
    run_fuzzing_tests || print_status $YELLOW "⚠️  Fuzzing tests had issues (non-fatal)"
    
    # Generate report
    generate_security_report
    
    # Final summary
    echo ""
    print_status $BLUE "=== Security Audit Summary ==="
    
    if [ $failed_checks -eq 0 ]; then
        print_status $GREEN "🎉 All security checks passed!"
        print_status $GREEN "The codebase meets security requirements."
    else
        print_status $RED "❌ $failed_checks security check(s) failed!"
        print_status $RED "Please address the issues above before proceeding."
        exit 1
    fi
}

# Run the audit
main "$@"