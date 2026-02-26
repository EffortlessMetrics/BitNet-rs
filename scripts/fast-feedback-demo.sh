#!/bin/bash

# Fast Feedback Demo Script for BitNet-rs
# This script demonstrates the fast feedback system capabilities

set -e

echo "🚀 BitNet-rs Fast Feedback System Demo"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Please run this script from the BitNet-rs root directory"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p tests/logs

print_status "Setting up fast feedback demo environment..."

# Set environment variables for demo
export BITNET_FAST_FEEDBACK=1
export BITNET_TEST_MODE=demo
export RUST_LOG=info

print_status "Environment variables set:"
echo "  BITNET_FAST_FEEDBACK=1"
echo "  BITNET_TEST_MODE=demo"
echo "  RUST_LOG=info"

# Function to run fast feedback with different configurations
run_fast_feedback() {
    local mode=$1
    local description=$2

    echo ""
    print_status "Running fast feedback in $mode mode: $description"
    echo "----------------------------------------"

    # Build the demo binary if it doesn't exist
    if [ ! -f "target/debug/fast_feedback_demo" ]; then
        print_status "Building fast feedback demo binary..."
        cargo build --bin fast_feedback_demo
    fi

    # Run the demo
    if cargo run --bin fast_feedback_demo -- "$mode"; then
        print_success "Fast feedback completed successfully in $mode mode"
    else
        print_warning "Fast feedback encountered issues in $mode mode"
    fi
}

# Demo 1: Development mode (fastest feedback)
run_fast_feedback "dev" "Optimized for development with 30-second target"

# Demo 2: CI mode (balanced speed and coverage)
run_fast_feedback "ci" "Optimized for CI with 90-second target"

# Demo 3: Auto-detection mode
run_fast_feedback "auto" "Auto-detected configuration based on environment"

# Demo 4: Default mode
run_fast_feedback "default" "Default configuration with 2-minute target"

echo ""
print_status "Demonstrating incremental testing..."

# Create a temporary file to simulate changes
echo "// Temporary change for demo" > temp_change.rs

print_status "Simulated file change: temp_change.rs"
print_status "Fast feedback should detect this change and run affected tests"

# Run incremental test
run_fast_feedback "dev" "Incremental testing with simulated changes"

# Clean up
rm -f temp_change.rs
print_status "Cleaned up temporary files"

echo ""
print_status "Fast feedback demo scenarios completed!"

# Show configuration file
if [ -f "tests/fast-feedback.toml" ]; then
    echo ""
    print_status "Fast feedback configuration file available at: tests/fast-feedback.toml"
    print_status "You can customize the configuration by editing this file"
fi

# Show logs
if [ -f "tests/logs/fast-feedback.log" ]; then
    echo ""
    print_status "Fast feedback logs available at: tests/logs/fast-feedback.log"
    print_status "Last 10 lines of the log:"
    echo "----------------------------------------"
    tail -n 10 tests/logs/fast-feedback.log
fi

echo ""
print_success "Demo completed! Fast feedback system is ready for use."

# Usage instructions
echo ""
print_status "Usage instructions:"
echo "  1. For development: cargo run --bin fast_feedback_demo -- dev"
echo "  2. For CI: cargo run --bin fast_feedback_demo -- ci"
echo "  3. Auto-detect: cargo run --bin fast_feedback_demo -- auto"
echo "  4. Custom config: Edit tests/fast-feedback.toml and run with default mode"

echo ""
print_status "Environment integration:"
echo "  - Set BITNET_FAST_FEEDBACK=1 to enable fast feedback"
echo "  - Set BITNET_INCREMENTAL=1 to enable incremental testing"
echo "  - CI environments automatically use optimized settings"

echo ""
print_status "Performance targets:"
echo "  - Development: 30 seconds for immediate feedback"
echo "  - CI: 90 seconds for balanced speed and coverage"
echo "  - Full suite: 15 minutes maximum execution time"

echo ""
print_success "🎉 Fast feedback system demo completed successfully!"
