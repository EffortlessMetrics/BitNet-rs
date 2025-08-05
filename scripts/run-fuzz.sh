#!/bin/bash
# Script to run fuzzing tests

set -e

echo "Setting up fuzzing environment..."

# Install cargo-fuzz if not present
if ! command -v cargo-fuzz &> /dev/null; then
    echo "Installing cargo-fuzz..."
    cargo install cargo-fuzz
fi

# Initialize fuzz directory if it doesn't exist
if [ ! -d "fuzz" ]; then
    echo "Initializing fuzz directory..."
    cargo fuzz init
fi

# List of fuzz targets
TARGETS=(
    "quantization_i2s"
    "gguf_parser"
    "kernel_matmul"
)

# Function to run a single fuzz target
run_fuzz_target() {
    local target=$1
    local duration=${2:-60}  # Default 60 seconds
    
    echo "Running fuzz target: $target for ${duration}s"
    
    # Create artifacts directory
    mkdir -p "fuzz/artifacts/$target"
    
    # Run the fuzzer
    timeout ${duration}s cargo fuzz run $target -- -max_total_time=${duration} || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "Fuzzing completed (timeout reached)"
        else
            echo "Fuzzing failed with exit code: $exit_code"
            return $exit_code
        fi
    }
    
    # Check for crashes
    if [ -d "fuzz/artifacts/$target" ] && [ "$(ls -A fuzz/artifacts/$target)" ]; then
        echo "⚠️  Crashes found for $target:"
        ls -la "fuzz/artifacts/$target"
        return 1
    else
        echo "✅ No crashes found for $target"
    fi
}

# Parse command line arguments
DURATION=60
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --target TARGET    Run specific fuzz target"
            echo "  -d, --duration SECONDS Duration to run each target (default: 60)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Available targets:"
            for target in "${TARGETS[@]}"; do
                echo "  - $target"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run specific target or all targets
if [ -n "$TARGET" ]; then
    if [[ " ${TARGETS[@]} " =~ " ${TARGET} " ]]; then
        run_fuzz_target "$TARGET" "$DURATION"
    else
        echo "Error: Unknown target '$TARGET'"
        echo "Available targets: ${TARGETS[*]}"
        exit 1
    fi
else
    echo "Running all fuzz targets for ${DURATION}s each..."
    failed_targets=()
    
    for target in "${TARGETS[@]}"; do
        if ! run_fuzz_target "$target" "$DURATION"; then
            failed_targets+=("$target")
        fi
        echo ""
    done
    
    # Summary
    echo "=== Fuzzing Summary ==="
    if [ ${#failed_targets[@]} -eq 0 ]; then
        echo "✅ All fuzz targets passed!"
    else
        echo "❌ Failed targets: ${failed_targets[*]}"
        exit 1
    fi
fi

echo "Fuzzing completed!"