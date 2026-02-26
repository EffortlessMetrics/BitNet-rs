# Fast Feedback Demo Script for BitNet-rs (PowerShell)
# This script demonstrates the fast feedback system capabilities

param(
    [string]$Mode = "auto"
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

Write-Host "🚀 BitNet-rs Fast Feedback System Demo" -ForegroundColor $Blue
Write-Host "======================================" -ForegroundColor $Blue

# Check if we're in the right directory
if (-not (Test-Path "Cargo.toml")) {
    Write-Error "Please run this script from the BitNet-rs root directory"
    exit 1
}

# Create logs directory if it doesn't exist
if (-not (Test-Path "tests/logs")) {
    New-Item -ItemType Directory -Path "tests/logs" -Force | Out-Null
}

Write-Status "Setting up fast feedback demo environment..."

# Set environment variables for demo
$env:BITNET_FAST_FEEDBACK = "1"
$env:BITNET_TEST_MODE = "demo"
$env:RUST_LOG = "info"

Write-Status "Environment variables set:"
Write-Host "  BITNET_FAST_FEEDBACK=1"
Write-Host "  BITNET_TEST_MODE=demo"
Write-Host "  RUST_LOG=info"

# Function to run fast feedback with different configurations
function Invoke-FastFeedback {
    param(
        [string]$TestMode,
        [string]$Description
    )

    Write-Host ""
    Write-Status "Running fast feedback in $TestMode mode: $Description"
    Write-Host "----------------------------------------"

    # Build the demo binary if it doesn't exist
    if (-not (Test-Path "target/debug/fast_feedback_demo.exe")) {
        Write-Status "Building fast feedback demo binary..."
        cargo build --bin fast_feedback_demo
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to build demo binary, continuing with available tests..."
            return
        }
    }

    # Run the demo
    try {
        cargo run --bin fast_feedback_demo -- $TestMode
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Fast feedback completed successfully in $TestMode mode"
        } else {
            Write-Warning "Fast feedback encountered issues in $TestMode mode"
        }
    } catch {
        Write-Warning "Error running fast feedback in $TestMode mode: $_"
    }
}

# Demo 1: Development mode (fastest feedback)
Invoke-FastFeedback "dev" "Optimized for development with 30-second target"

# Demo 2: CI mode (balanced speed and coverage)
Invoke-FastFeedback "ci" "Optimized for CI with 90-second target"

# Demo 3: Auto-detection mode
Invoke-FastFeedback "auto" "Auto-detected configuration based on environment"

# Demo 4: Default mode
Invoke-FastFeedback "default" "Default configuration with 2-minute target"

Write-Host ""
Write-Status "Demonstrating incremental testing..."

# Create a temporary file to simulate changes
"// Temporary change for demo" | Out-File -FilePath "temp_change.rs" -Encoding UTF8

Write-Status "Simulated file change: temp_change.rs"
Write-Status "Fast feedback should detect this change and run affected tests"

# Run incremental test
Invoke-FastFeedback "dev" "Incremental testing with simulated changes"

# Clean up
if (Test-Path "temp_change.rs") {
    Remove-Item "temp_change.rs"
    Write-Status "Cleaned up temporary files"
}

Write-Host ""
Write-Status "Fast feedback demo scenarios completed!"

# Show configuration file
if (Test-Path "tests/fast-feedback.toml") {
    Write-Host ""
    Write-Status "Fast feedback configuration file available at: tests/fast-feedback.toml"
    Write-Status "You can customize the configuration by editing this file"
}

# Show logs
if (Test-Path "tests/logs/fast-feedback.log") {
    Write-Host ""
    Write-Status "Fast feedback logs available at: tests/logs/fast-feedback.log"
    Write-Status "Last 10 lines of the log:"
    Write-Host "----------------------------------------"
    Get-Content "tests/logs/fast-feedback.log" -Tail 10
}

Write-Host ""
Write-Success "Demo completed! Fast feedback system is ready for use."

# Usage instructions
Write-Host ""
Write-Status "Usage instructions:"
Write-Host "  1. For development: cargo run --bin fast_feedback_demo -- dev"
Write-Host "  2. For CI: cargo run --bin fast_feedback_demo -- ci"
Write-Host "  3. Auto-detect: cargo run --bin fast_feedback_demo -- auto"
Write-Host "  4. Custom config: Edit tests/fast-feedback.toml and run with default mode"

Write-Host ""
Write-Status "Environment integration:"
Write-Host "  - Set BITNET_FAST_FEEDBACK=1 to enable fast feedback"
Write-Host "  - Set BITNET_INCREMENTAL=1 to enable incremental testing"
Write-Host "  - CI environments automatically use optimized settings"

Write-Host ""
Write-Status "Performance targets:"
Write-Host "  - Development: 30 seconds for immediate feedback"
Write-Host "  - CI: 90 seconds for balanced speed and coverage"
Write-Host "  - Full suite: 15 minutes maximum execution time"

Write-Host ""
Write-Success "🎉 Fast feedback system demo completed successfully!" -ForegroundColor $Green
