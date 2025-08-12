# Fast test execution script for Windows (PowerShell)
# Optimized for <15 minute execution time

param(
    [int]$TargetMinutes = 15,
    [string]$Profile = "fast",
    [int]$MaxParallel = 0,
    [switch]$Aggressive = $false,
    [switch]$SkipSlow = $true,
    [switch]$EnableCaching = $true,
    [switch]$Verbose = $false,
    [switch]$NoIncremental = $false,
    [string[]]$Categories = @(),
    [switch]$Help = $false
)

# Show help if requested
if ($Help) {
    Write-Host "BitNet Fast Test Runner (PowerShell)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "    .\scripts\fast-test.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "    -TargetMinutes <INT>     Target execution time in minutes [default: 15]"
    Write-Host "    -Profile <STRING>        Speed profile: lightning, fast, balanced, thorough [default: fast]"
    Write-Host "    -MaxParallel <INT>       Number of parallel test threads [default: auto]"
    Write-Host "    -Aggressive              Enable aggressive optimizations"
    Write-Host "    -SkipSlow                Skip slow tests [default: true]"
    Write-Host "    -EnableCaching           Enable test result caching [default: true]"
    Write-Host "    -Verbose                 Enable verbose output"
    Write-Host "    -NoIncremental           Disable incremental testing"
    Write-Host "    -Categories <LIST>       Comma-separated list of test categories"
    Write-Host "    -Help                    Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "    .\scripts\fast-test.ps1 -TargetMinutes 10 -Profile lightning"
    Write-Host "    .\scripts\fast-test.ps1 -MaxParallel 4 -Categories unit,integration"
    Write-Host "    .\scripts\fast-test.ps1 -Aggressive -Verbose"
    exit 0
}

# Configuration
$ErrorActionPreference = "Stop"
$StartTime = Get-Date

# Auto-detect parallel threads if not specified
if ($MaxParallel -eq 0) {
    $MaxParallel = [Environment]::ProcessorCount
    if ($MaxParallel -gt 8) { $MaxParallel = 8 }  # Cap at 8 for stability
}

# Colors for output
$Colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Write-Info { param([string]$Message) Write-ColorOutput "ℹ️ $Message" "Info" }
function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" "Success" }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️ $Message" "Warning" }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" "Error" }

Write-Info "Starting optimized test execution (target: $TargetMinutes minutes)"
Write-Info "Configuration:"
Write-Info "  - Max parallel: $MaxParallel"
Write-Info "  - Profile: $Profile"
Write-Info "  - Aggressive mode: $Aggressive"
Write-Info "  - Skip slow tests: $SkipSlow"
Write-Info "  - Enable caching: $EnableCaching"

# Set environment variables for optimization
$env:BITNET_TEST_PARALLEL = $MaxParallel
$env:BITNET_TEST_TIMEOUT = 60
$env:BITNET_TEST_LOG_LEVEL = if ($Verbose) { "debug" } else { "warn" }
$env:BITNET_TEST_GENERATE_COVERAGE = "false"
$env:BITNET_TEST_CACHE_DIR = "tests\cache"
$env:BITNET_TEST_AUTO_DOWNLOAD = if ($EnableCaching) { "true" } else { "false" }
$env:BITNET_TEST_MODE = "fast"
$env:RUST_BACKTRACE = "0"
$env:CARGO_TERM_QUIET = if (-not $Verbose) { "true" } else { "false" }

# Create cache directory
if (-not (Test-Path "tests\cache")) {
    New-Item -ItemType Directory -Path "tests\cache" -Force | Out-Null
}

# Function to run tests with timeout
function Invoke-TestsWithTimeout {
    param(
        [string]$TestArgs,
        [int]$TimeoutSeconds
    )
    
    Write-Info "Running tests with ${TimeoutSeconds}s timeout: cargo test $TestArgs"
    
    $job = Start-Job -ScriptBlock {
        param($args, $env_vars)
        
        # Set environment variables in job
        foreach ($var in $env_vars.GetEnumerator()) {
            Set-Item -Path "env:$($var.Key)" -Value $var.Value
        }
        
        # Run cargo test
        $process = Start-Process -FilePath "cargo" -ArgumentList $args -NoNewWindow -Wait -PassThru
        return $process.ExitCode
    } -ArgumentList @($TestArgs -split ' '), @{
        BITNET_TEST_PARALLEL = $env:BITNET_TEST_PARALLEL
        BITNET_TEST_TIMEOUT = $env:BITNET_TEST_TIMEOUT
        BITNET_TEST_LOG_LEVEL = $env:BITNET_TEST_LOG_LEVEL
        BITNET_TEST_GENERATE_COVERAGE = $env:BITNET_TEST_GENERATE_COVERAGE
        BITNET_TEST_CACHE_DIR = $env:BITNET_TEST_CACHE_DIR
        BITNET_TEST_AUTO_DOWNLOAD = $env:BITNET_TEST_AUTO_DOWNLOAD
        BITNET_TEST_MODE = $env:BITNET_TEST_MODE
        RUST_BACKTRACE = $env:RUST_BACKTRACE
        CARGO_TERM_QUIET = $env:CARGO_TERM_QUIET
    }
    
    # Wait for job with timeout
    $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds
    
    if ($completed) {
        $result = Receive-Job -Job $job
        Remove-Job -Job $job
        return $result
    } else {
        Write-Warning "Tests timed out after $TimeoutSeconds seconds"
        Stop-Job -Job $job
        Remove-Job -Job $job
        return 124  # Timeout exit code
    }
}

# Function to estimate test execution time
function Get-TestTimeEstimate {
    Write-Info "Analyzing test suite..."
    
    try {
        # Get list of test executables
        $testList = cargo test --workspace --no-run --message-format=json 2>$null | 
                   ConvertFrom-Json | 
                   Where-Object { $_.reason -eq "compiler-artifact" -and $_.target.kind -contains "test" } |
                   Select-Object -ExpandProperty executable
        
        if ($testList) {
            $testCount = $testList.Count
            Write-Info "Found $testCount test executables"
            
            # Estimate based on historical data or defaults
            $estimatedTimePerTest = 5  # seconds
            $totalEstimatedTime = $testCount * $estimatedTimePerTest
            $parallelEstimatedTime = [math]::Ceiling($totalEstimatedTime / $MaxParallel)
            
            Write-Info "Estimated execution time: ${parallelEstimatedTime}s (${totalEstimatedTime}s sequential)"
            
            return $parallelEstimatedTime -le ($TargetMinutes * 60)
        }
    } catch {
        Write-Warning "Could not analyze test suite: $($_.Exception.Message)"
    }
    
    return $true  # Assume we can run within time if analysis fails
}

# Function to run optimized test selection
function Invoke-OptimizedTests {
    $testArgs = @("test", "--workspace", "--test-threads=$MaxParallel")
    
    if ($Aggressive) {
        # Skip documentation tests for speed
        $testArgs += @("--lib", "--bins")
        
        # Skip slow integration tests if needed
        if ($SkipSlow) {
            $testArgs += "--exclude=crossval"
            Write-Info "Skipping slow cross-validation tests"
        }
    }
    
    # Add categories filter if specified
    if ($Categories.Count -gt 0) {
        foreach ($category in $Categories) {
            $testArgs += "--package=bitnet-$category"
        }
    }
    
    # Add timeout per test
    $testArgs += @("--", "--test-timeout=60")
    
    $timeoutSeconds = $TargetMinutes * 60
    return Invoke-TestsWithTimeout ($testArgs -join " ") $timeoutSeconds
}

# Function to run incremental tests (only changed code)
function Invoke-IncrementalTests {
    Write-Info "Attempting incremental test execution..."
    
    try {
        # Check if we can determine changed files using git
        if (Get-Command git -ErrorAction SilentlyContinue) {
            $changedFiles = git diff --name-only HEAD~1 2>$null
            if (-not $changedFiles) {
                $changedFiles = git diff --name-only --cached 2>$null
            }
            
            if ($changedFiles) {
                Write-Info "Detected changes in:"
                $changedFiles | ForEach-Object { Write-Info "  - $_" }
                
                # Run tests for changed crates only
                $changedCrates = @()
                foreach ($file in $changedFiles) {
                    if ($file -match "^crates/([^/]+)/") {
                        $crateName = $matches[1]
                        if ($changedCrates -notcontains $crateName) {
                            $changedCrates += $crateName
                        }
                    }
                }
                
                if ($changedCrates.Count -gt 0) {
                    Write-Info "Running tests for changed crates: $($changedCrates -join ', ')"
                    $testArgs = @("test") + ($changedCrates | ForEach-Object { "-p", $_ }) + @("--test-threads=$MaxParallel")
                    $timeoutSeconds = $TargetMinutes * 60
                    return Invoke-TestsWithTimeout ($testArgs -join " ") $timeoutSeconds
                }
            }
        }
    } catch {
        Write-Warning "Could not determine incremental changes: $($_.Exception.Message)"
    }
    
    Write-Info "Could not determine incremental changes, running full test suite"
    return $null
}

# Function to run fast unit tests only
function Invoke-FastUnitTests {
    Write-Info "Running fast unit tests only..."
    
    $testArgs = @(
        "test", 
        "--workspace", 
        "--lib", 
        "--test-threads=$MaxParallel",
        "--exclude=crossval",
        "--exclude=bitnet-sys",
        "--",
        "--test-timeout=30"
    )
    
    $timeoutSeconds = $TargetMinutes * 60
    return Invoke-TestsWithTimeout ($testArgs -join " ") $timeoutSeconds
}

# Function to cleanup and report
function Write-FinalReport {
    param([int]$ExitCode)
    
    $endTime = Get-Date
    $duration = $endTime - $StartTime
    $durationMinutes = [math]::Floor($duration.TotalMinutes)
    $durationSeconds = [math]::Floor($duration.TotalSeconds % 60)
    
    Write-Info "Test execution completed in ${durationMinutes}m ${durationSeconds}s"
    
    if ($ExitCode -eq 0) {
        if ($duration.TotalMinutes -le $TargetMinutes) {
            Write-Success "Tests completed successfully within $TargetMinutes minute target!"
        } else {
            Write-Warning "Tests completed successfully but exceeded $TargetMinutes minute target"
        }
    } else {
        Write-Error "Tests failed with exit code $ExitCode"
    }
    
    # Generate simple report
    $reportContent = @"
# Test Execution Report

**Target Time:** $TargetMinutes minutes
**Actual Time:** ${durationMinutes}m ${durationSeconds}s
**Status:** $(if ($ExitCode -eq 0) { "PASSED" } else { "FAILED" })
**Exit Code:** $ExitCode

## Configuration
- Max Parallel: $MaxParallel
- Profile: $Profile
- Aggressive Mode: $Aggressive
- Skip Slow Tests: $SkipSlow
- Enable Caching: $EnableCaching

## Performance
- Time Efficiency: $([math]::Round(($TargetMinutes * 60 * 100) / $duration.TotalSeconds, 1))% of target
- Parallel Efficiency: Estimated $([math]::Round($MaxParallel * 100 / ($MaxParallel + 1), 1))%

Generated at: $(Get-Date)
"@
    
    $reportContent | Out-File -FilePath "test-execution-report.txt" -Encoding UTF8
    Write-Info "Report saved to test-execution-report.txt"
}

# Main execution logic
try {
    # Check prerequisites
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo not found in PATH"
    }
    
    if (-not (Test-Path "Cargo.toml")) {
        throw "Cargo.toml not found - not in a Rust workspace"
    }
    
    # Determine execution strategy
    $exitCode = $null
    
    if (Get-TestTimeEstimate) {
        Write-Info "Estimated time is within target, running optimized test suite"
        $exitCode = Invoke-OptimizedTests
    } else {
        Write-Warning "Estimated time exceeds target, trying optimizations..."
        
        # Try incremental tests first
        if ($EnableCaching -and -not $NoIncremental) {
            $exitCode = Invoke-IncrementalTests
        }
        
        # Fall back to fast unit tests if incremental didn't work
        if ($null -eq $exitCode) {
            Write-Info "Falling back to fast unit tests only"
            $exitCode = Invoke-FastUnitTests
        }
    }
    
    # Handle null exit code (shouldn't happen, but just in case)
    if ($null -eq $exitCode) {
        $exitCode = 1
    }
    
    Write-FinalReport $exitCode
    exit $exitCode
    
} catch {
    Write-Error "Script execution failed: $($_.Exception.Message)"
    Write-FinalReport 1
    exit 1
}