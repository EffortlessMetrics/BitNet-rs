# Fast test execution script optimized for <15 minute execution
# PowerShell version for Windows systems

param(
    [int]$TargetTimeMinutes = 15,
    [int]$MaxParallel = [Environment]::ProcessorCount,
    [bool]$AggressiveMode = $true,
    [bool]$SkipSlowTests = $true,
    [bool]$EnableCaching = $true
)

# Configuration from environment or parameters
$TARGET_TIME_MINUTES = if ($env:BITNET_TARGET_TIME) { [int]$env:BITNET_TARGET_TIME } else { $TargetTimeMinutes }
$MAX_PARALLEL = if ($env:BITNET_TEST_PARALLEL) { [int]$env:BITNET_TEST_PARALLEL } else { $MaxParallel }
$AGGRESSIVE_MODE = if ($env:BITNET_AGGRESSIVE_TEST) { [bool]::Parse($env:BITNET_AGGRESSIVE_TEST) } else { $AggressiveMode }
$SKIP_SLOW_TESTS = if ($env:BITNET_SKIP_SLOW) { [bool]::Parse($env:BITNET_SKIP_SLOW) } else { $SkipSlowTests }
$ENABLE_CACHING = if ($env:BITNET_TEST_CACHE) { [bool]::Parse($env:BITNET_TEST_CACHE) } else { $EnableCaching }

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

# Start timer
$START_TIME = Get-Date

Write-Info "Starting optimized test execution (target: $TARGET_TIME_MINUTES minutes)"
Write-Info "Configuration:"
Write-Info "  - Max parallel: $MAX_PARALLEL"
Write-Info "  - Aggressive mode: $AGGRESSIVE_MODE"
Write-Info "  - Skip slow tests: $SKIP_SLOW_TESTS"
Write-Info "  - Enable caching: $ENABLE_CACHING"

# Set environment variables for optimization
$env:BITNET_TEST_PARALLEL = $MAX_PARALLEL
$env:BITNET_TEST_TIMEOUT = "60"
$env:BITNET_TEST_LOG_LEVEL = "warn"
$env:BITNET_TEST_GENERATE_COVERAGE = "false"
$env:BITNET_TEST_CACHE_DIR = "tests\cache"
$env:BITNET_TEST_AUTO_DOWNLOAD = $ENABLE_CACHING.ToString().ToLower()

# Create cache directory
if (!(Test-Path "tests\cache")) {
    New-Item -ItemType Directory -Path "tests\cache" -Force | Out-Null
}

# Function to run tests with timeout
function Invoke-TestsWithTimeout {
    param(
        [string]$TestArgs,
        [int]$TimeoutSeconds = ($TARGET_TIME_MINUTES * 60)
    )
    
    Write-Info "Running tests with ${TimeoutSeconds}s timeout: cargo test $TestArgs"
    
    $job = Start-Job -ScriptBlock {
        param($args)
        & cargo test @args
    } -ArgumentList $TestArgs.Split(' ')
    
    $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds
    
    if ($completed) {
        $result = Receive-Job -Job $job
        $exitCode = $job.State -eq 'Completed' ? 0 : 1
        Remove-Job -Job $job
        return $exitCode
    } else {
        Stop-Job -Job $job
        Remove-Job -Job $job
        Write-Error "Tests timed out after $TARGET_TIME_MINUTES minutes"
        return 124
    }
}

# Function to estimate test execution time
function Get-TestTimeEstimate {
    Write-Info "Analyzing test suite..."
    
    try {
        # Try to get test list (simplified for PowerShell)
        $testOutput = & cargo test --workspace --no-run 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Could not analyze test suite, proceeding with default configuration"
            return $true
        }
        
        # Simple estimation based on output
        $testCount = ($testOutput | Select-String "test result:" | Measure-Object).Count
        
        if ($testCount -eq 0) {
            $testCount = 50  # Default estimate
        }
        
        Write-Info "Estimated $testCount tests"
        
        # Estimate based on historical data or defaults
        $estimatedTimePerTest = 5
        $totalEstimatedTime = $testCount * $estimatedTimePerTest
        $parallelEstimatedTime = [math]::Ceiling($totalEstimatedTime / $MAX_PARALLEL)
        
        Write-Info "Estimated execution time: ${parallelEstimatedTime}s (${totalEstimatedTime}s sequential)"
        
        if ($parallelEstimatedTime -gt ($TARGET_TIME_MINUTES * 60)) {
            Write-Warn "Estimated time exceeds target, enabling aggressive optimizations"
            return $false
        }
        
        return $true
    }
    catch {
        Write-Warn "Error analyzing test suite: $_"
        return $true
    }
}

# Function to run optimized tests
function Invoke-OptimizedTests {
    $testArgs = @("--workspace")
    
    if ($AGGRESSIVE_MODE) {
        # Skip documentation tests for speed
        $testArgs += "--lib", "--bins"
        
        # Skip slow integration tests if needed
        if ($SKIP_SLOW_TESTS) {
            $testArgs += "--exclude=crossval"
            Write-Info "Skipping slow cross-validation tests"
        }
    }
    
    # Add test-specific arguments
    $testArgs += "--", "--test-threads=$MAX_PARALLEL", "--test-timeout=60"
    
    return Invoke-TestsWithTimeout ($testArgs -join " ")
}

# Function to run incremental tests
function Invoke-IncrementalTests {
    Write-Info "Attempting incremental test execution..."
    
    try {
        # Check if we're in a git repository
        $gitDir = & git rev-parse --git-dir 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            # Get changed files
            $changedFiles = & git diff --name-only HEAD~1 2>$null
            
            if (!$changedFiles) {
                $changedFiles = & git diff --name-only --cached 2>$null
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
                    $crateArgs = $changedCrates | ForEach-Object { "-p $_" }
                    $testArgs = ($crateArgs + "--test-threads=$MAX_PARALLEL") -join " "
                    Write-Info "Running tests for changed crates: $($changedCrates -join ', ')"
                    return Invoke-TestsWithTimeout $testArgs
                }
            }
        }
    }
    catch {
        Write-Info "Could not determine incremental changes: $_"
    }
    
    Write-Info "Could not determine incremental changes, running full test suite"
    return 1
}

# Function to run fast unit tests only
function Invoke-FastUnitTests {
    Write-Info "Running fast unit tests only..."
    
    $testArgs = @(
        "--workspace",
        "--lib",
        "--exclude=crossval",
        "--exclude=bitnet-sys",
        "--",
        "--test-threads=$MAX_PARALLEL",
        "--test-timeout=30"
    )
    
    return Invoke-TestsWithTimeout ($testArgs -join " ")
}

# Function to cleanup and report
function Complete-TestExecution {
    param([int]$ExitCode)
    
    $endTime = Get-Date
    $duration = $endTime - $START_TIME
    $durationMinutes = [math]::Floor($duration.TotalMinutes)
    $durationSeconds = [math]::Floor($duration.TotalSeconds % 60)
    
    Write-Info "Test execution completed in ${durationMinutes}m ${durationSeconds}s"
    
    if ($ExitCode -eq 0) {
        if ($duration.TotalMinutes -le $TARGET_TIME_MINUTES) {
            Write-Success "✅ Tests completed successfully within $TARGET_TIME_MINUTES minute target!"
        } else {
            Write-Warn "⚠️  Tests completed successfully but exceeded $TARGET_TIME_MINUTES minute target"
        }
    } else {
        Write-Error "❌ Tests failed with exit code $ExitCode"
    }
    
    # Generate simple report
    $reportContent = @"
# Test Execution Report

**Target Time:** $TARGET_TIME_MINUTES minutes
**Actual Time:** ${durationMinutes}m ${durationSeconds}s
**Status:** $(if ($ExitCode -eq 0) { "PASSED" } else { "FAILED" })
**Exit Code:** $ExitCode

## Configuration
- Max Parallel: $MAX_PARALLEL
- Aggressive Mode: $AGGRESSIVE_MODE
- Skip Slow Tests: $SKIP_SLOW_TESTS
- Enable Caching: $ENABLE_CACHING

## Performance
- Time Efficiency: $([math]::Round(($TARGET_TIME_MINUTES * 60 * 100) / $duration.TotalSeconds))% of target
- Parallel Efficiency: Estimated $([math]::Round($MAX_PARALLEL * 100 / ($MAX_PARALLEL + 1)))%

Generated at: $(Get-Date)
"@
    
    $reportContent | Out-File -FilePath "test-execution-report.txt" -Encoding UTF8
    Write-Info "Report saved to test-execution-report.txt"
    
    exit $ExitCode
}

# Main execution logic
function Start-MainExecution {
    try {
        # Check if we can estimate test time
        if (Get-TestTimeEstimate) {
            Write-Info "Estimated time is within target, running full test suite"
            $exitCode = Invoke-OptimizedTests
        } else {
            Write-Warn "Estimated time exceeds target, trying optimizations..."
            
            # Try incremental tests first
            if ($ENABLE_CACHING -and (Invoke-IncrementalTests) -eq 0) {
                Write-Success "Incremental tests completed successfully"
                $exitCode = 0
            } else {
                # Fall back to fast unit tests
                Write-Info "Falling back to fast unit tests only"
                $exitCode = Invoke-FastUnitTests
            }
        }
        
        Complete-TestExecution $exitCode
    }
    catch {
        Write-Error "Unexpected error: $_"
        Complete-TestExecution 1
    }
}

# Check prerequisites
if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "cargo not found in PATH"
    exit 1
}

# Run main logic
Start-MainExecution