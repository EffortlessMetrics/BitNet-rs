# PowerShell script to check for banned patterns in test API
$ErrorActionPreference = "Stop"
$fail = 0

Write-Host "🔍 Checking for banned patterns in test API..." -ForegroundColor Cyan

# Check for legacy TestResult<T> generic type
if (rg -n --glob 'tests/common/**' --pcre2 '->\s*TestResult<' 2>$null) {
    Write-Host "❌ Found legacy 'TestResult<T>' – use 'TestResultCompat<T>'." -ForegroundColor Red
    $fail = 1
}

# Check for legacy TestResultData alias
if (rg -n --glob 'tests/common/**' --pcre2 '\bTestResultData\b' 2>$null) {
    Write-Host "❌ Found legacy 'TestResultData' – use 'TestRecord'." -ForegroundColor Red
    $fail = 1
}

# Check for legacy TestResult constructors
if (rg -n --glob 'tests/common/**' --pcre2 '\bTestResult::(passed|failed|timeout)\b' 2>$null) {
    Write-Host "❌ Found legacy 'TestResult::...' – use 'TestRecord::...'." -ForegroundColor Red
    $fail = 1
}

# Check for split doc comments after closing brace
if (rg -n --glob 'tests/**' --pcre2 '^\s*}\s*///' 2>$null) {
    Write-Host "❌ Found split doc-comment after a '}'." -ForegroundColor Red
    $fail = 1
}

# Check for &Vec<T> anti-pattern (excluding target directory)
$vecRefs = rg -n --glob '**/*.rs' --pcre2 '&\s*Vec<[^>]+>' 2>$null | Where-Object { $_ -notmatch 'target' }
if ($vecRefs) {
    Write-Host "❌ Prefer '&[T]' over '&Vec<T>':" -ForegroundColor Red
    $vecRefs | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    $fail = 1
}

if ($fail -ne 0) {
    Write-Host "`n❌ API contract checks failed!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ All API contract checks passed!" -ForegroundColor Green
}