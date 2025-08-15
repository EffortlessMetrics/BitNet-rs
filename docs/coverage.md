# Coverage Collection and Reporting

This document describes the coverage collection system for BitNet.rs, including local development workflows and CI integration.

## Quick Start

### Local Coverage Collection

```bash
# Install tarpaulin (Linux recommended)
cargo install cargo-tarpaulin --locked

# Run coverage with HTML output
cargo cov-html

# Run coverage with all formats (XML, LCOV, HTML)
cargo cov-all

# View results
open target/coverage/tarpaulin-report.html
```

### Using Aliases

The project includes convenient cargo aliases:

```bash
# Coverage collection
cargo cov          # XML + LCOV output
cargo cov-html      # HTML output only  
cargo cov-all       # All formats

# Testing
cargo test-unit     # Unit tests only
cargo test-integration  # Integration tests only
cargo test-all      # All tests

# Reporting
cargo report-example    # Generate sample reports
cargo report-demo       # Comprehensive demo
```

## Coverage Workflow

### 1. Local Development

```bash
# Run tests with coverage
cargo cov-html

# Check coverage thresholds
cargo cov | grep "Coverage"

# Generate reports
cargo report-example
```

### 2. CI Integration

The CI system automatically:

- Collects coverage on every push/PR
- Generates HTML, LCOV, and XML reports
- Uploads artifacts for 30 days
- Enforces minimum coverage thresholds
- Creates weekly coverage tracking issues

### 3. Coverage Types

- **Unit Tests**: Individual crate testing
- **Integration Tests**: Cross-crate workflows
- **Combined Coverage**: Comprehensive analysis

## Configuration

### Coverage Thresholds

Default minimum coverage: **90%**

Override in CI:
```yaml
workflow_dispatch:
  inputs:
    coverage_threshold:
      default: "85"  # Lower threshold
```

### Excluded Crates

The following crates are excluded from coverage:
- `bitnet-sys` - FFI bindings
- `crossval` - Cross-validation utilities  
- `xtask` - Build utilities
- `bitnet-cli` - CLI application

### Features

Coverage collection uses CPU-only features:
- `cpu` - CPU backend
- `avx2` - SIMD optimizations (when available)

## Output Formats

### HTML Reports
- **Location**: `target/coverage/tarpaulin-report.html`
- **Features**: Interactive line-by-line coverage
- **Best for**: Development and debugging

### LCOV Format
- **Location**: `target/coverage/lcov.info`
- **Features**: Machine-readable format
- **Best for**: CI integration and tooling

### XML Format (Cobertura)
- **Location**: `target/coverage/cobertura.xml`
- **Features**: CI-compatible format
- **Best for**: Coverage badges and dashboards

## CI Artifacts

Coverage artifacts are uploaded on every CI run:

### Available Downloads
- `coverage-unit-tests` - Unit test coverage
- `coverage-integration-tests` - Integration test coverage  
- `coverage-combined-coverage` - Complete analysis
- `coverage-summary` - Comprehensive summary

### Retention
- **Coverage Reports**: 30 days
- **Summary Reports**: 90 days

## Quality Gates

### Automatic Checks
- ✅ Minimum coverage threshold enforcement
- ✅ Per-crate coverage analysis
- ✅ Trend tracking over time
- ✅ Regression detection

### Manual Review
- 📊 Weekly coverage reports
- 🎯 Coverage goal tracking
- 📈 Improvement recommendations

## Troubleshooting

### Common Issues

**Tarpaulin not found**:
```bash
cargo install cargo-tarpaulin --locked
```

**Permission denied (Linux)**:
```bash
# Tarpaulin requires ptrace permissions
sudo sysctl kernel.yama.ptrace_scope=0
```

**Low coverage warnings**:
```bash
# Check which lines are uncovered
cargo cov-html
open target/coverage/tarpaulin-report.html
```

### Platform Support

- ✅ **Linux**: Full support with ptrace
- ⚠️ **macOS**: Limited support, use CI for official coverage
- ❌ **Windows**: Not supported, use CI for coverage collection

## Integration with Reporting System

Coverage data integrates with the test reporting system:

```bash
# Generate test reports with coverage context
cargo report-example

# View comprehensive analysis
open tests/example_reports/example_report.html
```

The HTML test reports include coverage summaries and links to detailed coverage analysis.

## Best Practices

### Development Workflow
1. Write tests with coverage in mind
2. Run `cargo cov-html` before committing
3. Review uncovered lines in HTML report
4. Add tests for critical uncovered paths

### CI Integration
1. Coverage collected automatically
2. Artifacts available for download
3. Thresholds enforced on PRs
4. Weekly reports track trends

### Quality Assurance
1. Maintain >90% line coverage
2. Focus on branch coverage for complex logic
3. Ensure integration test coverage
4. Monitor coverage trends over time

## Advanced Usage

### Custom Coverage Collection

```bash
# Specific crate coverage
cargo tarpaulin --package bitnet-common --out Html

# With specific features
cargo tarpaulin --features "cpu,avx2" --out Html

# Exclude specific tests
cargo tarpaulin --exclude-files "tests/integration/*" --out Html
```

### Coverage Analysis

```bash
# Generate detailed analysis
cargo cov-all

# Extract metrics
grep "Coverage" target/coverage/tarpaulin-report.txt

# Compare with previous runs
diff previous-coverage.lcov target/coverage/lcov.info
```

## Resources

- [Tarpaulin Documentation](https://github.com/xd009642/tarpaulin)
- [LCOV Format Specification](http://ltp.sourceforge.net/coverage/lcov/genhtml.1.php)
- [Codecov Integration](https://docs.codecov.io/docs)
- [Coverage Best Practices](https://testing.googleblog.com/2020/08/code-coverage-best-practices.html)