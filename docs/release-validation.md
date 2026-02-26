# Release Validation Pipeline

This document describes the comprehensive release validation pipeline for BitNet-rs, which ensures that release candidates meet all quality standards before being approved for release.

## Overview

The release validation pipeline is a comprehensive automated system that validates release candidates across multiple dimensions:

- **Cross-platform compatibility** - Ensures the software builds and runs on all supported platforms
- **Comprehensive testing** - Runs the full test suite including unit, integration, and cross-validation tests
- **Performance validation** - Compares performance against baseline to detect regressions
- **Security validation** - Scans for security vulnerabilities and unsafe code usage
- **Documentation validation** - Verifies documentation examples and links are valid
- **Quality gates** - Enforces minimum quality standards before release approval

## Usage

### Triggering Release Validation

The release validation pipeline is triggered manually via GitHub Actions workflow dispatch:

```bash
# Navigate to Actions tab in GitHub
# Select "Release Validation Pipeline"
# Click "Run workflow"
# Fill in the required parameters:
#   - Release candidate version (e.g., v1.0.0-rc.1)
#   - Baseline version for comparison (e.g., v0.9.0 or main)
#   - Optional: Skip performance validation
#   - Optional: Skip cross-validation tests
```

### Required Parameters

- **Release Candidate Version**: The version being validated (must follow semver format)
- **Baseline Version**: The version to compare against for performance validation

### Optional Parameters

- **Skip Performance**: Skip performance benchmarking (useful for documentation-only releases)
- **Skip Cross-validation**: Skip cross-implementation comparison tests

## Pipeline Stages

### 1. Pre-validation Checks

- Validates release candidate version format
- Checks that the tag doesn't already exist
- Determines validation scope based on parameters

### 2. Cross-platform Validation Matrix

Tests the software across multiple dimensions:

**Operating Systems:**
- Ubuntu (latest and LTS)
- macOS (Intel and Apple Silicon)
- Windows (MSVC and GNU toolchains)

**Rust Versions:**
- Stable
- MSRV (Minimum Supported Rust Version)
- Beta
- Nightly (informational)

**Feature Combinations:**
- Default features
- GPU features
- All features

### 3. Comprehensive Test Suite

- **Unit Tests**: >90% code coverage across all crates
- **Integration Tests**: End-to-end workflow validation
- **Cross-validation Tests**: Rust vs C++ implementation comparison
- **Performance Tests**: Benchmark execution and regression detection

### 4. Performance Validation

Compares performance against the baseline version:

- **Inference Benchmarks**: Token generation speed and throughput
- **Kernel Benchmarks**: Low-level operation performance
- **Memory Usage**: Peak and average memory consumption
- **Regression Detection**: Identifies performance regressions >5%

### 5. Security and Quality Validation

- **Security Audit**: Scans for known vulnerabilities in dependencies
- **Unsafe Code Analysis**: Reviews unsafe code usage and patterns
- **Code Quality**: Runs clippy with strict lints
- **Formatting**: Ensures consistent code formatting

### 6. Documentation Validation

- **API Documentation**: Builds and validates all documentation
- **Example Validation**: Tests that README examples compile and run
- **Link Checking**: Verifies all documentation links are valid
- **Completeness**: Ensures all public APIs are documented

### 7. Quality Gates Evaluation

The pipeline enforces several quality gates that must pass:

| Quality Gate | Requirement | Failure Impact |
|--------------|-------------|----------------|
| Code Coverage | ≥85% | Blocks release |
| Performance Regression | <5% regression | Blocks release |
| Security Vulnerabilities | No high/critical | Blocks release |
| Cross-platform Builds | All platforms succeed | Blocks release |
| Documentation Links | ≥90% valid links | Warning only |

### 8. Release Approval Workflow

If all quality gates pass:

1. **Approval Issue Created**: GitHub issue created for maintainer review
2. **Manual Review**: Maintainers review validation results
3. **Approval Required**: Issue must be approved to proceed
4. **Automated Release**: Upon approval, release is automatically created

## Configuration

The pipeline behavior can be customized via `config/release-validation.toml`:

```toml
[quality_gates]
min_coverage = 85.0
max_performance_regression = 0.95
fail_on_security_levels = ["high", "critical"]

[validation_scope]
run_performance_validation = true
run_cross_validation = true
run_security_audit = true
validate_documentation = true
```

## Reports and Artifacts

The pipeline generates comprehensive reports:

### HTML Report
- **Visual Dashboard**: Interactive overview of validation results
- **Quality Gate Status**: Pass/fail status for each gate
- **Performance Charts**: Benchmark comparisons and trends
- **Coverage Details**: Line-by-line coverage analysis

### JSON Report
- **Machine Readable**: For integration with other tools
- **Detailed Metrics**: All performance and quality metrics
- **Historical Data**: For trend analysis and regression tracking

### Artifacts
- **Build Artifacts**: Binaries for all platforms
- **Test Results**: Detailed test execution logs
- **Coverage Reports**: HTML and JSON coverage data
- **Performance Data**: Benchmark results and comparisons

## Troubleshooting

### Common Issues

**Quality Gate Failures:**

1. **Coverage Below Threshold**
   - Add tests for uncovered code paths
   - Review coverage report for specific files
   - Consider adjusting threshold if appropriate

2. **Performance Regression**
   - Review performance comparison report
   - Identify specific benchmarks that regressed
   - Optimize code or adjust baseline if intentional

3. **Security Vulnerabilities**
   - Review security audit report
   - Update dependencies with vulnerabilities
   - Consider security patches or workarounds

4. **Cross-platform Build Failures**
   - Check platform-specific build logs
   - Verify dependencies are available on all platforms
   - Test locally on failing platform

**Pipeline Failures:**

1. **Timeout Issues**
   - Increase timeout values in configuration
   - Optimize slow tests or benchmarks
   - Check for resource contention

2. **Network Issues**
   - Retry the pipeline
   - Check external service availability
   - Use cached dependencies when possible

### Getting Help

- **GitHub Issues**: Report pipeline bugs or feature requests
- **Documentation**: Check docs/ directory for detailed guides
- **Maintainer Team**: Contact @maintainer-team for approval issues

## Best Practices

### Before Running Validation

1. **Local Testing**: Run tests locally first
2. **Performance Check**: Run benchmarks locally to catch obvious regressions
3. **Documentation Review**: Ensure all documentation is up to date
4. **Dependency Updates**: Update dependencies and run security audit

### During Validation

1. **Monitor Progress**: Watch the pipeline execution for early failure detection
2. **Review Logs**: Check detailed logs for any warnings or issues
3. **Performance Analysis**: Review performance trends and comparisons

### After Validation

1. **Report Review**: Thoroughly review the validation report
2. **Issue Resolution**: Address any quality gate failures
3. **Approval Process**: Ensure proper review and approval workflow
4. **Release Notes**: Prepare comprehensive release notes

## Integration with CI/CD

The release validation pipeline integrates with the broader CI/CD system:

- **Triggered After**: Feature development and testing completion
- **Triggers**: Release creation and deployment pipelines
- **Artifacts**: Used by downstream deployment processes
- **Notifications**: Integrated with team communication channels

## Security Considerations

- **Secrets Management**: All secrets are managed via GitHub Secrets
- **Access Control**: Pipeline requires appropriate permissions
- **Audit Trail**: All validation activities are logged and auditable
- **Vulnerability Scanning**: Automated security scanning is included

## Future Enhancements

Planned improvements to the release validation pipeline:

- **Automated Performance Baselines**: Dynamic baseline selection
- **ML-based Regression Detection**: Smarter performance regression detection
- **Extended Platform Support**: Additional architectures and operating systems
- **Integration Testing**: More comprehensive integration test scenarios
- **Deployment Validation**: Post-deployment validation and rollback capabilities
