# BitNet.rs Testing Framework Documentation

This directory contains comprehensive documentation for the BitNet.rs testing framework, covering everything from basic usage to advanced testing scenarios.

## Documentation Structure

### Getting Started
- **[Quick Start Guide](quick-start-guide.md)** - Get writing tests in under 10 minutes
- **[Test Templates](test-templates.md)** - Copy-paste templates for common test scenarios

### Comprehensive Guides
- **[Framework Overview](framework-overview.md)** - Architecture and core concepts
- **[Test Authoring Guide](test-authoring-guide.md)** - Guidelines and best practices for writing tests
- **[Cross-Validation Guide](cross-validation-guide.md)** - Setup and usage of cross-implementation comparison
- **[Performance Testing Guide](performance-testing-guide.md)** - Benchmarking and performance validation
- **[Troubleshooting Guide](troubleshooting-guide.md)** - Common issues and debugging techniques

## Quick Start

For developers new to the testing framework:

1. **New to testing?** Start with the [Quick Start Guide](quick-start-guide.md) to write your first test in 5 minutes
2. **Need examples?** Check out the [Test Templates](test-templates.md) for copy-paste examples
3. **Want to understand the architecture?** Read the [Framework Overview](framework-overview.md)
4. **Ready for advanced patterns?** Follow the [Test Authoring Guide](test-authoring-guide.md)
5. **Need to validate against C++?** Use the [Cross-Validation Guide](cross-validation-guide.md)
6. **Having issues?** Refer to the [Troubleshooting Guide](troubleshooting-guide.md)

## Running Tests

```bash
# Run all tests
cargo test --no-default-features --features cpu

# Run unit tests only
cargo test --no-default-features --features cpu --lib

# Run integration tests
cargo test --no-default-features --features cpu --test integration_tests

# Run cross-validation tests
cargo test --no-default-features --features crossval --test crossval_tests

# Generate coverage report
cargo tarpaulin --out html --output-dir target/coverage
```

## Example Tests

The framework includes several example tests that demonstrate best practices:

```bash
# Run the unit test example (shows 14 different test patterns)
cargo test --no-default-features --features cpu --test unit_test_example

# Run the integration test example (shows workflow and component testing)
cargo test --no-default-features --features cpu --test integration_test_example

# Run the reporting system example
cargo run -p bitnet-tests --example reporting_example
```

These examples show:
- **Unit Test Example** (`tests/unit_test_example.rs`): Comprehensive unit testing patterns including error handling, performance testing, resource management, caching, and concurrent testing
- **Integration Test Example** (`tests/integration_test_example.rs`): Complete workflow testing with component interactions, data flow validation, file I/O, and system integration
- **Reporting Example** (`tests/examples/reporting_example.rs`): How to generate HTML, JSON, JUnit XML, and Markdown reports

You can also run specific tests from the examples:
```bash
# Run a specific unit test
cargo test --no-default-features --features cpu --test unit_test_example test_process_basic_functionality

# List all available tests in an example
cargo test --no-default-features --features cpu --test unit_test_example -- --list
```

## Developer Tools

The framework includes helpful tools to make test authoring easier:

### Test Template Generator

Quickly create test files from templates:

```bash
# Linux/macOS
./scripts/create-test-template.sh my_feature_test

# Windows PowerShell  
.\scripts\create-test-template.ps1 my_feature_test
```

See [Testing Tools README](../../scripts/testing-tools-README.md) for full usage instructions.

## Contributing

When adding new tests or modifying the testing framework, please:

1. Use the test template generator to create properly structured test files
2. Follow the guidelines in the [Test Authoring Guide](test-authoring-guide.md)
3. Update documentation as needed
4. Ensure all tests pass before submitting changes
5. Add appropriate test coverage for new functionality

## Support

If you encounter issues not covered in the troubleshooting guide, please:

1. Check existing GitHub issues
2. Create a new issue with detailed reproduction steps
3. Include relevant logs and system information