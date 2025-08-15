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
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests
cargo test --test integration_tests

# Run cross-validation tests
cargo test --test crossval_tests

# Generate coverage report
cargo tarpaulin --out html --output-dir target/coverage
```

## Example Tests

The framework includes several example tests that demonstrate best practices:

```bash
# Run the unit test example
cargo test --test unit_test_example

# Run the integration test example  
cargo test --test integration_test_example

# Run the reporting system example
cargo run -p bitnet-tests --example reporting_example
```

These examples show:
- **Unit Test Example**: Comprehensive unit testing patterns including error handling, performance testing, and resource management
- **Integration Test Example**: Complete workflow testing with component interactions and data flow validation
- **Reporting Example**: How to generate HTML, JSON, JUnit XML, and Markdown reports

## Contributing

When adding new tests or modifying the testing framework, please:

1. Follow the guidelines in the [Test Authoring Guide](test-authoring-guide.md)
2. Update documentation as needed
3. Ensure all tests pass before submitting changes
4. Add appropriate test coverage for new functionality

## Support

If you encounter issues not covered in the troubleshooting guide, please:

1. Check existing GitHub issues
2. Create a new issue with detailed reproduction steps
3. Include relevant logs and system information