# How to Use Incremental Testing

This guide explains how to use the incremental testing system in BitNet.rs.

The incremental testing system detects changes and runs only affected tests.

## Change Detection

- **Git Integration**: Detect changes since last commit
- **Filesystem Monitoring**: Track file modification times
- **Dependency Analysis**: Understand test dependencies

## Test Selection

- **Direct Dependencies**: Tests that directly use changed code
- **Transitive Dependencies**: Tests affected through dependencies
- **Pattern Matching**: Tests matching changed file patterns

## Example

```bash
# Make changes to a source file
echo "// New feature" >> crates/bitnet-common/src/lib.rs

# Run incremental testing
BITNET_INCREMENTAL=1 cargo run --bin fast_feedback_demo -- dev
```
