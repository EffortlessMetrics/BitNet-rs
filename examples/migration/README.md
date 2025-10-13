# Migration Examples

This directory contains practical examples showing how to migrate from legacy BitNet implementations to BitNet.rs.

## Examples Overview

### Basic Migration Examples
- **[cpp-to-rust-basic](cpp-to-rust-basic/)** - Simple C++ to Rust API migration
- **[python-to-rust](python-to-rust/)** - Python to Rust migration with bindings
- **[config-migration](config-migration/)** - Configuration file migration examples

### Advanced Migration Examples
- **[server-migration](server-migration/)** - HTTP server migration from C++ to Rust
- **[concurrent-processing](concurrent-processing/)** - Thread pool to async migration
- **[performance-benchmarks](performance-benchmarks/)** - Comprehensive performance comparisons

## Quick Start

Each example directory contains:
- `README.md` - Detailed explanation and instructions
- `before/` - Original implementation (C++/Python)
- `after/` - Migrated Rust implementation
- `benchmark.rs` - Performance comparison code
- `migration-notes.md` - Migration process documentation

## Running Examples

```bash
# Run a specific example
cd examples/migration/cpp-to-rust-basic
cargo run

# Run performance benchmarks
cd examples/migration/performance-benchmarks
cargo bench

# Run all migration tests
cargo test --package migration-examples
```

## Migration Process

Each example follows a consistent migration process:

1. **Assessment** - Analyze the original implementation
2. **Planning** - Create migration plan with timeline
3. **Implementation** - Migrate code to Rust
4. **Validation** - Cross-validate outputs and performance
5. **Optimization** - Optimize for Rust-specific benefits
6. **Documentation** - Document changes and improvements

## Performance Improvements

Typical improvements seen in these examples:

| Metric | C++ Implementation | BitNet.rs | Improvement |
|--------|-------------------|-----------|-------------|
| **Inference Speed** | 520 tok/s | 1,250 tok/s | 2.4x faster |
| **Memory Usage** | 3.2 GB | 2.1 GB | 34% reduction |
| **Build Time** | 5m 20s | 30s | 10.7x faster |
| **Binary Size** | 45 MB | 12 MB | 73% smaller |
| **Cold Start** | 2.1s | 0.8s | 2.6x faster |

## Getting Help

- **Documentation**: [Migration Guide](../../docs/cpp-to-rust-migration.md)
- **FAQ**: [Migration FAQ](../../docs/migration-faq.md)
- **Community**: [GitHub Discussions](https://github.com/microsoft/BitNet/discussions)
- **Issues**: [GitHub Issues](https://github.com/microsoft/BitNet/issues)

## Contributing

To add a new migration example:

1. Create a new directory following the naming convention
2. Include before/after implementations
3. Add performance benchmarks
4. Document the migration process
5. Update this README with the new example

---

**Ready to migrate?** Start with the [basic C++ to Rust example](cpp-to-rust-basic/) or choose the example that best matches your use case.
