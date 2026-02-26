# Repository Restructure Requirements

## Introduction

This specification defines the requirements for restructuring the BitNet repository to clearly establish BitNet-rs as the primary implementation, with the original C++ BitNet implementation serving as a legacy benchmark and cross-validation target.

## Current State Analysis

The repository currently has a confusing mixed structure:
- Root level contains both Rust (Cargo.toml) and C++ (CMakeLists.txt) build systems
- C++ source files in `/src/` and `/include/`
- Python scripts for C++ build process in root
- Rust implementation properly organized in `/crates/`
- Mixed documentation and examples

This creates confusion about:
- Which implementation is primary
- How to build and use each implementation
- What the project's main focus is
- How to contribute to the right codebase

## Requirements

### Requirement 1: Clear Primary Implementation

**User Story:** As a developer discovering this project, I want to immediately understand that BitNet-rs is the primary, production-ready implementation.

#### Acceptance Criteria

1. WHEN a developer visits the repository root THEN they SHALL see Rust-focused documentation and build instructions
2. WHEN a developer examines the root directory THEN they SHALL find only Rust-related build files (Cargo.toml, build.rs)
3. WHEN a developer reads the README THEN they SHALL understand BitNet-rs is the main implementation with superior performance and safety

### Requirement 2: External Legacy Reference

**User Story:** As a maintainer, I want to test against the original BitNet.cpp without maintaining or hosting C++ code in our repository.

#### Acceptance Criteria

1. WHEN the repository is restructured THEN all C++ source code SHALL be removed from the repository
2. WHEN developers work on Rust code THEN they SHALL NOT encounter any C++ build systems or dependencies
3. WHEN cross-validation is needed THEN the system SHALL fetch and build the original Microsoft BitNet.cpp on-demand
4. WHEN CI/CD runs THEN it SHALL primarily focus on Rust builds with optional external legacy comparison

### Requirement 3: Cross-Validation Infrastructure

**User Story:** As a quality engineer, I want to validate that BitNet-rs maintains compatibility and performance parity with the original implementation.

#### Acceptance Criteria

1. WHEN cross-validation tests run THEN they SHALL compare outputs between Rust and legacy C++ implementations
2. WHEN performance benchmarks execute THEN they SHALL measure both implementations and report comparisons
3. WHEN numerical accuracy tests run THEN they SHALL verify token-level output matching within 1e-6 tolerance
4. WHEN integration tests execute THEN they SHALL validate API compatibility between implementations

### Requirement 4: Clean Build System Separation

**User Story:** As a developer, I want clear, separate build processes for each implementation without conflicts.

#### Acceptance Criteria

1. WHEN building the Rust implementation THEN developers SHALL use standard `cargo build` commands
2. WHEN building the legacy implementation THEN developers SHALL use commands within the `/legacy/` directory
3. WHEN both implementations are built THEN they SHALL NOT interfere with each other's artifacts
4. WHEN CI builds run THEN they SHALL use appropriate build systems for each implementation

### Requirement 5: Documentation Clarity

**User Story:** As a user, I want clear documentation that explains the relationship between implementations and guides me to the right one.

#### Acceptance Criteria

1. WHEN reading the main README THEN users SHALL understand BitNet-rs is the recommended implementation
2. WHEN accessing legacy documentation THEN users SHALL find clear migration guidance to BitNet-rs
3. WHEN following quick start guides THEN users SHALL be directed to Rust-based workflows
4. WHEN seeking performance comparisons THEN users SHALL find benchmarks showing Rust advantages

### Requirement 6: Migration Path Preservation

**User Story:** As an existing user of BitNet.cpp, I want a clear migration path to BitNet-rs with compatibility guarantees.

#### Acceptance Criteria

1. WHEN migrating from C++ to Rust THEN users SHALL find detailed migration documentation
2. WHEN using existing models THEN they SHALL work with both implementations during transition
3. WHEN comparing APIs THEN users SHALL find compatibility matrices and equivalent functions
4. WHEN performance testing THEN users SHALL be able to validate improvements in their specific use cases

### Requirement 7: Development Workflow Optimization

**User Story:** As a contributor, I want development workflows optimized for Rust development with optional legacy testing.

#### Acceptance Criteria

1. WHEN setting up development environment THEN contributors SHALL follow Rust-standard practices
2. WHEN running tests THEN the default test suite SHALL focus on Rust implementation
3. WHEN submitting PRs THEN CI SHALL primarily validate Rust code quality and functionality
4. WHEN legacy comparison is needed THEN it SHALL be available as an optional, clearly marked process

### Requirement 8: Deployment and Distribution Focus

**User Story:** As a DevOps engineer, I want deployment and distribution focused on the Rust implementation.

#### Acceptance Criteria

1. WHEN building Docker images THEN they SHALL primarily contain the Rust implementation
2. WHEN publishing packages THEN Rust crates SHALL be the primary distribution method
3. WHEN deploying to production THEN documentation SHALL recommend the Rust implementation
4. WHEN legacy support is needed THEN it SHALL be available but clearly marked as legacy/compatibility mode
