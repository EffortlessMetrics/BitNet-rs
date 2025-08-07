# Repository Restructure Implementation Plan

## Overview

This document outlines the detailed implementation tasks for restructuring the BitNet repository to establish BitNet.rs as the primary implementation with the C++ implementation as a legacy benchmark target.

## Task Breakdown

### Phase 1: Repository Cleanup

- [x] 1.1 Remove all C++ source code from repository
  - Remove `src/` directory and all C++ source files
  - Remove `include/` directory and all C++ headers
  - Remove `CMakeLists.txt` and C++ build configuration
  - Remove `3rdparty/` directory and C++ dependencies
  - _Requirements: 2.1, 2.2_

- [x] 1.2 Remove C++ related Python scripts
  - Remove `setup_env.py`, `run_inference.py`, `run_inference_server.py`
  - Remove `requirements.txt` (C++ Python dependencies)
  - Remove `gpu/`, `utils/`, `preset_kernels/` directories
  - _Requirements: 2.1, 2.2_

- [x] 1.3 Create CI helper scripts directory
  - Create `/ci/` directory for build automation
  - Set up structure for external dependency management
  - Add `.gitignore` entries for cached external builds
  - Create `.cargo/config.toml` to globally disable crossval feature by default
  - _Requirements: 2.3, 4.1_

- [x] 1.4 Create cross-validation crate structure
  - Create `/crossval/` directory as separate Rust crate
  - Create subdirectories: `src/`, `tests/`, `benches/`, `fixtures/`
  - Set up `crossval/Cargo.toml` with feature gates
  - _Requirements: 3.1, 3.2_

- [x] 1.5 Create patches directory (initially empty)
  - Create `/patches/` directory for minimal patches if needed
  - Add README explaining patch policy (prefer upstream fixes)
  - Set up patch application automation
  - _Requirements: 2.4, 3.1_

### Phase 2: External Dependency System

- [x] 2.1 Create BitNet.cpp fetch script
  - Implement `ci/fetch_bitnet_cpp.sh` with version pinning
  - Add checksum verification for downloaded source
  - Set up caching in `$HOME/.cache/bitnet_cpp/`
  - _Requirements: 2.3, 4.1_

- [x] 2.2 Create patch application system
  - Implement `ci/apply_patches.sh` for minimal patches
  - Add patch validation and ordering system
  - Create patch creation and maintenance documentation
  - _Requirements: 2.4, 3.1_

- [x] 2.3 Update root Cargo.toml workspace
  - Remove any C++ build dependencies from root Cargo.toml
  - Add `crossval` crate to workspace members
  - Add `bitnet-sys` crate for FFI bindings
  - _Requirements: 1.2, 4.1_

- [x] 2.4 Create bitnet-sys FFI crate
  - Implement `crates/bitnet-sys/` with feature gates
  - Add bindgen integration for C++ headers with clang detection
  - Set up conditional compilation for crossval feature
  - Add helpful build.rs error messages when clang is missing
  - _Requirements: 3.1, 3.2_

- [x] 2.5 Create version management system
  - Implement `ci/bump_bitnet_tag.sh` for version updates
  - Add automated dependency update checking
  - Create documentation for version update process
  - _Requirements: 2.3, 4.2_

### Phase 3: Documentation Restructure

- [x] 3.1 Rewrite main README.md
  - Focus on Rust implementation as primary
  - Add clear performance and safety advantages
  - Include quick start guide for Rust
  - Add section about legacy implementation location
  - _Requirements: 1.1, 1.3, 5.1_

- [x] 3.2 Create legacy README
  - Create `legacy/bitnet.cpp/README.md`
  - Mark clearly as legacy implementation
  - Provide migration guidance to Rust
  - Include build and usage instructions for legacy
  - _Requirements: 2.3, 5.2, 6.1_

- [x] 3.3 Update documentation in docs/
  - Rewrite all documentation to focus on Rust
  - Update API references and examples
  - Add migration guides from C++ to Rust
  - Update troubleshooting guides
  - _Requirements: 5.1, 5.3, 6.2_

- [x] 3.4 Create tooling documentation
  - Create `tools/crossval/README.md`
  - Document comparison methodology
  - Provide usage instructions for benchmarking
  - Include interpretation guides for results
  - _Requirements: 3.3, 5.4_

- [x] 3.5 Update CHANGELOG and release notes
  - Create new CHANGELOG focused on Rust releases
  - Archive C++ changelog in legacy directory
  - Document the restructure as a major milestone
  - _Requirements: 5.1, 6.3_

### Phase 4: Cross-Validation Framework

- [x] 4.1 Implement token equivalence tests
  - Create `crossval/tests/token_equivalence.rs`
  - Implement exact token matching for small models
  - Add tolerance-based comparison for floating point values
  - Include multiple model format support
  - _Requirements: 3.1, 3.3_

- [x] 4.2 Create performance benchmarks
  - Implement `crossval/benches/performance.rs` using Criterion
  - Add throughput and latency measurements
  - Create 5% performance regression guards
  - Generate comparison reports and charts
  - _Requirements: 3.2, 3.4_

- [x] 4.3 Set up test fixtures
  - Create small test models (~20KB) in `crossval/fixtures/`
  - Add deterministic fixture generator (`cargo xtask gen-fixtures`)
  - Include test prompt datasets for various scenarios
  - Set up baseline performance numbers in committed JSON files
  - _Requirements: 3.1, 3.2_

- [x] 4.4 Implement FFI integration
  - Create safe Rust wrappers around C++ functions
  - Add error handling and memory management
  - Implement model loading and inference calls
  - Set up proper resource cleanup
  - _Requirements: 3.1, 3.2_

- [x] 4.5 Add feature gate integration
  - Implement conditional compilation for crossval feature
  - Ensure zero overhead when feature is disabled
  - Add documentation for feature usage
  - Create developer setup instructions
  - _Requirements: 3.4, 7.1_

### Phase 5: CI/CD and Workflow Updates

- [x] 5.1 Update primary CI workflow
  - Modify `.github/workflows/` to focus on Rust
  - Ensure fast, Rust-focused CI pipeline
  - Add comprehensive Rust testing and quality checks
  - _Requirements: 1.2, 7.1, 7.3_

- [x] 5.2 Create nightly cross-validation workflow
  - Add nightly cron job for upstream API change detection
  - Create optional `nightly-crossval` feature for long-running benchmarks
  - Set up performance baseline tracking and drift detection
  - Configure GitHub step outputs for tag/SHA visibility
  - _Requirements: 2.4, 7.3_

- [x] 5.3 Update release workflows
  - Focus release process on Rust crates and binaries
  - Add automated crate publishing to crates.io
  - Create GitHub releases for Rust binaries
  - _Requirements: 8.2, 8.3_

- [x] 5.4 Set up development environment scripts
  - Create `scripts/dev-setup.sh` for Rust development
  - Add `scripts/dev-crossval.sh` one-liner for cross-validation setup
  - Include environment variable configuration and fetch automation
  - Add patch policy enforcement in CI (fail if patches/ non-empty without upstream issue)
  - _Requirements: 7.1, 7.2_

### Phase 6: Migration and Compatibility

- [x] 6.1 Create migration documentation
  - Write comprehensive C++ to Rust migration guide
  - Include API compatibility matrices
  - Add "why we don't vendor C++" rationale in FAQ
  - Document upstream tag, license, and attribution requirements
  - Add clear split: "Rust quick-start" vs "Legacy comparison guide"
  - _Requirements: 6.1, 6.3_

- [x] 6.2 Implement migration tools
  - Create scripts to help migrate C++ configurations to Rust
  - Add model format compatibility validation
  - Include automated migration testing
  - _Requirements: 6.2, 6.4_

- [x] 6.3 Set up compatibility testing
  - Create tests that validate API compatibility
  - Add model format compatibility tests
  - Include performance comparison validation
  - _Requirements: 6.2, 6.4_

- [-] 6.4 Create migration examples
  - Add example projects showing migration patterns
  - Include before/after code comparisons
  - Add performance benchmarking examples
  - _Requirements: 6.3, 6.4_

### Phase 7: Deployment and Distribution Updates

- [ ] 7.1 Update Docker configurations
  - Modify Dockerfiles to focus on Rust implementation
  - Create multi-stage builds for optimal Rust containers
  - Add optional legacy container builds
  - _Requirements: 8.1, 8.3_

- [ ] 7.2 Update Kubernetes and Helm configurations
  - Modify K8s deployments for Rust services
  - Update Helm charts to deploy Rust implementation
  - Add legacy deployment options if needed
  - _Requirements: 8.1, 8.3_

- [ ] 7.3 Update monitoring and observability
  - Configure monitoring for Rust services
  - Add performance comparison dashboards
  - Set up alerts for cross-validation failures
  - _Requirements: 8.3_

- [ ] 7.4 Update package distribution
  - Ensure crates.io publishing is primary distribution
  - Add binary releases for major platforms
  - Create installation scripts focused on Rust
  - _Requirements: 8.2, 8.3_

### Phase 8: Polish and Nice-to-Haves

- [ ] 8.1 Add binary caching optimization
  - Set up GitHub Container Registry for pre-built bitnet_cpp libraries
  - Cache builds keyed by tag+SHA for faster CI
  - Reduce CI time from ~7min to <1min while maintaining reproducibility
  - _Requirements: Performance optimization_

- [ ] 8.2 Implement patch policy automation
  - Add CI check that fails if patches/ is non-empty without upstream issue link
  - Create automated issue creation when patches are added
  - Set up patch lifecycle tracking and cleanup reminders
  - _Requirements: 2.4, Maintenance_

- [ ] 8.3 Add performance baseline tracking
  - Commit baseline performance numbers to crossval/baselines.json
  - Upload fresh benchmark results as CI artifacts
  - Create performance drift detection and alerting
  - _Requirements: 3.2, 3.4_

- [ ] 8.4 Create developer convenience tools
  - Add `cargo xtask gen-fixtures` for deterministic test model generation
  - Create `./scripts/dev-crossval.sh` one-liner for easy setup
  - Add IDE configuration to prevent accidental crossval feature activation
  - _Requirements: 7.1, 7.2_

### Phase 9: Validation and Testing

- [ ] 9.1 Comprehensive build testing
  - Test Rust builds on all supported platforms
  - Validate legacy builds in isolated environment
  - Ensure no build system conflicts
  - _Requirements: 4.2, 4.3_

- [ ] 9.2 Cross-validation framework testing
  - Test comparison scripts with real models
  - Validate numerical accuracy detection
  - Test performance benchmarking accuracy
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 9.3 Documentation validation
  - Test all documentation examples and code snippets
  - Validate migration guides with real scenarios
  - Ensure all links and references are correct
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 9.4 End-to-end workflow testing
  - Test complete development workflows
  - Validate CI/CD pipelines
  - Test deployment and distribution processes
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 9.5 Performance validation
  - Run comprehensive performance comparisons
  - Validate that Rust implementation meets performance claims
  - Test cross-validation accuracy and reliability
  - _Requirements: 3.2, 3.4, 6.4_

## Success Criteria

### Technical Success
- [ ] All Rust builds pass on supported platforms
- [ ] Legacy builds work in isolated environment
- [ ] Cross-validation framework produces accurate comparisons
- [ ] No build system conflicts between implementations
- [ ] All documentation examples work correctly

### User Experience Success
- [ ] New users can quickly get started with Rust implementation
- [ ] Existing C++ users have clear migration path
- [ ] Documentation clearly establishes Rust as primary implementation
- [ ] Cross-validation results demonstrate Rust advantages

### Operational Success
- [ ] CI/CD focuses on Rust with optional legacy testing
- [ ] Deployment and distribution prioritize Rust implementation
- [ ] Monitoring and observability work for both implementations
- [ ] Performance comparisons validate Rust improvements

## Risk Mitigation

### File Movement Risks
- **Risk:** Loss of git history for moved files
- **Mitigation:** Use `git mv` commands and preserve history links
- **Validation:** Verify git history preservation after moves

### Build System Conflicts
- **Risk:** C++ and Rust builds interfering with each other
- **Mitigation:** Complete isolation of build systems and artifacts
- **Validation:** Test both builds simultaneously without conflicts

### Documentation Accuracy
- **Risk:** Outdated or incorrect migration documentation
- **Mitigation:** Test all documentation examples with real scenarios
- **Validation:** Automated testing of documentation code snippets

### Performance Regression
- **Risk:** Restructure causing performance degradation
- **Mitigation:** Comprehensive performance testing before and after
- **Validation:** Cross-validation framework confirms performance parity

### User Confusion
- **Risk:** Users confused about which implementation to use
- **Mitigation:** Clear, prominent documentation about primary implementation
- **Validation:** User feedback and documentation review