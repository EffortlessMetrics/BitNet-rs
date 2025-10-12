# Issue: Fix compilation failures across workspace crates

## Context

Multiple compilation failures have been identified across the BitNet.rs workspace that prevent successful builds with various feature flag combinations. The issues span four crates (bitnet-server, bitnet-inference, bitnet-tests, tests) and represent breaking changes in dependencies, missing type exports, and API drift. These failures block development workflows and CI pipeline health.

The most critical issue is the OpenTelemetry Prometheus version incompatibility in bitnet-server, which causes all builds with observability features to fail. This affects production deployments and monitoring capabilities. The remaining issues prevent `--all-features` builds from succeeding and block comprehensive testing.

**Affected Components:**
- **bitnet-server**: OpenTelemetry/Prometheus integration (observability layer)
- **bitnet-inference**: Full-engine test infrastructure (integration testing)
- **bitnet-tests**: Fixture management (test utilities)
- **tests**: Test harness configuration (shared test infrastructure)

**Priority Classification:**
- **P0 (Critical)**: bitnet-server OpenTelemetry incompatibility (blocks production monitoring)
- **P2 (High)**: bitnet-inference full-engine tests (blocks comprehensive feature testing)
- **P3 (Medium)**: bitnet-tests fixtures module and tests crate config drift (blocks specific test scenarios)

## User Story

As a BitNet.rs developer, I want all workspace crates to compile successfully with their respective feature combinations so that I can develop, test, and deploy the inference system without encountering compilation barriers that block CI/CD pipelines and production observability.

## Acceptance Criteria

### P0: OpenTelemetry Migration (bitnet-server)

**AC1**: Remove deprecated Prometheus exporter dependency from bitnet-server and migrate to OTLP exporter pattern
- Remove `opentelemetry_prometheus = "0.29.1"` dependency
- Add `opentelemetry_otlp` crate with metrics+traces support
- Command validation: `cargo check -p bitnet-server --no-default-features --features opentelemetry` succeeds

**AC2**: Implement OTLP-based metrics initialization with environment variable configuration
- Support `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable (default: `http://127.0.0.1:4317`)
- Implement unified metrics+traces exporter using OTLP protocol
- Preserve existing metric instrumentation points (request counts, latencies, resource usage)
- Command validation: `cargo build -p bitnet-server --no-default-features --features opentelemetry --release` succeeds

**AC3**: Remove all Prometheus-specific code paths and verify observability feature compiles cleanly
- Delete PrometheusExporter initialization code and related type conversions
- Update observability integration tests to use OTLP mock endpoints
- Command validation: `cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings` passes

### P2: Inference Full-Engine Tests (bitnet-inference)

**AC4**: Export production inference engine types required by full-engine feature tests
- Export or re-export `EngineConfig`, `ProductionInferenceEngine` in crate public API
- Add missing imports (`std::env`, `anyhow::Context`) to test modules
- Command validation: `cargo check -p bitnet-inference --all-features` succeeds

**AC5**: Ensure full-engine feature compiles with minimal stubs for WIP implementation
- Implement minimal stubs for incomplete full-engine functionality if needed
- Mark WIP tests with `#[ignore]` attribute to prevent false failures
- Preserve test structure for future implementation completion
- Command validation: `cargo test -p bitnet-inference --no-default-features --features full-engine --no-run` succeeds

### P3: Fixtures Module Declaration (bitnet-tests)

**AC6**: Declare fixtures module in bitnet-tests when fixtures feature is enabled
- Add `#[cfg(feature = "fixtures")] pub mod fixtures;` to `tests/lib.rs` or equivalent module root
- Ensure `FixtureManager` and related types are accessible to `debug_integration.rs`
- Command validation: `cargo test -p bitnet-tests --no-default-features --features fixtures --no-run` succeeds

### P3: Test Configuration API Drift (tests crate)

**AC7**: Update tests crate to match current TestConfig API shape
- Replace deprecated field access: `timeout_seconds` → `test_timeout`
- Replace deprecated field access: `fail_fast` → `reporting.fail_fast`
- Update all test files using old API patterns
- Command validation: `cargo test -p tests --no-run` succeeds

### CI Gate Improvements

**AC8**: Update CI pipeline to reflect compilation reality with feature-aware gates
- Maintain strict required gate: `cargo clippy --workspace --no-default-features --features cpu -- -D warnings`
- Maintain strict required gate: `cargo test --workspace --no-default-features --features cpu`
- Add separate exploratory job (allowed to fail): `cargo clippy --workspace --all-features -- -D warnings`
- Add separate exploratory job (allowed to fail): `cargo test --workspace --all-features`
- Promote exploratory jobs to required status after AC1-AC7 are completed
- Command validation: CI pipeline runs without required gate failures

## Technical Implementation Notes

### Affected Crates
- **bitnet-server**: Observability layer (OpenTelemetry integration)
- **bitnet-inference**: Autoregressive generation engine (full-engine feature)
- **bitnet-tests**: Shared test utilities (fixtures feature)
- **tests**: Integration test harness

### Pipeline Stages
- **Build**: Workspace-wide compilation with feature flag combinations
- **Testing**: Feature-gated test execution across CPU/GPU configurations
- **CI/CD**: Gate validation for quality assurance

### Performance Considerations
- OpenTelemetry OTLP exporter adds minimal runtime overhead (~1-2% for metrics collection)
- No impact on inference performance (observability is server-only concern)
- Test compilation time may increase slightly with `--all-features` builds

### Feature Flags
- `opentelemetry`: Server observability (metrics + traces via OTLP)
- `full-engine`: Production inference engine with complete API surface
- `fixtures`: Test fixture management utilities
- **Core flags preserved**: `cpu`, `gpu` for inference device selection

### Testing Strategy
1. **TDD Scaffolding**: Each AC maps to specific compilation validation command
   - AC1: `// AC1: Remove Prometheus dependency` tag in dependency manifest changes
   - AC2: `// AC2: OTLP metrics init` tag in observability module
   - AC3: `// AC3: Clean observability compilation` tag in clippy validation
   - AC4: `// AC4: Export engine types` tag in public API modules
   - AC5: `// AC5: Full-engine stubs` tag in conditional compilation blocks
   - AC6: `// AC6: Fixtures module declaration` tag in module structure
   - AC7: `// AC7: TestConfig API update` tag in test configuration code
   - AC8: `// AC8: CI feature gates` tag in workflow files

2. **Validation Commands**:
   ```bash
   # AC1-AC3: OpenTelemetry migration
   cargo check -p bitnet-server --no-default-features --features opentelemetry
   cargo build -p bitnet-server --no-default-features --features opentelemetry --release
   cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings

   # AC4-AC5: Inference engine types
   cargo check -p bitnet-inference --all-features
   cargo test -p bitnet-inference --no-default-features --features full-engine --no-run

   # AC6: Fixtures module
   cargo test -p bitnet-tests --no-default-features --features fixtures --no-run

   # AC7: Test config API
   cargo test -p tests --no-run

   # AC8: CI exploratory gates (post-fixes)
   cargo clippy --workspace --all-features -- -D warnings
   cargo test --workspace --all-features
   ```

3. **Cross-Validation**: Not required (no inference algorithm changes)

4. **Feature Smoke Testing**:
   - Validate `--no-default-features --features cpu` (baseline requirement)
   - Validate `--no-default-features --features opentelemetry` (server observability)
   - Validate `--all-features` after all fixes complete (comprehensive build)

### GGUF Compatibility
- No impact (compilation fixes only, no model format changes)

### Error Handling
- Preserve existing `anyhow::Result<T>` patterns in modified code
- Add proper error context for OTLP endpoint connection failures
- Maintain backward compatibility for observability feature flag behavior

### Time & Risk Estimates
- **AC1-AC3 (OTel migration)**: ~0.5–1.5 days; medium risk but isolated to bitnet-server observability
- **AC4-AC5 (Inference stubs/exports)**: ~0.5 days; low risk if behavior kept `#[ignore]` for WIP
- **AC6-AC7 (Fixtures + config drift)**: ~0.5 days combined; trivial risk (mechanical changes)
- **AC8 (CI changes)**: ~0.25 days; low risk (workflow configuration only)

### Priority Order
1. **P0 (AC1-AC3)**: Fix bitnet-server OpenTelemetry incompatibility (blocks production monitoring)
2. **P2 (AC4-AC5)**: Fix bitnet-inference full-engine tests (blocks comprehensive feature testing)
3. **P3 (AC6-AC7)**: Fix bitnet-tests fixtures and tests config drift (blocks specific test scenarios)
4. **CI (AC8)**: Update CI gates to reflect compilation reality (enables proactive quality assurance)

### BitNet.rs-Specific Patterns
- **Feature-Gated Compilation**: All fixes must respect `--no-default-features` baseline requirement
- **Workspace Structure**: Changes span multiple crates requiring coordinated compilation validation
- **Observability Isolation**: OpenTelemetry changes isolated to bitnet-server (no inference impact)
- **Test Infrastructure**: Fixture and config changes affect shared test utilities across workspace
- **CI Gates**: Maintain strict CPU feature gate as required; exploratory all-features gate allowed to fail until fixes complete
