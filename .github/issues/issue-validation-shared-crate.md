# Refactor validation rules into shared `bitnet-validation` crate

**Labels:** `refactor`, `validation`, `architecture`

**Priority:** Medium (Code quality improvement)

**Depends on:** PR #452 (receipt verification)

## Summary

Migrate validation rules and helpers duplicated between CLI inspector and st-tools into a shared `bitnet-validation` crate to prevent policy drift.

## Problem

Currently, LayerNorm validation, projection checks, and correction policies are duplicated across:
- `bitnet-cli/src/inspect.rs`
- `bitnet-st-tools/src/validation/*.rs`
- Some logic in `bitnet-models`

This creates maintenance burden and risks policy drift.

## Acceptance Criteria

- [ ] Create new crate: `crates/bitnet-validation/`
- [ ] Migrate shared validation logic:
  - LayerNorm RMS validation with architecture-aware envelopes
  - Projection weight validation
  - Correction policy parsing and application
  - Strict mode handling
- [ ] Consumers:
  - `bitnet-cli` uses `bitnet-validation` for `inspect` command
  - `bitnet-st-tools` uses for GGUF validation
  - `bitnet-models` can optionally use for load-time validation
- [ ] No duplicated validation code
- [ ] Update documentation to reference single validation source

## API Design

```rust
// Proposed API for crates/bitnet-validation/src/lib.rs

pub struct ValidationConfig {
    pub mode: ValidationMode,
    pub strict: bool,
    pub policy_path: Option<PathBuf>,
    pub policy_key: Option<String>,
}

pub enum ValidationMode {
    None,
    Auto,
    Policy,
}

pub struct ValidationReport {
    pub passed: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Validate LayerNorm weights with architecture-aware envelopes
pub fn validate_layernorm_weights(
    weights: &[f32],
    arch: ModelArchitecture,
    config: &ValidationConfig,
) -> Result<ValidationReport>;

/// Validate projection weights
pub fn validate_projection_weights(
    weights: &[f32],
    config: &ValidationConfig,
) -> Result<ValidationReport>;

/// Parse and apply correction policy
pub fn apply_correction_policy(
    model: &mut Model,
    policy_path: &Path,
    policy_key: &str,
) -> Result<()>;
```

## Migration Plan

1. **Create crate structure:**
   ```
   crates/bitnet-validation/
   ├── Cargo.toml
   ├── src/
   │   ├── lib.rs
   │   ├── layernorm.rs
   │   ├── projection.rs
   │   ├── policy.rs
   │   └── config.rs
   ```

2. **Extract shared logic:**
   - Move LayerNorm RMS validation from `bitnet-cli` and `bitnet-st-tools`
   - Move projection validation logic
   - Move policy parsing (currently in both tools)

3. **Update consumers:**
   - `bitnet-cli/Cargo.toml` adds `bitnet-validation` dependency
   - `bitnet-st-tools/Cargo.toml` adds `bitnet-validation` dependency
   - Replace duplicated validation code with calls to shared crate

4. **Add tests:**
   - Unit tests for each validation function
   - Integration tests for policy application
   - Ensure backward compatibility

## Implementation Notes

- Keep `bitnet-validation` dependency-light (only `anyhow`, `serde`, `regex`)
- Use builder pattern for configs
- Consider making correction policy application explicit (not automatic)
- Maintain strict mode semantics across all consumers

## Example Usage

```rust
use bitnet_validation::{ValidationConfig, ValidationMode, validate_layernorm_weights};

let config = ValidationConfig {
    mode: ValidationMode::Auto,
    strict: std::env::var("BITNET_STRICT_MODE").is_ok(),
    policy_path: None,
    policy_key: None,
};

let report = validate_layernorm_weights(
    &ln_gamma_weights,
    ModelArchitecture::LLaMA,
    &config,
)?;

if !report.passed {
    for error in &report.errors {
        eprintln!("ERROR: {}", error);
    }
    if config.strict {
        std::process::exit(8);
    }
}
```

## Files to Create

- `crates/bitnet-validation/Cargo.toml`
- `crates/bitnet-validation/src/lib.rs`
- `crates/bitnet-validation/src/layernorm.rs`
- `crates/bitnet-validation/src/projection.rs`
- `crates/bitnet-validation/src/policy.rs`
- `crates/bitnet-validation/src/config.rs`
- `crates/bitnet-validation/tests/*.rs`

## Files to Modify

- `crates/bitnet-cli/Cargo.toml` - Add dependency
- `crates/bitnet-cli/src/inspect.rs` - Replace with shared validation
- `crates/bitnet-st-tools/Cargo.toml` - Add dependency
- `crates/bitnet-st-tools/src/validation/*.rs` - Replace with shared validation
- `docs/howto/validate-models.md` - Update to reference shared crate

## Related

- Related: PR #448 (validation MVP)
- Related: `docs/howto/validate-models.md`

## Estimated Effort

~2 days
