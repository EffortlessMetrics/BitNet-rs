//! WGSL compute shader sources for `BitNet` inference kernels.
//!
//! This crate provides embedded WGSL shader sources for core neural network
//! operations targeting WebGPU / wgpu. Shaders are exposed as `&'static str`
//! constants for runtime compilation by the wgpu pipeline.
//!
//! # Shader categories
//!
//! - **matmul** — matrix multiplication (naive, tiled, matrix-vector)
//! - **elementwise** — add, mul, scale, `ReLU`, GELU, `SiLU`
//! - **softmax** — row-wise softmax and log-softmax with shared-memory reduction
//! - **`layer_norm`** — `LayerNorm` (gamma+beta) and RMS normalization (LLaMA-style)
//! - **rope** — Rotary Position Embeddings

pub mod elementwise;
pub mod layer_norm;
pub mod matmul;
pub mod rope;
pub mod softmax;

/// Returns all shader sources as `(name, source)` pairs for bulk validation.
pub fn all_shader_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        // matmul
        ("matmul_naive", matmul::MATMUL_NAIVE_SRC),
        ("matmul_tiled", matmul::MATMUL_TILED_SRC),
        ("matmul_vec", matmul::MATMUL_VEC_SRC),
        // elementwise
        ("add", elementwise::ADD_SRC),
        ("mul", elementwise::MUL_SRC),
        ("scale", elementwise::SCALE_SRC),
        ("relu", elementwise::RELU_SRC),
        ("gelu", elementwise::GELU_SRC),
        ("silu", elementwise::SILU_SRC),
        // softmax
        ("softmax", softmax::SOFTMAX_SRC),
        ("log_softmax", softmax::LOG_SOFTMAX_SRC),
        // layer_norm
        ("layer_norm", layer_norm::LAYER_NORM_SRC),
        ("rms_norm", layer_norm::RMS_NORM_SRC),
        // rope
        ("rope", rope::ROPE_SRC),
    ]
}

#[cfg(test)]
mod tests {
    use naga::front::wgsl;

    fn validate_wgsl(source: &str) -> Result<(), String> {
        let module = wgsl::parse_str(source).map_err(|e| format!("{e}"))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).map_err(|e| format!("{e}"))?;
        Ok(())
    }

    // ── matmul ──────────────────────────────────────────────────

    #[test]
    fn test_matmul_naive_valid() {
        validate_wgsl(super::matmul::MATMUL_NAIVE_SRC).unwrap();
    }

    #[test]
    fn test_matmul_tiled_valid() {
        validate_wgsl(super::matmul::MATMUL_TILED_SRC).unwrap();
    }

    #[test]
    fn test_matmul_vec_valid() {
        validate_wgsl(super::matmul::MATMUL_VEC_SRC).unwrap();
    }

    // ── elementwise ─────────────────────────────────────────────

    #[test]
    fn test_add_valid() {
        validate_wgsl(super::elementwise::ADD_SRC).unwrap();
    }

    #[test]
    fn test_mul_valid() {
        validate_wgsl(super::elementwise::MUL_SRC).unwrap();
    }

    #[test]
    fn test_scale_valid() {
        validate_wgsl(super::elementwise::SCALE_SRC).unwrap();
    }

    #[test]
    fn test_relu_valid() {
        validate_wgsl(super::elementwise::RELU_SRC).unwrap();
    }

    #[test]
    fn test_gelu_valid() {
        validate_wgsl(super::elementwise::GELU_SRC).unwrap();
    }

    #[test]
    fn test_silu_valid() {
        validate_wgsl(super::elementwise::SILU_SRC).unwrap();
    }

    // ── softmax ─────────────────────────────────────────────────

    #[test]
    fn test_softmax_valid() {
        validate_wgsl(super::softmax::SOFTMAX_SRC).unwrap();
    }

    #[test]
    fn test_log_softmax_valid() {
        validate_wgsl(super::softmax::LOG_SOFTMAX_SRC).unwrap();
    }

    // ── layer_norm ──────────────────────────────────────────────

    #[test]
    fn test_layer_norm_valid() {
        validate_wgsl(super::layer_norm::LAYER_NORM_SRC).unwrap();
    }

    #[test]
    fn test_rms_norm_valid() {
        validate_wgsl(super::layer_norm::RMS_NORM_SRC).unwrap();
    }

    // ── rope ────────────────────────────────────────────────────

    #[test]
    fn test_rope_valid() {
        validate_wgsl(super::rope::ROPE_SRC).unwrap();
    }

    // ── bulk ────────────────────────────────────────────────────

    #[test]
    fn test_all_shader_sources_validate() {
        let sources = super::all_shader_sources();
        assert_eq!(sources.len(), 14, "expected 14 shader sources");
        for (name, source) in &sources {
            validate_wgsl(source).unwrap_or_else(|e| {
                panic!("shader '{name}' failed validation: {e}");
            });
        }
    }

    #[test]
    fn test_all_shader_sources_non_empty() {
        for (name, source) in super::all_shader_sources() {
            assert!(!source.trim().is_empty(), "shader '{name}' is empty");
        }
    }

    #[test]
    fn test_shader_names_unique() {
        let sources = super::all_shader_sources();
        let names: Vec<_> = sources.iter().map(|(n, _)| *n).collect();
        let mut deduped = names.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "duplicate shader names found");
    }
}
