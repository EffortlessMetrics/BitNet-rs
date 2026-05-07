use std::fmt;

use crate::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};

pub const MACHINE_ID: &str = "apple-m4-mac-mini";
pub const ARTIFACT_KIND: &str = "smoke";
pub const PARITY_ARTIFACT_KIND: &str = "parity";
pub const PHASE_CONTRIBUTION_ARTIFACT_KIND: &str = "phase_contribution";
pub const SUBGRAPH_ARTIFACT_KIND: &str = "subgraph";
pub const REQUESTED_BACKEND: &str = "apple-m4-metal";
pub const SELECTED_BACKEND: &str = "apple-m4-metal";
pub const REFERENCE_BACKEND: &str = "apple-m4-cpu-neon";
pub const RUNTIME_API: &str = "metal";
pub const TINY_METAL_ADD_SMOKE_KERNEL_ID: &str = "tiny_metal_add_smoke";
pub const TINY_METAL_ADD_PARITY_KERNEL_ID: &str = "tiny_metal_add_parity";
pub const I2S_METAL_PARITY_KERNEL_ID: &str = "tiny_metal_i2s_parity";
pub const I2S_METAL_PREFILL_CONTRIBUTION_KERNEL_ID: &str = "tiny_metal_i2s_prefill_contribution";
pub const I2S_METAL_PROJECTION_RESIDUAL_KERNEL_ID: &str = "tiny_metal_i2s_projection_residual";
pub const I2S_PROJECTION_RESIDUAL_GRAPH_ID: &str = "tiny_i2s_projection_residual_subgraph";
pub const I2S_KERNEL_FAMILY: &str = "i2_s";
pub const I2S_EXECUTION_PHASE: &str = "parity";
pub const I2S_PREFILL_EXECUTION_PHASE: &str = "prefill";
pub const I2S_PREFILL_PHASE_SCOPE: &str = "prefill_projection_fixture";
pub const I2S_PREFILL_KV_CACHE_BEHAVIOR: &str = "not_exercised";
pub const I2S_PROJECTION_RESIDUAL_EXECUTION_PHASE: &str = "parity";
pub const I2S_PROJECTION_RESIDUAL_PHASE_SCOPE: &str = "projection_residual_subgraph";
pub const I2S_PROJECTION_RESIDUAL_OPS: [&str; 2] = ["packed_i2_s_matmul", "residual_add"];
pub const I2S_LAYOUT_SOURCE: &str = "fixture_packed_i2_s";
pub const I2S_TRANSPORT_LAYOUT: &str = "u32_le_words_from_i2s_bytes";
pub const I2S_PARITY_M: usize = 1;
pub const I2S_PARITY_N: usize = 4;
pub const I2S_PARITY_K: usize = 32;
pub const I2S_PARITY_BLOCK_SIZE: usize = 32;
pub const I2S_PREFILL_TOKENS: usize = 2;
pub const SMOKE_ELEMENT_COUNT: usize = 64;
pub const SMOKE_WORKGROUP_SIZE: u32 = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct TinyMetalAddSmokeReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub kernel_id: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

impl TinyMetalAddSmokeReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        element_count: usize,
        comparison: SmokeComparison,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            kernel_id: TINY_METAL_ADD_SMOKE_KERNEL_ID,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            element_count,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TinyMetalAddParityReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub kernel_id: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalParityReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub kernel_id: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalPrefillContributionReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub kernel_id: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub kv_cache_behavior: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub prefill_tokens: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalProjectionResidualReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub reference_backend: &'static str,
    pub target_backend: &'static str,
    pub graph_id: &'static str,
    pub kernel_id: &'static str,
    pub kernel_family: &'static str,
    pub execution_phase: &'static str,
    pub phase_scope: &'static str,
    pub layout_source: &'static str,
    pub transport_layout: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub tokens: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

impl TinyMetalAddParityReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        element_count: usize,
        comparison: SmokeComparison,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PARITY_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            kernel_id: TINY_METAL_ADD_PARITY_KERNEL_ID,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            element_count,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl I2sMetalParityReceipt {
    pub fn passed(artifact_path: impl Into<String>, comparison: SmokeComparison) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PARITY_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            kernel_id: I2S_METAL_PARITY_KERNEL_ID,
            kernel_family: I2S_KERNEL_FAMILY,
            execution_phase: I2S_EXECUTION_PHASE,
            layout_source: I2S_LAYOUT_SOURCE,
            transport_layout: I2S_TRANSPORT_LAYOUT,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            m: I2S_PARITY_M,
            n: I2S_PARITY_N,
            k: I2S_PARITY_K,
            block_size: I2S_PARITY_BLOCK_SIZE,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl I2sMetalPrefillContributionReceipt {
    pub fn passed(artifact_path: impl Into<String>, comparison: SmokeComparison) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: PHASE_CONTRIBUTION_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            kernel_id: I2S_METAL_PREFILL_CONTRIBUTION_KERNEL_ID,
            kernel_family: I2S_KERNEL_FAMILY,
            execution_phase: I2S_PREFILL_EXECUTION_PHASE,
            phase_scope: I2S_PREFILL_PHASE_SCOPE,
            layout_source: I2S_LAYOUT_SOURCE,
            transport_layout: I2S_TRANSPORT_LAYOUT,
            kv_cache_behavior: I2S_PREFILL_KV_CACHE_BEHAVIOR,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            prefill_tokens: I2S_PREFILL_TOKENS,
            n: I2S_PARITY_N,
            k: I2S_PARITY_K,
            block_size: I2S_PARITY_BLOCK_SIZE,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

impl I2sMetalProjectionResidualReceipt {
    pub fn passed(artifact_path: impl Into<String>, comparison: SmokeComparison) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: SUBGRAPH_ARTIFACT_KIND,
            requested_backend: REQUESTED_BACKEND,
            selected_backend: SELECTED_BACKEND,
            runtime_api: RUNTIME_API,
            reference_backend: REFERENCE_BACKEND,
            target_backend: SELECTED_BACKEND,
            graph_id: I2S_PROJECTION_RESIDUAL_GRAPH_ID,
            kernel_id: I2S_METAL_PROJECTION_RESIDUAL_KERNEL_ID,
            kernel_family: I2S_KERNEL_FAMILY,
            execution_phase: I2S_PROJECTION_RESIDUAL_EXECUTION_PHASE,
            phase_scope: I2S_PROJECTION_RESIDUAL_PHASE_SCOPE,
            layout_source: I2S_LAYOUT_SOURCE,
            transport_layout: I2S_TRANSPORT_LAYOUT,
            fallback_used: false,
            result: "pass",
            artifact_path: artifact_path.into(),
            tokens: I2S_PREFILL_TOKENS,
            n: I2S_PARITY_N,
            k: I2S_PARITY_K,
            block_size: I2S_PARITY_BLOCK_SIZE,
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalParityFixture {
    pub activations: Vec<f32>,
    pub weights_packed: Vec<u8>,
    pub weights_packed_words: Vec<u32>,
    pub scales: Vec<f32>,
    pub expected: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct I2sMetalProjectionResidualFixture {
    pub base: I2sMetalParityFixture,
    pub residual: Vec<f32>,
    pub expected: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmokeComparison {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TinyMetalSmokeError {
    EmptyInput,
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteOutput { index: usize, value: f32 },
    OutputMismatch { index: usize, expected: f32, actual: f32, tolerance: f32 },
}

impl fmt::Display for TinyMetalSmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "tiny Metal smoke input must not be empty"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "tiny Metal smoke length mismatch: expected {expected}, actual {actual}")
            }
            Self::NonFiniteOutput { index, value } => {
                write!(f, "tiny Metal smoke output at index {index} is non-finite: {value}")
            }
            Self::OutputMismatch { index, expected, actual, tolerance } => write!(
                f,
                "tiny Metal smoke output mismatch at index {index}: expected {expected}, actual {actual}, tolerance {tolerance}"
            ),
        }
    }
}

impl std::error::Error for TinyMetalSmokeError {}

pub fn metal_smoke_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-smoke.json")
}

pub fn metal_parity_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-parity.json")
}

pub fn metal_i2s_parity_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-i2s-parity.json")
}

pub fn metal_i2s_prefill_contribution_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-i2s-prefill-contribution.json")
}

pub fn metal_i2s_projection_residual_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/metal-i2s-projection-residual.json")
}

pub fn tiny_add_inputs() -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..SMOKE_ELEMENT_COUNT).map(|i| i as f32).collect();
    let rhs = (0..SMOKE_ELEMENT_COUNT).map(|i| (i as f32) * 2.0).collect();
    (lhs, rhs)
}

pub fn expected_tiny_add(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, TinyMetalSmokeError> {
    if lhs.is_empty() {
        return Err(TinyMetalSmokeError::EmptyInput);
    }
    if lhs.len() != rhs.len() {
        return Err(TinyMetalSmokeError::LengthMismatch { expected: lhs.len(), actual: rhs.len() });
    }

    Ok(lhs.iter().zip(rhs.iter()).map(|(left, right)| left + right).collect())
}

pub fn i2s_metal_parity_fixture() -> I2sMetalParityFixture {
    i2s_metal_fixture(I2S_PARITY_M)
}

pub fn i2s_metal_prefill_fixture() -> I2sMetalParityFixture {
    i2s_metal_fixture(I2S_PREFILL_TOKENS)
}

pub fn i2s_metal_projection_residual_fixture() -> I2sMetalProjectionResidualFixture {
    let base = i2s_metal_fixture(I2S_PREFILL_TOKENS);
    let residual = (0..base.expected.len())
        .map(|index| {
            let row = index / base.n;
            let col = index % base.n;
            ((row as i32 * 3 + col as i32 * 2) % 7 - 3) as f32 * 0.125
        })
        .collect::<Vec<_>>();
    let expected = base
        .expected
        .iter()
        .zip(residual.iter())
        .map(|(value, residual)| value + residual)
        .collect();

    I2sMetalProjectionResidualFixture { base, residual, expected }
}

fn i2s_metal_fixture(m: usize) -> I2sMetalParityFixture {
    let activations = (0..m * I2S_PARITY_K)
        .map(|index| {
            let row = index / I2S_PARITY_K;
            ((index as i32 + row as i32 * 2) % 9 - 4) as f32 * 0.25
        })
        .collect::<Vec<_>>();
    let scales = vec![0.25, 0.5, 0.75, 1.0];

    let mut weights_packed = Vec::with_capacity((I2S_PARITY_K / 4) * I2S_PARITY_N);
    for col in 0..I2S_PARITY_N {
        for chunk_start in (0..I2S_PARITY_K).step_by(4) {
            let vals = std::array::from_fn(|offset| {
                let selector = (chunk_start + offset + col * 3) % 5;
                match selector {
                    0 | 3 => -1,
                    1 => 0,
                    _ => 1,
                }
            });
            weights_packed.push(pack_i2s(vals));
        }
    }

    let weights_packed_words = pack_i2s_bytes_to_u32_words(&weights_packed);
    let mut expected = vec![0.0; m * I2S_PARITY_N];
    i2s_matmul_f32(
        &activations,
        &weights_packed,
        &scales,
        &mut expected,
        m,
        I2S_PARITY_N,
        I2S_PARITY_K,
        I2S_PARITY_BLOCK_SIZE,
    )
    .expect("deterministic M4 I2_S parity fixture is valid");

    I2sMetalParityFixture {
        activations,
        weights_packed,
        weights_packed_words,
        scales,
        expected,
        m,
        n: I2S_PARITY_N,
        k: I2S_PARITY_K,
        block_size: I2S_PARITY_BLOCK_SIZE,
    }
}

pub fn i2s_parity_shape_words(fixture: &I2sMetalParityFixture) -> [u32; 6] {
    [
        fixture.m as u32,
        fixture.n as u32,
        fixture.k as u32,
        fixture.k.div_ceil(4) as u32,
        fixture.block_size as u32,
        fixture.k.div_ceil(fixture.block_size) as u32,
    ]
}

pub fn pack_i2s_bytes_to_u32_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut padded = [0_u8; 4];
            padded[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(padded)
        })
        .collect()
}

pub fn compare_tiny_add_outputs(
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> Result<SmokeComparison, TinyMetalSmokeError> {
    if expected.is_empty() {
        return Err(TinyMetalSmokeError::EmptyInput);
    }
    if expected.len() != actual.len() {
        return Err(TinyMetalSmokeError::LengthMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    let mut max_abs_error = 0.0_f32;
    let mut total_abs_error = 0.0_f32;

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        if !actual_value.is_finite() {
            return Err(TinyMetalSmokeError::NonFiniteOutput { index, value: actual_value });
        }

        let abs_error = (actual_value - expected_value).abs();
        if abs_error > tolerance {
            return Err(TinyMetalSmokeError::OutputMismatch {
                index,
                expected: expected_value,
                actual: actual_value,
                tolerance,
            });
        }

        max_abs_error = max_abs_error.max(abs_error);
        total_abs_error += abs_error;
    }

    Ok(SmokeComparison { max_abs_error, mean_abs_error: total_abs_error / expected.len() as f32 })
}

pub fn is_apple_m4_adapter_name(adapter_name: &str) -> bool {
    adapter_name.to_ascii_lowercase().contains("apple m4")
}
