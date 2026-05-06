use std::fmt;

pub const MACHINE_ID: &str = "apple-m4-mac-mini";
pub const ARTIFACT_KIND: &str = "smoke";
pub const PARITY_ARTIFACT_KIND: &str = "parity";
pub const REQUESTED_BACKEND: &str = "apple-m4-metal";
pub const SELECTED_BACKEND: &str = "apple-m4-metal";
pub const REFERENCE_BACKEND: &str = "apple-m4-cpu-neon";
pub const RUNTIME_API: &str = "metal";
pub const TINY_METAL_ADD_SMOKE_KERNEL_ID: &str = "tiny_metal_add_smoke";
pub const TINY_METAL_ADD_PARITY_KERNEL_ID: &str = "tiny_metal_add_parity";
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
