//! Apple MPSGraph reference-lane smoke helpers.
//!
//! This module is intentionally separate from native Metal probes and kernels.
//! A passing MPSGraph smoke proves only that a tiny graph executed through the
//! MPSGraph API; it is not native Metal, Neural Engine, or BitNet inference
//! proof.

use std::fmt;

pub const MACHINE_ID: &str = "apple-m4-mac-mini";
pub const ARTIFACT_KIND: &str = "smoke";
pub const APPLE_M4_MPSGRAPH_BACKEND: &str = "apple-m4-mpsgraph";
pub const APPLE_M4_MPSGRAPH_RUNTIME_API: &str = "mpsgraph";
pub const APPLE_M4_MPSGRAPH_RESOLVED_TARGET_UNKNOWN: &str = "unknown";
pub const TINY_MPSGRAPH_MATMUL_GRAPH_ID: &str = "tiny_mpsgraph_matmul";
pub const MPSGRAPH_SMOKE_ELEMENT_COUNT: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct TinyMpsGraphSmokeReceipt {
    pub machine_id: &'static str,
    pub artifact_kind: &'static str,
    pub requested_backend: &'static str,
    pub selected_backend: &'static str,
    pub runtime_api: &'static str,
    pub graph_id: &'static str,
    pub resolved_target: &'static str,
    pub fallback_used: bool,
    pub result: &'static str,
    pub artifact_path: String,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

impl TinyMpsGraphSmokeReceipt {
    pub fn passed(
        artifact_path: impl Into<String>,
        element_count: usize,
        comparison: TinyMpsGraphSmokeComparison,
    ) -> Self {
        Self {
            machine_id: MACHINE_ID,
            artifact_kind: ARTIFACT_KIND,
            requested_backend: APPLE_M4_MPSGRAPH_BACKEND,
            selected_backend: APPLE_M4_MPSGRAPH_BACKEND,
            runtime_api: APPLE_M4_MPSGRAPH_RUNTIME_API,
            graph_id: TINY_MPSGRAPH_MATMUL_GRAPH_ID,
            resolved_target: APPLE_M4_MPSGRAPH_RESOLVED_TARGET_UNKNOWN,
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
pub struct TinyMpsGraphSmokeComparison {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TinyMpsGraphSmokeError {
    EmptyInput,
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteOutput { index: usize, value: f32 },
    OutputMismatch { index: usize, expected: f32, actual: f32, tolerance: f32 },
}

impl fmt::Display for TinyMpsGraphSmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "tiny MPSGraph smoke input must not be empty"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "tiny MPSGraph smoke length mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::NonFiniteOutput { index, value } => {
                write!(f, "tiny MPSGraph smoke output at index {index} is non-finite: {value}")
            }
            Self::OutputMismatch { index, expected, actual, tolerance } => write!(
                f,
                "tiny MPSGraph smoke output mismatch at index {index}: expected {expected}, actual {actual}, tolerance {tolerance}"
            ),
        }
    }
}

impl std::error::Error for TinyMpsGraphSmokeError {}

#[must_use]
pub fn apple_mpsgraph_smoke_artifact_path(date: &str) -> String {
    format!("ci/hardware/{MACHINE_ID}/{date}/mpsgraph-smoke.json")
}

#[must_use]
pub fn tiny_mpsgraph_matmul_inputs()
-> ([f32; MPSGRAPH_SMOKE_ELEMENT_COUNT], [f32; MPSGRAPH_SMOKE_ELEMENT_COUNT]) {
    ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
}

pub fn expected_tiny_mpsgraph_matmul(
    lhs: &[f32],
    rhs: &[f32],
) -> Result<Vec<f32>, TinyMpsGraphSmokeError> {
    if lhs.is_empty() || rhs.is_empty() {
        return Err(TinyMpsGraphSmokeError::EmptyInput);
    }
    if lhs.len() != MPSGRAPH_SMOKE_ELEMENT_COUNT {
        return Err(TinyMpsGraphSmokeError::LengthMismatch {
            expected: MPSGRAPH_SMOKE_ELEMENT_COUNT,
            actual: lhs.len(),
        });
    }
    if rhs.len() != MPSGRAPH_SMOKE_ELEMENT_COUNT {
        return Err(TinyMpsGraphSmokeError::LengthMismatch {
            expected: MPSGRAPH_SMOKE_ELEMENT_COUNT,
            actual: rhs.len(),
        });
    }

    Ok(vec![
        lhs[0] * rhs[0] + lhs[1] * rhs[2],
        lhs[0] * rhs[1] + lhs[1] * rhs[3],
        lhs[2] * rhs[0] + lhs[3] * rhs[2],
        lhs[2] * rhs[1] + lhs[3] * rhs[3],
    ])
}

pub fn compare_tiny_mpsgraph_matmul_outputs(
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> Result<TinyMpsGraphSmokeComparison, TinyMpsGraphSmokeError> {
    if expected.is_empty() {
        return Err(TinyMpsGraphSmokeError::EmptyInput);
    }
    if expected.len() != actual.len() {
        return Err(TinyMpsGraphSmokeError::LengthMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    let mut max_abs_error = 0.0_f32;
    let mut total_abs_error = 0.0_f32;

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        if !actual_value.is_finite() {
            return Err(TinyMpsGraphSmokeError::NonFiniteOutput { index, value: actual_value });
        }

        let abs_error = (actual_value - expected_value).abs();
        if abs_error > tolerance {
            return Err(TinyMpsGraphSmokeError::OutputMismatch {
                index,
                expected: expected_value,
                actual: actual_value,
                tolerance,
            });
        }

        max_abs_error = max_abs_error.max(abs_error);
        total_abs_error += abs_error;
    }

    Ok(TinyMpsGraphSmokeComparison {
        max_abs_error,
        mean_abs_error: total_abs_error / expected.len() as f32,
    })
}

#[must_use]
pub fn tiny_mpsgraph_smoke_swift_source() -> &'static str {
    r#"
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

let graph = MPSGraph()
let lhs = [Float32](arrayLiteral: 1, 2, 3, 4)
let rhs = [Float32](arrayLiteral: 5, 6, 7, 8)
let lhsTensor = graph.constant(
    Data(bytes: lhs, count: lhs.count * MemoryLayout<Float32>.size),
    shape: [2, 2] as [NSNumber],
    dataType: .float32
)
let rhsTensor = graph.constant(
    Data(bytes: rhs, count: rhs.count * MemoryLayout<Float32>.size),
    shape: [2, 2] as [NSNumber],
    dataType: .float32
)
let resultTensor = graph.matrixMultiplication(
    primary: lhsTensor,
    secondary: rhsTensor,
    name: "tiny_mpsgraph_matmul"
)
guard let metalDevice = MTLCreateSystemDefaultDevice(),
      let commandQueue = metalDevice.makeCommandQueue() else {
    fputs("M4-007 MPSGraph smoke requires a Metal command queue\n", stderr)
    exit(2)
}
let results = graph.run(
    with: commandQueue,
    feeds: [:],
    targetTensors: [resultTensor],
    targetOperations: nil
)
guard let tensorData = results[resultTensor] else {
    fputs("M4-007 MPSGraph smoke did not return target tensor data\n", stderr)
    exit(3)
}
let ndarray = tensorData.mpsndarray()
var output = [Float32](repeating: 0, count: 4)
ndarray.readBytes(&output, strideBytes: nil)
let payload: [String: Any] = [
    "graph_id": "tiny_mpsgraph_matmul",
    "device_name": metalDevice.name,
    "resolved_target": "unknown",
    "output": output
]
let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
"#
}
