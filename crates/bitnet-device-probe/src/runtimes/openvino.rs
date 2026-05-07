//! OpenVINO runtime visibility probing without linking OpenVINO.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use super::command_output;

/// OpenVINO property value captured without compiling a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoPropertyProbe {
    /// OpenVINO property name.
    pub name: String,
    /// Stringified property value reported by OpenVINO.
    pub value: String,
}

/// OpenVINO device facts used in platform receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoDeviceProbe {
    /// OpenVINO device token, such as `CPU`, `GPU.0`, or `NPU`.
    pub device: String,
    /// `FULL_DEVICE_NAME` when OpenVINO reports it.
    pub full_name: Option<String>,
    /// Supported property names when collected.
    pub supported_properties: Vec<String>,
    /// Selected property values collected for runtime identity receipts.
    pub properties: Vec<OpenVinoPropertyProbe>,
}

/// OpenVINO runtime visibility result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoProbe {
    /// Whether Python could import OpenVINO and instantiate `ov.Core`.
    pub runtime_available: bool,
    /// OpenVINO Python package version.
    pub version: Option<String>,
    /// Available OpenVINO device tokens.
    pub available_devices: Vec<String>,
    /// Per-device properties collected without compiling a model.
    pub devices: Vec<OpenVinoDeviceProbe>,
    /// Non-fatal probe error when OpenVINO was absent or unusable.
    pub error: Option<String>,
}

/// Tiny static OpenVINO NPU graph smoke result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct OpenVinoNpuTinyGraphSmoke {
    /// Whether the tiny graph executed and matched the CPU expected output.
    pub passed: bool,
    /// Proof stage reached by this smoke result.
    pub proof_stage: String,
    /// Requested backend identity for the Intel NPU lane.
    pub requested_backend: String,
    /// Selected backend when OpenVINO reported or used an NPU device.
    pub selected_backend: Option<String>,
    /// Runtime API used by the smoke.
    pub runtime_api: Option<String>,
    /// Concrete OpenVINO runtime device token, normally `NPU`.
    pub runtime_device: Option<String>,
    /// OpenVINO Python package version when available.
    pub openvino_version: Option<String>,
    /// OpenVINO available device tokens observed by the smoke script.
    pub openvino_available_devices: Vec<String>,
    /// Tiny static graph identifier.
    pub graph_name: String,
    /// Static shape mode expected by the Intel NPU lane.
    pub shape_mode: String,
    /// Input shape used by the tiny graph.
    pub input_shape: Vec<usize>,
    /// Output shape returned by the tiny graph when available.
    pub output_shape: Option<Vec<usize>>,
    /// Graph precision label.
    pub precision: String,
    /// Absolute/mean tolerance used for the CPU expected-output comparison.
    pub tolerance: f32,
    /// Maximum absolute error versus CPU expected output.
    pub max_abs_error: Option<f32>,
    /// Mean absolute error versus CPU expected output.
    pub mean_abs_error: Option<f32>,
    /// Compile time to OpenVINO `NPU` when measured.
    pub compile_ms: Option<f64>,
    /// First inference time when measured.
    pub first_infer_ms: Option<f64>,
    /// Always false: this smoke never substitutes CPU fallback.
    pub fallback_used: bool,
    /// Always false: CPU fallback is not allowed for NPU smoke proof.
    pub cpu_fallback_allowed: bool,
    /// Whether graph execution occurred.
    pub graph_execution: bool,
    /// Always false: this is a tiny graph smoke, not BitNet inference.
    pub bitnet_inference: bool,
    /// Non-fatal error for unavailable runtime, missing NPU, compile, infer, or parity failure.
    pub error: Option<String>,
}

/// Selected static BitNet subgraph parity result on OpenVINO NPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct OpenVinoNpuBitnetSubgraphParity {
    /// Whether the static subgraph executed on NPU and matched the CPU reference.
    pub passed: bool,
    /// Proof stage reached by this parity result.
    pub proof_stage: String,
    /// Requested backend identity for the Intel NPU lane.
    pub requested_backend: String,
    /// Selected backend when OpenVINO reported or used an NPU device.
    pub selected_backend: Option<String>,
    /// Runtime API used by the parity probe.
    pub runtime_api: Option<String>,
    /// Concrete OpenVINO runtime device token, normally `NPU`.
    pub runtime_device: Option<String>,
    /// OpenVINO Python package version when available.
    pub openvino_version: Option<String>,
    /// OpenVINO available device tokens observed by the parity script.
    pub openvino_available_devices: Vec<String>,
    /// Static subgraph identifier.
    pub subgraph_name: String,
    /// BitNet operation represented by this selected subgraph.
    pub bitnet_op: String,
    /// CPU reference path used for parity.
    pub reference_path: String,
    /// Static shape mode expected by the Intel NPU lane.
    pub shape_mode: String,
    /// Input shape used by the selected subgraph.
    pub input_shape: Vec<usize>,
    /// Output shape returned by the selected subgraph when available.
    pub output_shape: Option<Vec<usize>>,
    /// Graph precision label.
    pub precision: String,
    /// RMSNorm epsilon used by the subgraph and CPU reference.
    pub epsilon: f32,
    /// Absolute/mean tolerance used for the CPU reference comparison.
    pub tolerance: f32,
    /// Maximum absolute error versus CPU reference output.
    pub max_abs_error: Option<f32>,
    /// Mean absolute error versus CPU reference output.
    pub mean_abs_error: Option<f32>,
    /// Compile time to OpenVINO `NPU` when measured.
    pub compile_ms: Option<f64>,
    /// First inference time when measured.
    pub first_infer_ms: Option<f64>,
    /// Always false: this parity probe never substitutes CPU fallback.
    pub fallback_used: bool,
    /// Always false: CPU fallback is not allowed for NPU subgraph parity proof.
    pub cpu_fallback_allowed: bool,
    /// Whether graph execution occurred.
    pub graph_execution: bool,
    /// Always false: selected subgraph parity is not full BitNet inference.
    pub bitnet_inference: bool,
    /// Always false: this selected subgraph does not prove packed QK256 decode.
    pub qk256_decode: bool,
    /// Non-fatal error for unavailable runtime, missing NPU, compile, infer, or parity failure.
    pub error: Option<String>,
}

impl OpenVinoNpuTinyGraphSmoke {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            proof_stage: "runtime_detected".to_owned(),
            requested_backend: "intel-npu".to_owned(),
            selected_backend: None,
            runtime_api: None,
            runtime_device: None,
            openvino_version: None,
            openvino_available_devices: Vec::new(),
            graph_name: "tiny_matmul_add_f16_1x16".to_owned(),
            shape_mode: "static".to_owned(),
            input_shape: vec![1, 16],
            output_shape: None,
            precision: "F16".to_owned(),
            tolerance: 0.001,
            max_abs_error: None,
            mean_abs_error: None,
            compile_ms: None,
            first_infer_ms: None,
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: false,
            bitnet_inference: false,
            error: Some(reason.into()),
        }
    }
}

impl OpenVinoNpuBitnetSubgraphParity {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            proof_stage: "runtime_detected".to_owned(),
            requested_backend: "intel-npu".to_owned(),
            selected_backend: None,
            runtime_api: None,
            runtime_device: None,
            openvino_version: None,
            openvino_available_devices: Vec::new(),
            subgraph_name: "bitnet_rmsnorm_f16_1x16".to_owned(),
            bitnet_op: "rmsnorm".to_owned(),
            reference_path: "cpu_numpy_rmsnorm_f32".to_owned(),
            shape_mode: "static".to_owned(),
            input_shape: vec![1, 16],
            output_shape: None,
            precision: "F16".to_owned(),
            epsilon: 0.00001,
            tolerance: 0.005,
            max_abs_error: None,
            mean_abs_error: None,
            compile_ms: None,
            first_infer_ms: None,
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: Some(reason.into()),
        }
    }
}

impl OpenVinoProbe {
    /// Build an unavailable OpenVINO probe result.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            runtime_available: false,
            version: None,
            available_devices: Vec::new(),
            devices: Vec::new(),
            error: Some(reason.into()),
        }
    }

    /// Return the first OpenVINO GPU token, preferring `GPU.0`.
    pub fn gpu_device_token(&self) -> Option<String> {
        self.available_devices
            .iter()
            .find(|device| device.as_str() == "GPU.0")
            .or_else(|| self.available_devices.iter().find(|device| device.starts_with("GPU")))
            .cloned()
    }

    /// Return the full name for an OpenVINO device token.
    pub fn full_name_for(&self, token: &str) -> Option<String> {
        self.device_for(token).and_then(|device| device.full_name.clone())
    }

    /// Return per-device facts for an OpenVINO device token.
    pub fn device_for(&self, token: &str) -> Option<&OpenVinoDeviceProbe> {
        self.devices.iter().find(|device| device.device == token)
    }

    /// Return whether OpenVINO exposes an `NPU` device.
    pub fn npu_visible(&self) -> bool {
        self.available_devices.iter().any(|device| device == "NPU" || device.starts_with("NPU."))
    }

    /// Return the first OpenVINO NPU token, preferring the unindexed `NPU` alias.
    pub fn npu_device_token(&self) -> Option<String> {
        self.available_devices
            .iter()
            .find(|device| device.as_str() == "NPU")
            .or_else(|| self.available_devices.iter().find(|device| device.starts_with("NPU.")))
            .cloned()
    }

    /// Return a stringified OpenVINO property value for a device.
    pub fn property_value_for(&self, token: &str, property: &str) -> Option<String> {
        self.device_for(token).and_then(|device| {
            device
                .properties
                .iter()
                .find(|entry| entry.name == property)
                .map(|entry| entry.value.clone())
        })
    }
}

/// Probe OpenVINO runtime visibility through Python, without compiling a model.
pub fn probe_openvino() -> OpenVinoProbe {
    let script = r#"
import openvino as ov

core = ov.Core()
print("OPENVINO_VERSION=" + str(ov.__version__))
common_props = [
    "FULL_DEVICE_NAME",
    "SUPPORTED_PROPERTIES",
]
npu_props = [
    "NPU_DRIVER_VERSION",
    "NPU_COMPILER_VERSION",
    "NPU_DEVICE_TOTAL_MEM_SIZE",
    "NPU_DEVICE_ALLOC_MEM_SIZE",
    "NPU_MAX_TILES",
]
for dev in core.available_devices:
    print("DEVICE=" + str(dev))
    props = list(common_props)
    if str(dev) == "NPU" or str(dev).startswith("NPU."):
        props.extend(npu_props)
    for prop in props:
        try:
            print("PROP=" + str(dev) + "=" + prop + "=" + str(core.get_property(dev, prop)))
        except Exception as exc:
            print("PROP_ERR=" + str(dev) + "=" + prop + "=" + repr(exc))
"#;

    for python in ["python3", "python"] {
        if let Ok(stdout) = command_output(python, ["-c", script]) {
            return parse_openvino_line_output(&stdout);
        }
    }

    OpenVinoProbe::unavailable("python openvino import unavailable")
}

/// Run a tiny static OpenVINO graph on `NPU` through Python, without linking OpenVINO.
pub fn run_openvino_npu_tiny_graph_smoke() -> OpenVinoNpuTinyGraphSmoke {
    let script = r#"
import time

try:
    import openvino as ov
    import numpy as np
    from openvino import opset8 as opset

    core = ov.Core()
    print("OPENVINO_VERSION=" + str(ov.__version__))
    for dev in core.available_devices:
        print("AVAILABLE_DEVICE=" + str(dev))

    if not any(str(dev) == "NPU" or str(dev).startswith("NPU.") for dev in core.available_devices):
        print("RESULT=fail")
        print("ERROR=OpenVINO did not report NPU")
    else:
        device = "NPU"
        input_data = np.arange(16, dtype=np.float16).reshape(1, 16)
        expected = input_data + np.ones((1, 16), dtype=np.float16)

        param = opset.parameter([1, 16], dtype=np.float16, name="input")
        weights = opset.constant(np.eye(16, dtype=np.float16))
        bias = opset.constant(np.ones((1, 16), dtype=np.float16))
        matmul = opset.matmul(param, weights, False, False)
        output = opset.add(matmul, bias)
        model = ov.Model([output], [param], "tiny_matmul_add_f16_1x16")

        compile_start = time.perf_counter()
        compiled = core.compile_model(model, device)
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

        infer_start = time.perf_counter()
        result = compiled([input_data])
        first_infer_ms = (time.perf_counter() - infer_start) * 1000.0
        actual = np.asarray(next(iter(result.values())))

        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        passed = bool(max_abs <= 0.001)

        print("SELECTED_DEVICE=" + device)
        print("GRAPH_NAME=tiny_matmul_add_f16_1x16")
        print("SHAPE_MODE=static")
        print("PRECISION=F16")
        print("INPUT_SHAPE=1,16")
        print("OUTPUT_SHAPE=" + ",".join(str(dim) for dim in actual.shape))
        print("COMPILE_MS=" + str(compile_ms))
        print("FIRST_INFER_MS=" + str(first_infer_ms))
        print("MAX_ABS_ERROR=" + str(max_abs))
        print("MEAN_ABS_ERROR=" + str(mean_abs))
        print("RESULT=" + ("pass" if passed else "fail"))
        if not passed:
            print("ERROR=Output mismatch versus CPU expected output")
except Exception as exc:
    print("RESULT=fail")
    print("ERROR=" + repr(exc))
"#;

    for python in ["python3", "python"] {
        if let Ok(stdout) = command_output(python, ["-c", script]) {
            return parse_openvino_npu_tiny_graph_smoke_output(&stdout);
        }
    }

    OpenVinoNpuTinyGraphSmoke::unavailable("python openvino smoke unavailable")
}

/// Run selected static BitNet subgraph parity on `NPU` through Python, without linking OpenVINO.
pub fn run_openvino_npu_bitnet_subgraph_parity() -> OpenVinoNpuBitnetSubgraphParity {
    let script = r#"
import time

try:
    import openvino as ov
    import numpy as np
    from openvino import opset8 as opset

    core = ov.Core()
    print("OPENVINO_VERSION=" + str(ov.__version__))
    for dev in core.available_devices:
        print("AVAILABLE_DEVICE=" + str(dev))

    if not any(str(dev) == "NPU" or str(dev).startswith("NPU.") for dev in core.available_devices):
        print("RESULT=fail")
        print("ERROR=OpenVINO did not report NPU")
    else:
        device = "NPU"
        epsilon = np.float32(1e-5)
        tolerance = np.float32(0.005)
        input_data = np.linspace(-1.5, 1.5, 16, dtype=np.float16).reshape(1, 16)
        weight_data = np.linspace(0.75, 1.25, 16, dtype=np.float16).reshape(1, 16)

        x32 = input_data.astype(np.float32)
        w32 = weight_data.astype(np.float32)
        expected = (x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)) * w32

        param = opset.parameter([1, 16], dtype=np.float16, name="input")
        weights = opset.constant(weight_data.astype(np.float16))
        axes = opset.constant(np.array([-1], dtype=np.int64))
        square = opset.multiply(param, param)
        mean = opset.reduce_mean(square, axes, True)
        eps = opset.constant(np.array(epsilon, dtype=np.float16))
        denom = opset.sqrt(opset.add(mean, eps))
        normalized = opset.divide(param, denom)
        output = opset.multiply(normalized, weights)
        model = ov.Model([output], [param], "bitnet_rmsnorm_f16_1x16")

        compile_start = time.perf_counter()
        compiled = core.compile_model(model, device)
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

        infer_start = time.perf_counter()
        result = compiled([input_data])
        first_infer_ms = (time.perf_counter() - infer_start) * 1000.0
        actual = np.asarray(next(iter(result.values())))

        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        passed = bool(max_abs <= tolerance)

        print("SELECTED_DEVICE=" + device)
        print("SUBGRAPH_NAME=bitnet_rmsnorm_f16_1x16")
        print("BITNET_OP=rmsnorm")
        print("REFERENCE_PATH=cpu_numpy_rmsnorm_f32")
        print("SHAPE_MODE=static")
        print("PRECISION=F16")
        print("EPSILON=" + str(float(epsilon)))
        print("TOLERANCE=" + str(float(tolerance)))
        print("INPUT_SHAPE=1,16")
        print("OUTPUT_SHAPE=" + ",".join(str(dim) for dim in actual.shape))
        print("COMPILE_MS=" + str(compile_ms))
        print("FIRST_INFER_MS=" + str(first_infer_ms))
        print("MAX_ABS_ERROR=" + str(max_abs))
        print("MEAN_ABS_ERROR=" + str(mean_abs))
        print("RESULT=" + ("pass" if passed else "fail"))
        if not passed:
            print("ERROR=RMSNorm output mismatch versus CPU reference output")
except Exception as exc:
    print("RESULT=fail")
    print("ERROR=" + repr(exc))
"#;

    for python in ["python3", "python"] {
        if let Ok(stdout) = command_output(python, ["-c", script]) {
            return parse_openvino_npu_bitnet_subgraph_parity_output(&stdout);
        }
    }

    OpenVinoNpuBitnetSubgraphParity::unavailable("python openvino subgraph parity unavailable")
}

pub(crate) fn parse_openvino_line_output(output: &str) -> OpenVinoProbe {
    let mut probe = OpenVinoProbe {
        runtime_available: true,
        version: None,
        available_devices: Vec::new(),
        devices: Vec::new(),
        error: None,
    };

    for line in output.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if let Some(value) = line.strip_prefix("OPENVINO_VERSION=") {
            probe.version = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("DEVICE=") {
            if !probe.available_devices.iter().any(|device| device == value) {
                probe.available_devices.push(value.to_owned());
                probe.devices.push(OpenVinoDeviceProbe {
                    device: value.to_owned(),
                    full_name: None,
                    supported_properties: Vec::new(),
                    properties: Vec::new(),
                });
            }
        } else if let Some(rest) = line.strip_prefix("PROP=") {
            apply_property_line(&mut probe, rest);
        }
    }

    probe
}

pub(crate) fn parse_openvino_npu_tiny_graph_smoke_output(
    output: &str,
) -> OpenVinoNpuTinyGraphSmoke {
    let mut smoke = OpenVinoNpuTinyGraphSmoke::unavailable("OpenVINO NPU smoke did not pass");
    smoke.openvino_available_devices.clear();

    for line in output.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if let Some(value) = line.strip_prefix("RESULT=") {
            smoke.passed = value == "pass";
        } else if let Some(value) = line.strip_prefix("ERROR=") {
            smoke.error = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("OPENVINO_VERSION=") {
            smoke.openvino_version = Some(value.to_owned());
            smoke.runtime_api = Some("openvino".to_owned());
        } else if let Some(value) = line.strip_prefix("AVAILABLE_DEVICE=") {
            smoke.openvino_available_devices.push(value.to_owned());
            if value == "NPU" || value.starts_with("NPU.") {
                smoke.selected_backend = Some("intel-npu-openvino".to_owned());
                smoke.runtime_api = Some("openvino".to_owned());
                smoke.runtime_device = Some(value.to_owned());
            }
        } else if let Some(value) = line.strip_prefix("SELECTED_DEVICE=") {
            smoke.selected_backend = Some("intel-npu-openvino".to_owned());
            smoke.runtime_api = Some("openvino".to_owned());
            smoke.runtime_device = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("GRAPH_NAME=") {
            value.clone_into(&mut smoke.graph_name);
        } else if let Some(value) = line.strip_prefix("SHAPE_MODE=") {
            value.clone_into(&mut smoke.shape_mode);
        } else if let Some(value) = line.strip_prefix("PRECISION=") {
            value.clone_into(&mut smoke.precision);
        } else if let Some(value) = line.strip_prefix("INPUT_SHAPE=") {
            smoke.input_shape = parse_usize_list(value);
        } else if let Some(value) = line.strip_prefix("OUTPUT_SHAPE=") {
            smoke.output_shape = Some(parse_usize_list(value));
        } else if let Some(value) = line.strip_prefix("COMPILE_MS=") {
            smoke.compile_ms = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("FIRST_INFER_MS=") {
            smoke.first_infer_ms = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("MAX_ABS_ERROR=") {
            smoke.max_abs_error = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("MEAN_ABS_ERROR=") {
            smoke.mean_abs_error = value.parse().ok();
        }
    }

    smoke.graph_execution = smoke.passed;
    smoke.proof_stage = if smoke.passed {
        smoke.error = None;
        "kernel_smoke_tested".to_owned()
    } else {
        "runtime_detected".to_owned()
    };
    smoke.fallback_used = false;
    smoke.cpu_fallback_allowed = false;
    smoke.bitnet_inference = false;
    smoke
}

pub(crate) fn parse_openvino_npu_bitnet_subgraph_parity_output(
    output: &str,
) -> OpenVinoNpuBitnetSubgraphParity {
    let mut parity = OpenVinoNpuBitnetSubgraphParity::unavailable(
        "OpenVINO NPU BitNet subgraph parity did not pass",
    );
    parity.openvino_available_devices.clear();

    for line in output.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if let Some(value) = line.strip_prefix("RESULT=") {
            parity.passed = value == "pass";
        } else if let Some(value) = line.strip_prefix("ERROR=") {
            parity.error = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("OPENVINO_VERSION=") {
            parity.openvino_version = Some(value.to_owned());
            parity.runtime_api = Some("openvino".to_owned());
        } else if let Some(value) = line.strip_prefix("AVAILABLE_DEVICE=") {
            parity.openvino_available_devices.push(value.to_owned());
            if value == "NPU" || value.starts_with("NPU.") {
                parity.selected_backend = Some("intel-npu-openvino".to_owned());
                parity.runtime_api = Some("openvino".to_owned());
                parity.runtime_device = Some(value.to_owned());
            }
        } else if let Some(value) = line.strip_prefix("SELECTED_DEVICE=") {
            parity.selected_backend = Some("intel-npu-openvino".to_owned());
            parity.runtime_api = Some("openvino".to_owned());
            parity.runtime_device = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("SUBGRAPH_NAME=") {
            value.clone_into(&mut parity.subgraph_name);
        } else if let Some(value) = line.strip_prefix("BITNET_OP=") {
            value.clone_into(&mut parity.bitnet_op);
        } else if let Some(value) = line.strip_prefix("REFERENCE_PATH=") {
            value.clone_into(&mut parity.reference_path);
        } else if let Some(value) = line.strip_prefix("SHAPE_MODE=") {
            value.clone_into(&mut parity.shape_mode);
        } else if let Some(value) = line.strip_prefix("PRECISION=") {
            value.clone_into(&mut parity.precision);
        } else if let Some(value) = line.strip_prefix("EPSILON=") {
            parity.epsilon = value.parse().unwrap_or(parity.epsilon);
        } else if let Some(value) = line.strip_prefix("TOLERANCE=") {
            parity.tolerance = value.parse().unwrap_or(parity.tolerance);
        } else if let Some(value) = line.strip_prefix("INPUT_SHAPE=") {
            parity.input_shape = parse_usize_list(value);
        } else if let Some(value) = line.strip_prefix("OUTPUT_SHAPE=") {
            parity.output_shape = Some(parse_usize_list(value));
        } else if let Some(value) = line.strip_prefix("COMPILE_MS=") {
            parity.compile_ms = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("FIRST_INFER_MS=") {
            parity.first_infer_ms = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("MAX_ABS_ERROR=") {
            parity.max_abs_error = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("MEAN_ABS_ERROR=") {
            parity.mean_abs_error = value.parse().ok();
        }
    }

    parity.graph_execution = parity.passed;
    parity.proof_stage = if parity.passed {
        parity.error = None;
        "parity_tested".to_owned()
    } else {
        "runtime_detected".to_owned()
    };
    parity.fallback_used = false;
    parity.cpu_fallback_allowed = false;
    parity.bitnet_inference = false;
    parity.qk256_decode = false;
    parity
}

fn parse_usize_list(value: &str) -> Vec<usize> {
    value.split(',').filter_map(|entry| entry.trim().parse::<usize>().ok()).collect()
}

fn apply_property_line(probe: &mut OpenVinoProbe, rest: &str) {
    let mut parts = rest.splitn(3, '=');
    let Some(device_token) = parts.next() else {
        return;
    };
    let Some(property) = parts.next() else {
        return;
    };
    let Some(value) = parts.next() else {
        return;
    };

    if !probe.available_devices.iter().any(|device| device == device_token) {
        probe.available_devices.push(device_token.to_owned());
    }

    let idx =
        if let Some(idx) = probe.devices.iter().position(|device| device.device == device_token) {
            idx
        } else {
            probe.devices.push(OpenVinoDeviceProbe {
                device: device_token.to_owned(),
                full_name: None,
                supported_properties: Vec::new(),
                properties: Vec::new(),
            });
            probe.devices.len() - 1
        };

    match property {
        "FULL_DEVICE_NAME" if !value.trim().is_empty() => {
            probe.devices[idx].full_name = Some(value.trim().to_owned());
        }
        "SUPPORTED_PROPERTIES" => {
            probe.devices[idx].supported_properties = value
                .trim_matches(['[', ']'])
                .split(',')
                .map(|entry| entry.trim().trim_matches(['\'', '"']).to_owned())
                .filter(|entry| !entry.is_empty())
                .collect();
        }
        _ => upsert_property_value(&mut probe.devices[idx], property, value),
    }
}

fn upsert_property_value(device: &mut OpenVinoDeviceProbe, property: &str, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }

    if let Some(entry) = device.properties.iter_mut().find(|entry| entry.name == property) {
        value.clone_into(&mut entry.value);
    } else {
        device
            .properties
            .push(OpenVinoPropertyProbe { name: property.to_owned(), value: value.to_owned() });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_openvino_line_output, parse_openvino_npu_bitnet_subgraph_parity_output,
        parse_openvino_npu_tiny_graph_smoke_output,
    };

    #[test]
    fn parses_openvino_npu_runtime_properties() {
        let output = r"
OPENVINO_VERSION=2026.1
DEVICE=CPU
PROP=CPU=FULL_DEVICE_NAME=Intel CPU
DEVICE=NPU
PROP=NPU=FULL_DEVICE_NAME=Intel(R) AI Boost
PROP=NPU=SUPPORTED_PROPERTIES=['FULL_DEVICE_NAME', 'NPU_DRIVER_VERSION']
PROP=NPU=NPU_DRIVER_VERSION=1.2.3
PROP=NPU=NPU_COMPILER_VERSION=4.5.6
PROP=NPU=NPU_DEVICE_TOTAL_MEM_SIZE=123456
PROP=NPU=NPU_DEVICE_ALLOC_MEM_SIZE=65432
PROP=NPU=NPU_MAX_TILES=2
";

        let probe = parse_openvino_line_output(output);

        assert!(probe.runtime_available);
        assert_eq!(probe.version.as_deref(), Some("2026.1"));
        assert_eq!(probe.npu_device_token().as_deref(), Some("NPU"));
        assert_eq!(probe.full_name_for("NPU").as_deref(), Some("Intel(R) AI Boost"));
        assert_eq!(probe.property_value_for("NPU", "NPU_DRIVER_VERSION").as_deref(), Some("1.2.3"));
        assert_eq!(
            probe.property_value_for("NPU", "NPU_DEVICE_TOTAL_MEM_SIZE").as_deref(),
            Some("123456")
        );
    }

    #[test]
    fn parses_openvino_npu_tiny_graph_smoke_pass() {
        let output = r"
OPENVINO_VERSION=2026.1
AVAILABLE_DEVICE=CPU
AVAILABLE_DEVICE=NPU
SELECTED_DEVICE=NPU
GRAPH_NAME=tiny_matmul_add_f16_1x16
SHAPE_MODE=static
PRECISION=F16
INPUT_SHAPE=1,16
OUTPUT_SHAPE=1,16
COMPILE_MS=12.5
FIRST_INFER_MS=1.25
MAX_ABS_ERROR=0.0
MEAN_ABS_ERROR=0.0
RESULT=pass
";

        let smoke = parse_openvino_npu_tiny_graph_smoke_output(output);

        assert!(smoke.passed);
        assert_eq!(smoke.proof_stage, "kernel_smoke_tested");
        assert_eq!(smoke.selected_backend.as_deref(), Some("intel-npu-openvino"));
        assert_eq!(smoke.runtime_api.as_deref(), Some("openvino"));
        assert_eq!(smoke.runtime_device.as_deref(), Some("NPU"));
        assert_eq!(smoke.input_shape, [1, 16]);
        assert_eq!(smoke.output_shape.as_deref(), Some(&[1, 16][..]));
        assert_eq!(smoke.precision, "F16");
        assert_eq!(smoke.max_abs_error, Some(0.0));
        assert!(!smoke.fallback_used);
        assert!(!smoke.cpu_fallback_allowed);
        assert!(smoke.graph_execution);
        assert!(!smoke.bitnet_inference);
        assert!(smoke.error.is_none());
    }

    #[test]
    fn parses_openvino_npu_tiny_graph_smoke_missing_npu_as_runtime_only() {
        let output = r"
OPENVINO_VERSION=2026.1
AVAILABLE_DEVICE=CPU
RESULT=fail
ERROR=OpenVINO did not report NPU
";

        let smoke = parse_openvino_npu_tiny_graph_smoke_output(output);

        assert!(!smoke.passed);
        assert_eq!(smoke.proof_stage, "runtime_detected");
        assert_eq!(smoke.runtime_api.as_deref(), Some("openvino"));
        assert!(smoke.runtime_device.is_none());
        assert!(!smoke.graph_execution);
        assert!(!smoke.fallback_used);
        assert_eq!(smoke.error.as_deref(), Some("OpenVINO did not report NPU"));
    }

    #[test]
    fn parses_openvino_npu_bitnet_subgraph_parity_pass() {
        let output = r"
OPENVINO_VERSION=2026.1
AVAILABLE_DEVICE=CPU
AVAILABLE_DEVICE=NPU
SELECTED_DEVICE=NPU
SUBGRAPH_NAME=bitnet_rmsnorm_f16_1x16
BITNET_OP=rmsnorm
REFERENCE_PATH=cpu_numpy_rmsnorm_f32
SHAPE_MODE=static
PRECISION=F16
EPSILON=0.00001
TOLERANCE=0.005
INPUT_SHAPE=1,16
OUTPUT_SHAPE=1,16
COMPILE_MS=14.5
FIRST_INFER_MS=1.5
MAX_ABS_ERROR=0.0009
MEAN_ABS_ERROR=0.0002
RESULT=pass
";

        let parity = parse_openvino_npu_bitnet_subgraph_parity_output(output);

        assert!(parity.passed);
        assert_eq!(parity.proof_stage, "parity_tested");
        assert_eq!(parity.selected_backend.as_deref(), Some("intel-npu-openvino"));
        assert_eq!(parity.runtime_api.as_deref(), Some("openvino"));
        assert_eq!(parity.runtime_device.as_deref(), Some("NPU"));
        assert_eq!(parity.subgraph_name, "bitnet_rmsnorm_f16_1x16");
        assert_eq!(parity.bitnet_op, "rmsnorm");
        assert_eq!(parity.reference_path, "cpu_numpy_rmsnorm_f32");
        assert_eq!(parity.input_shape, [1, 16]);
        assert_eq!(parity.output_shape.as_deref(), Some(&[1, 16][..]));
        assert_eq!(parity.max_abs_error, Some(0.0009));
        assert_eq!(parity.mean_abs_error, Some(0.0002));
        assert!(!parity.fallback_used);
        assert!(!parity.cpu_fallback_allowed);
        assert!(parity.graph_execution);
        assert!(!parity.bitnet_inference);
        assert!(!parity.qk256_decode);
        assert!(parity.error.is_none());
    }

    #[test]
    fn parses_openvino_npu_bitnet_subgraph_missing_npu_as_runtime_only() {
        let output = r"
OPENVINO_VERSION=2026.1
AVAILABLE_DEVICE=CPU
RESULT=fail
ERROR=OpenVINO did not report NPU
";

        let parity = parse_openvino_npu_bitnet_subgraph_parity_output(output);

        assert!(!parity.passed);
        assert_eq!(parity.proof_stage, "runtime_detected");
        assert_eq!(parity.runtime_api.as_deref(), Some("openvino"));
        assert!(parity.runtime_device.is_none());
        assert_eq!(parity.subgraph_name, "bitnet_rmsnorm_f16_1x16");
        assert_eq!(parity.bitnet_op, "rmsnorm");
        assert!(!parity.graph_execution);
        assert!(!parity.fallback_used);
        assert!(!parity.bitnet_inference);
        assert!(!parity.qk256_decode);
        assert_eq!(parity.error.as_deref(), Some("OpenVINO did not report NPU"));
    }
}
