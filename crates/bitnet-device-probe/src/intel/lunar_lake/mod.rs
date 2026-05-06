//! Intel Core Ultra 7 258V Lunar Lake platform visibility probes.

pub mod arc140v;
pub mod cpu;
pub mod npu;
pub mod platform;

pub use crate::runtimes::OpenVinoProbe;
pub use arc140v::{IntelArc140vProbe, probe_intel_arc_140v};
pub use cpu::{Lnl258vCpuProbe, probe_lnl258v_cpu};
pub use npu::{
    INTEL_NPU_OPENVINO_BACKEND, INTEL_NPU_PROOF_STAGE_RUNTIME_DETECTED,
    INTEL_NPU_REQUESTED_BACKEND, INTEL_NPU_RUNTIME_API_OPENVINO, IntelNpuProbe, probe_intel_npu,
};
pub use platform::{
    LNL258V_PROOF_STAGE_RUNTIME_DETECTED, Lnl258vPlatformProbe, PlatformMemoryProbe,
    PlatformPowerProbe, probe_lnl258v_platform,
};
