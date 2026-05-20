use serde::{Deserialize, Serialize};

/// Device configuration mode for runtime initialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DeviceConfig {
    /// Automatically select the best available device (prefer GPU if available).
    #[default]
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Force GPU execution on specific device ID.
    Gpu(usize),
    /// Preserve Intel NPU backend identity without mapping through GPU/Metal/CPU.
    IntelNpu(usize),
    /// Preserve Intel NPU through OpenVINO backend identity.
    OpenVinoNpu,
    /// Preserve the RTX 5070 Ti CUDA proof-lane backend identity.
    NvidiaRtx5070TiCuda,
    /// Preserve the RTX 5070 Ti WGPU reference-lane backend identity.
    NvidiaRtx5070TiWgpu,
    /// Preserve the Intel Arc A770 native OpenCL proof-lane backend identity.
    IntelA770OpenCl,
    /// Preserve a native Metal backend identity.
    Metal,
    /// Preserve an MPSGraph graph/reference backend identity.
    MpsGraph,
    /// Preserve the Apple M4 native Metal backend identity.
    AppleM4Metal,
    /// Preserve the Apple M4 MPSGraph graph/reference backend identity.
    AppleM4MpsGraph,
    /// Preserve the Apple M4 CPU/NEON fallback/parity backend identity.
    AppleM4CpuNeon,
    /// Preserve the Apple M3 MacBook Air native Metal backend identity.
    AppleM3AirMetal,
    /// Preserve the Apple M3 MacBook Air MPSGraph backend identity.
    AppleM3AirMpsGraph,
    /// Preserve the Apple M3 MacBook Air CPU/NEON backend identity.
    AppleM3AirCpuNeon,
}
