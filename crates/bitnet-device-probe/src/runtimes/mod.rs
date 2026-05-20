//! Runtime visibility probes used by lane-specific hardware probes.

use std::{
    ffi::OsStr,
    process::{Command, Stdio},
};

pub mod level_zero;
pub mod opencl;
pub mod openvino;

pub use level_zero::LevelZeroProbe;
pub use opencl::{
    OpenClRuntimeDevice, OpenClRuntimeProbe, OpenClTinyKernelSmoke,
    run_a770_opencl_tiny_kernel_smoke, run_arc140v_opencl_tiny_kernel_smoke,
};
pub use openvino::{
    OpenVinoDeviceProbe, OpenVinoGpuTinyGraphSmoke, OpenVinoNpuBitnetSubgraphParity,
    OpenVinoNpuTinyGraphSmoke, OpenVinoProbe, OpenVinoPropertyProbe,
    run_openvino_gpu_tiny_graph_smoke, run_openvino_npu_bitnet_ffn_relu2_parity,
    run_openvino_npu_bitnet_linear_projection_parity, run_openvino_npu_bitnet_subgraph_parity,
    run_openvino_npu_tiny_graph_smoke,
};

pub(crate) fn command_output<I, S>(command: &str, args: I) -> Result<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(command)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("{command} unavailable: {err}"))
        .and_then(|output| {
            if output.status.success() {
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
                let reason = if stderr.is_empty() {
                    format!("{command} exited with {}", output.status)
                } else {
                    format!("{command} exited with {}: {stderr}", output.status)
                };
                Err(reason)
            }
        })
}
