//! WebGPU compute backend for cross-platform GPU inference.
//!
//! This crate provides a [`WebGpuBackend`] that wraps wgpu to run WGSL compute
//! shaders on any GPU backend supported by the platform (Vulkan, Metal, DX12,
//! OpenGL, or WebGPU in the browser).

pub mod buffer;
pub mod device;
pub mod error;
pub mod pipeline;
pub mod shader;

pub use buffer::GpuBuffer;
pub use device::WebGpuDevice;
pub use error::WebGpuError;
pub use pipeline::ComputePipeline;

use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use tracing::info;

/// Uniform parameters for the matrix-multiply shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MatmulParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _pad: u32,
}

/// Uniform parameters for the softmax shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SoftmaxParams {
    pub n: u32,
    pub _pad: u32,
}

/// Uniform parameters for element-wise operations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ElementwiseParams {
    pub len: u32,
    pub op: u32,
}

/// Element-wise operation codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ElementwiseOp {
    Add = 0,
    Mul = 1,
    Relu = 2,
    Silu = 3,
}

/// High-level WebGPU backend wrapping device, pipelines and dispatch.
pub struct WebGpuBackend {
    pub gpu: WebGpuDevice,
    matmul_pipeline: ComputePipeline,
    softmax_pipeline: ComputePipeline,
    elementwise_pipeline: ComputePipeline,
}

impl WebGpuBackend {
    /// Initialise the backend: request adapter/device, compile all shaders.
    pub async fn new() -> Result<Self> {
        let gpu = WebGpuDevice::new().await?;

        let matmul_pipeline =
            ComputePipeline::new(&gpu.device, shader::MATMUL_WGSL, "matmul", "main")?;
        let softmax_pipeline =
            ComputePipeline::new(&gpu.device, shader::SOFTMAX_WGSL, "softmax", "main")?;
        let elementwise_pipeline =
            ComputePipeline::new(&gpu.device, shader::ELEMENTWISE_WGSL, "elementwise", "main")?;

        info!(adapter = %gpu.adapter_name(), "WebGPU backend ready");

        Ok(Self { gpu, matmul_pipeline, softmax_pipeline, elementwise_pipeline })
    }

    /// Run matrix multiplication: `C = A × B`.
    ///
    /// `a` is M×K row-major, `b` is K×N row-major, returns M×N.
    pub async fn matmul(&self, a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Result<Vec<f32>> {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        let buf_a = GpuBuffer::from_slice(device, queue, a, "a");
        let buf_b = GpuBuffer::from_slice(device, queue, b, "b");
        let buf_c = GpuBuffer::new_uninit::<f32>(device, (m * n) as usize, "c");
        let params = MatmulParams { m, n, k, _pad: 0 };
        let buf_params = GpuBuffer::new_uniform(device, queue, &params, "matmul-params");

        let bind_group = self.matmul_pipeline.bind_group(
            device,
            &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        );

        let workgroups_x = (n + 15) / 16;
        let workgroups_y = (m + 15) / 16;

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matmul_pipeline.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        buf_c.read_back::<f32>(device, queue).await
    }

    /// Run row-wise softmax over `rows` × `n` matrix.
    pub async fn softmax(&self, input: &[f32], rows: u32, n: u32) -> Result<Vec<f32>> {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        let buf_in = GpuBuffer::from_slice(device, queue, input, "softmax-in");
        let buf_out = GpuBuffer::new_uninit::<f32>(device, input.len(), "softmax-out");
        let params = SoftmaxParams { n, _pad: 0 };
        let buf_params = GpuBuffer::new_uniform(device, queue, &params, "softmax-params");

        let bind_group = self.softmax_pipeline.bind_group(
            device,
            &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        );

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.softmax_pipeline.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(rows, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        buf_out.read_back::<f32>(device, queue).await
    }

    /// Run an element-wise binary/unary operation.
    pub async fn elementwise(&self, a: &[f32], b: &[f32], op: ElementwiseOp) -> Result<Vec<f32>> {
        let len = a.len() as u32;
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;

        let buf_a = GpuBuffer::from_slice(device, queue, a, "ew-a");
        let buf_b = GpuBuffer::from_slice(device, queue, b, "ew-b");
        let buf_c = GpuBuffer::new_uninit::<f32>(device, a.len(), "ew-c");
        let params = ElementwiseParams { len, op: op as u32 };
        let buf_params = GpuBuffer::new_uniform(device, queue, &params, "ew-params");

        let bind_group = self.elementwise_pipeline.bind_group(
            device,
            &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.storage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        );

        let workgroups = (len + 255) / 256;
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.elementwise_pipeline.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        buf_c.read_back::<f32>(device, queue).await
    }

    /// Backend name for logging / registry.
    pub fn name(&self) -> &'static str {
        "webgpu"
    }

    /// Whether the backend is available (always true once constructed).
    pub fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_params_pod_layout() {
        assert_eq!(std::mem::size_of::<MatmulParams>(), 16);
    }

    #[test]
    fn softmax_params_pod_layout() {
        assert_eq!(std::mem::size_of::<SoftmaxParams>(), 8);
    }

    #[test]
    fn elementwise_params_pod_layout() {
        assert_eq!(std::mem::size_of::<ElementwiseParams>(), 8);
    }

    #[test]
    fn elementwise_op_values() {
        assert_eq!(ElementwiseOp::Add as u32, 0);
        assert_eq!(ElementwiseOp::Mul as u32, 1);
        assert_eq!(ElementwiseOp::Relu as u32, 2);
        assert_eq!(ElementwiseOp::Silu as u32, 3);
    }

    #[test]
    fn shader_sources_non_empty() {
        assert!(!shader::MATMUL_WGSL.is_empty());
        assert!(!shader::SOFTMAX_WGSL.is_empty());
        assert!(!shader::ELEMENTWISE_WGSL.is_empty());
    }

    #[test]
    fn matmul_shader_contains_entry_point() {
        assert!(shader::MATMUL_WGSL.contains("fn main"));
    }

    #[test]
    fn softmax_shader_contains_workgroup_barrier() {
        assert!(shader::SOFTMAX_WGSL.contains("workgroupBarrier"));
    }

    #[test]
    fn elementwise_shader_supports_silu() {
        assert!(shader::ELEMENTWISE_WGSL.contains("silu"));
    }

    #[test]
    fn error_display_no_adapter() {
        let e = WebGpuError::NoAdapter;
        assert_eq!(format!("{e}"), "no suitable GPU adapter found");
    }

    #[test]
    fn error_display_invalid_dims() {
        let e = WebGpuError::InvalidDimensions("M must be > 0".into());
        assert!(format!("{e}").contains("M must be > 0"));
    }

    #[test]
    fn matmul_params_zeroed() {
        let p = MatmulParams::zeroed();
        assert_eq!(p.m, 0);
        assert_eq!(p.n, 0);
        assert_eq!(p.k, 0);
    }

    #[test]
    fn elementwise_op_debug() {
        assert_eq!(format!("{:?}", ElementwiseOp::Silu), "Silu");
    }

    // --- integration tests (require GPU adapter) ---

    #[tokio::test]
    #[ignore = "requires GPU adapter - run manually on machines with a GPU"]
    async fn backend_initialisation() {
        let backend = WebGpuBackend::new().await.expect("backend init");
        assert!(backend.is_available());
        assert_eq!(backend.name(), "webgpu");
    }

    #[tokio::test]
    #[ignore = "requires GPU adapter - run manually on machines with a GPU"]
    async fn matmul_identity() {
        let backend = WebGpuBackend::new().await.unwrap();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = backend.matmul(&a, &b, 2, 2, 2).await.unwrap();
        assert_eq!(c.len(), 4);
        for (got, want) in c.iter().zip(b.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }
}
