//! Metal Shading Language (MSL) compute kernels for Apple Silicon GPU inference.
//!
//! This crate provides embedded MSL kernel sources for core neural network
//! operations. Kernels are compiled at runtime by the Metal framework on macOS.
//!
//! # Kernels
//!
//! - **matmul** — matrix multiplication (naive + tiled)
//! - **softmax** — numerically stable softmax with threadgroup reduction
//! - **rmsnorm** — RMS normalization with threadgroup reduction
//! - **rope** — rotary position embeddings + frequency table builder
//! - **attention** — scaled dot-product attention with causal mask
//! - **elementwise** — `SiLU`, GELU, add, mul, `silu_mul`, `scalar_mul`

pub mod command;
pub mod error;
pub mod kernels;
pub mod pipeline;
pub mod shader;

pub use command::{CommandBuffer, CommandBufferState};
pub use kernels::{MetalKernelSource, kernel_function_names, kernel_source};
pub use pipeline::{PipelineCache, PipelineDescriptor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_shader_non_empty() {
        assert!(!shader::MATMUL_MSL.is_empty());
    }

    #[test]
    fn softmax_shader_non_empty() {
        assert!(!shader::SOFTMAX_MSL.is_empty());
    }

    #[test]
    fn rmsnorm_shader_non_empty() {
        assert!(!shader::RMSNORM_MSL.is_empty());
    }

    #[test]
    fn matmul_shader_has_kernel_function() {
        assert!(shader::MATMUL_MSL.contains("kernel void matmul"));
    }

    #[test]
    fn softmax_shader_has_barrier() {
        assert!(shader::SOFTMAX_MSL.contains("threadgroup_barrier"));
    }

    #[test]
    fn rmsnorm_shader_has_eps() {
        assert!(shader::RMSNORM_MSL.contains("eps"));
    }

    #[test]
    fn rope_shader_non_empty() {
        assert!(!shader::ROPE_MSL.is_empty());
    }

    #[test]
    fn rope_shader_has_kernel_function() {
        assert!(shader::ROPE_MSL.contains("kernel void rope"));
    }

    #[test]
    fn attention_shader_non_empty() {
        assert!(!shader::ATTENTION_MSL.is_empty());
    }

    #[test]
    fn attention_shader_has_kernel_function() {
        assert!(shader::ATTENTION_MSL.contains("kernel void attention_scores"));
    }

    #[test]
    fn all_kernels_registry_has_five_entries() {
        assert_eq!(shader::ALL_KERNELS.len(), 5);
    }

    #[test]
    fn get_kernel_source_found() {
        assert!(shader::get_kernel_source("matmul").is_some());
        assert!(shader::get_kernel_source("rope").is_some());
        assert!(shader::get_kernel_source("attention").is_some());
    }

    #[test]
    fn get_kernel_source_not_found() {
        assert!(shader::get_kernel_source("nonexistent").is_none());
    }

    #[test]
    fn pipeline_cache_validates_all_builtin_kernels() {
        let cache = PipelineCache::new();
        for (name, _) in shader::ALL_KERNELS {
            assert!(cache.validate(name).is_ok(), "pipeline missing: {name}");
        }
    }

    #[test]
    fn command_buffer_encode_commit_wait_lifecycle() {
        let mut buf = CommandBuffer::new();
        let cache = PipelineCache::new();
        let pipe = cache.get("softmax").unwrap();
        buf.encode_dispatch(pipe, (32, 1, 1)).unwrap();
        buf.commit().unwrap();
        buf.wait_until_completed().unwrap();
        assert_eq!(buf.state(), CommandBufferState::Completed);
    }
}
