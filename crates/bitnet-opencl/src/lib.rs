//! OpenCL backend for BitNet GPU inference.
//!
//! Provides optimized OpenCL primitives for Intel Arc GPU inference,
//! including context pooling, local memory optimizations, prefetch
//! pipelines, KV cache, and paged attention.

pub mod context_pool;
pub mod kv_cache;
pub mod paged_attention;
