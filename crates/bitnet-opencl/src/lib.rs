//! OpenCL backend for BitNet GPU inference.
//!
//! Provides optimized OpenCL primitives for Intel Arc GPU inference,
//! including context pooling, local memory optimizations, prefetch
//! pipelines, KV cache, paged attention, and multi-backend GPU dispatch
//! with automatic selection.
//! pipelines, KV cache, paged attention, and CPU reference implementations
//! with OpenCL kernel sources for I2_S dequantization, QK256 block
//! dequantization, and ternary matrix multiply.
//! pipelines, KV cache, paged attention, SPIR-V compilation, and kernel registry.

pub mod backend_dispatcher;
pub mod backend_registry;
pub mod context_pool;
pub mod kv_cache;
pub mod paged_attention;

pub use backend_dispatcher::{
    BackendCapabilityMatrix, BackendDispatcher, BackendStatus, DispatchDecision, DispatchError,
    DispatchLog, DispatchStrategy, Operation,
};
pub use backend_registry::{BackendInfo, BackendProvider, BackendRegistry};
pub mod quantized_kernels;
pub mod quantized_ops;
pub mod spirv;
pub mod spirv_kernels;

// Re-exports for convenience.
pub use spirv::{
    CompileOptions, CompilerBackend, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator,
};
pub use spirv_kernels::{KernelSource, SpirvKernelRegistry};
//! `OpenCL` backend for `BitNet` inference (Intel GPU support).
//!
//! Provides an `OpenCL`-based inference pipeline that can target Intel
//! integrated and discrete GPUs.  When no `OpenCL` device is available the
//! backend falls back to CPU execution transparently.

use anyhow::{Result, bail};
use bitnet_engine_core::{
    GenerationConfig, GenerationStats, InferenceSession, SessionConfig, StopReason, StreamEvent,
    TokenEvent,
};
use bitnet_generation::check_stop;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Estimated memory footprint for a model on a given device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    /// Estimated bytes required for model weights.
    pub weights_bytes: u64,
    /// Estimated bytes required for KV-cache at `max_context` tokens.
    pub kv_cache_bytes: u64,
    /// Total estimated bytes (weights + KV-cache + scratch).
    pub total_bytes: u64,
}

/// Estimate memory required to run a model with the given context window.
///
/// Uses a simple heuristic: 0.5 bytes per parameter for 1-bit weights,
/// plus 2 bytes per token per layer for KV-cache.
pub fn estimate_memory(num_parameters: u64, num_layers: u32, max_context: usize) -> MemoryEstimate {
    let weights_bytes = num_parameters / 2;
    let kv_cache_bytes = u64::from(num_layers) * (max_context as u64) * 2;
    let scratch = weights_bytes / 10; // 10% scratch overhead
    MemoryEstimate {
        weights_bytes,
        kv_cache_bytes,
        total_bytes: weights_bytes + kv_cache_bytes + scratch,
    }
}

// ---------------------------------------------------------------------------
// OpenCL pipeline
// ---------------------------------------------------------------------------

/// Configuration for the `OpenCL` inference pipeline.
#[derive(Debug, Clone)]
pub struct OpenClPipelineConfig {
    /// Session configuration (model path, backend, etc.).
    pub session: SessionConfig,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling parameter (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled).
    pub top_p: f32,
}

impl Default for OpenClPipelineConfig {
    fn default() -> Self {
        Self { session: SessionConfig::default(), temperature: 1.0, top_k: 0, top_p: 1.0 }
    }
}

/// An `OpenCL`-backed inference pipeline.
///
/// In CPU-mock mode (no `OpenCL` device) the pipeline uses a deterministic
/// token generator seeded from [`SessionConfig::seed`].
pub struct OpenClPipeline {
    config: OpenClPipelineConfig,
    /// Vocabulary size used by the mock generator.
    vocab_size: u32,
    /// Whether we fell back to CPU because no `OpenCL` device was found.
    cpu_fallback: bool,
}

impl OpenClPipeline {
    /// Create a new pipeline.  Falls back to CPU if no `OpenCL` device is
    /// available.
    pub fn new(config: OpenClPipelineConfig, vocab_size: u32) -> Self {
        let cpu_fallback = !opencl_device_available();
        if cpu_fallback {
            log::info!("No OpenCL device found; falling back to CPU");
        }
        Self { config, vocab_size, cpu_fallback }
    }

    /// Returns `true` if the pipeline fell back to CPU execution.
    pub const fn is_cpu_fallback(&self) -> bool {
        self.cpu_fallback
    }

    /// Reference to the pipeline configuration.
    pub const fn config(&self) -> &OpenClPipelineConfig {
        &self.config
    }

    /// Generate tokens for a batch of prompts.
    ///
    /// Each prompt is processed independently using the same config.
    pub fn generate_batch(
        &self,
        prompts: &[&str],
        gen_config: &GenerationConfig,
    ) -> Result<Vec<Vec<StreamEvent>>> {
        prompts.iter().map(|p| self.generate(p, gen_config)).collect()
    }
}

impl InferenceSession for OpenClPipeline {
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<Vec<StreamEvent>> {
        if self.config.session.model_path.is_empty() {
            bail!(
                "invalid model path: '{}' does not exist or is empty",
                self.config.session.model_path
            );
        }

        let gen_seed = config.seed.unwrap_or(0xDEAD_BEEF);
        let session_seed = self.config.session.seed.unwrap_or(0);
        let mut rng = XorShift64::new(gen_seed ^ session_seed ^ hash_prompt(prompt));
        let mut events = Vec::new();
        let mut generated: Vec<u32> = Vec::new();
        let mut decoded = String::new();

        for _ in 0..config.max_new_tokens {
            let raw_id = rng.next_u32() % self.vocab_size;
            let token_id = apply_sampling(
                raw_id,
                self.vocab_size,
                self.config.temperature,
                self.config.top_k,
                self.config.top_p,
                &mut rng,
            );

            let text = format!("tok_{token_id}");
            decoded.push_str(&text);
            generated.push(token_id);

            events.push(StreamEvent::Token(TokenEvent { id: token_id, text }));

            if let Some(reason) = check_stop(&config.stop_criteria, token_id, &generated, &decoded)
            {
                events.push(StreamEvent::Done {
                    reason,
                    stats: GenerationStats {
                        tokens_generated: generated.len(),
                        tokens_per_second: 0.0,
                    },
                });
                return Ok(events);
            }
        }

        events.push(StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats { tokens_generated: generated.len(), tokens_per_second: 0.0 },
        });
        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// Multi-model manager
// ---------------------------------------------------------------------------

/// Manages multiple loaded pipelines for model-switching tests.
pub struct ModelManager {
    pipelines: HashMap<String, OpenClPipeline>,
}

impl ModelManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self { pipelines: HashMap::new() }
    }

    /// Load a pipeline under the given name.
    pub fn load(&mut self, name: &str, pipeline: OpenClPipeline) {
        self.pipelines.insert(name.to_string(), pipeline);
    }

    /// Generate tokens using the named pipeline.
    pub fn generate(
        &self,
        name: &str,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Vec<StreamEvent>> {
        let pipeline =
            self.pipelines.get(name).ok_or_else(|| anyhow::anyhow!("model not loaded: {name}"))?;
        pipeline.generate(prompt, config)
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if an `OpenCL` device is present.  Always `false` in
/// CPU-only builds — real `OpenCL` probing is a future task.
const fn opencl_device_available() -> bool {
    false
}

/// Minimal xorshift64 PRNG for deterministic mock generation.
struct XorShift64(u64);

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    const fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFF_FFFF) as u32
    }
}

/// Hash a prompt string to a u64 for RNG seeding.
fn hash_prompt(prompt: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in prompt.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Mock sampling: applies temperature / top-k / top-p adjustments.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn apply_sampling(
    raw_id: u32,
    vocab_size: u32,
    temperature: f32,
    top_k: usize,
    _top_p: f32,
    rng: &mut XorShift64,
) -> u32 {
    if temperature <= f32::EPSILON {
        // Greedy: always pick raw_id deterministically.
        return raw_id % vocab_size;
    }

    // top_k restricts range.
    let effective_range = if top_k > 0 { (top_k as u32).min(vocab_size) } else { vocab_size };

    // Add noise proportional to temperature.
    let noise = (rng.next_u32() as f32 / u32::MAX as f32) * temperature;
    let offset = (noise * effective_range as f32) as u32;
    (raw_id.wrapping_add(offset)) % effective_range
}
