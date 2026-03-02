//! Prefill/decode phase optimization for Intel Arc A770 GPUs.
//!
//! Separates inference into two phases — **prefill** (prompt processing) and
//! **decode** (autoregressive token generation) — so that kernel dispatch can
//! be tuned differently for each:
//!
//! * **Prefill** processes the entire prompt in parallel batches and benefits
//!   from wide work-groups and high occupancy.
//! * **Decode** generates one token at a time, reading from the KV cache, and
//!   benefits from low-latency single-wavefront dispatch.
//!
//! All compute paths have CPU reference implementations so that the logic can
//! be validated without OpenCL hardware.

use std::fmt;
use std::time::Instant;

// ── InferencePhase ─────────────────────────────────────────────────

/// Which phase of inference the engine is currently in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferencePhase {
    /// Prompt processing — all tokens processed in parallel batches.
    Prefill,
    /// Autoregressive generation — one token at a time, reading KV cache.
    Decode,
}

impl fmt::Display for InferencePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prefill => write!(f, "Prefill"),
            Self::Decode => write!(f, "Decode"),
        }
    }
}

// ── PhaseError ─────────────────────────────────────────────────────

/// Errors that can occur during phase scheduling or execution.
#[derive(Debug)]
pub enum PhaseError {
    /// Configuration value is out of range or inconsistent.
    InvalidConfig(String),
    /// Attempted an operation that is not valid in the current phase.
    WrongPhase { expected: InferencePhase, actual: InferencePhase },
    /// A kernel execution failed.
    KernelFailure(String),
    /// The prompt is empty (zero tokens).
    EmptyPrompt,
}

impl fmt::Display for PhaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::WrongPhase { expected, actual } => {
                write!(f, "expected phase {expected}, but in {actual}")
            }
            Self::KernelFailure(msg) => write!(f, "kernel failure: {msg}"),
            Self::EmptyPrompt => write!(f, "empty prompt"),
        }
    }
}

impl std::error::Error for PhaseError {}

// ── PrefillConfig ──────────────────────────────────────────────────

/// Tuning knobs for the prefill (prompt-processing) phase.
#[derive(Debug, Clone)]
pub struct PrefillConfig {
    /// Maximum tokens processed per chunk (long prompts are split).
    pub chunk_size: usize,
    /// Upper bound on total prompt tokens to accept.
    pub max_batch_tokens: usize,
    /// Number of attention heads processed in parallel.
    pub parallel_heads: usize,
}

impl Default for PrefillConfig {
    fn default() -> Self {
        Self { chunk_size: 512, max_batch_tokens: 4096, parallel_heads: 32 }
    }
}

impl PrefillConfig {
    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), PhaseError> {
        if self.chunk_size == 0 {
            return Err(PhaseError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if self.max_batch_tokens == 0 {
            return Err(PhaseError::InvalidConfig("max_batch_tokens must be > 0".into()));
        }
        if self.parallel_heads == 0 {
            return Err(PhaseError::InvalidConfig("parallel_heads must be > 0".into()));
        }
        Ok(())
    }

    /// Number of chunks needed for a prompt of `token_count` tokens.
    pub fn num_chunks(&self, token_count: usize) -> usize {
        if token_count == 0 {
            return 0;
        }
        token_count.div_ceil(self.chunk_size)
    }
}

// ── DecodeConfig ───────────────────────────────────────────────────

/// Tuning knobs for the decode (token-generation) phase.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// KV cache memory budget in megabytes.
    pub kv_cache_budget_mb: usize,
    /// Number of speculative tokens to draft before verification.
    pub speculative_tokens: usize,
    /// Allow the decoder to exit early when an EOS token is produced.
    pub early_exit: bool,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self { kv_cache_budget_mb: 512, speculative_tokens: 0, early_exit: true }
    }
}

impl DecodeConfig {
    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), PhaseError> {
        if self.kv_cache_budget_mb == 0 {
            return Err(PhaseError::InvalidConfig("kv_cache_budget_mb must be > 0".into()));
        }
        Ok(())
    }

    /// Maximum number of KV entries that fit in the budget, given
    /// `bytes_per_entry` (key + value size for one position per layer).
    pub fn max_kv_entries(&self, bytes_per_entry: usize) -> usize {
        if bytes_per_entry == 0 {
            return 0;
        }
        (self.kv_cache_budget_mb * 1024 * 1024) / bytes_per_entry
    }
}

// ── PhaseTransition ────────────────────────────────────────────────

/// Snapshot of the system state at the moment the scheduler switches
/// from prefill to decode.
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// Total prompt tokens that were processed during prefill.
    pub tokens_processed: usize,
    /// How much of the KV cache was filled (0.0–1.0).
    pub kv_cache_filled: f64,
    /// Wall-clock latency of the transition itself, in microseconds.
    pub transition_latency_us: u64,
}

// ── PhaseBenchmark ─────────────────────────────────────────────────

/// End-of-generation performance summary.
#[derive(Debug, Clone)]
pub struct PhaseBenchmark {
    /// Prefill throughput (prompt tokens / prefill wall-time).
    pub prefill_tok_per_s: f64,
    /// Decode throughput (generated tokens / decode wall-time).
    pub decode_tok_per_s: f64,
    /// Time-to-first-token in milliseconds (prefill latency).
    pub ttft_ms: f64,
}

impl fmt::Display for PhaseBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prefill {:.1} tok/s | decode {:.1} tok/s | TTFT {:.2} ms",
            self.prefill_tok_per_s, self.decode_tok_per_s, self.ttft_ms,
        )
    }
}

// ── CPU reference: prefill ─────────────────────────────────────────

/// Process a single prefill chunk on the CPU.
///
/// Simulates batch-parallel hidden-state computation: every token in
/// `chunk` is mapped through a toy linear projection and accumulated.
/// Returns one f32 per token (a scalar summary of the hidden state).
pub fn cpu_prefill_chunk(chunk: &[u32], hidden_dim: usize) -> Vec<f32> {
    chunk
        .iter()
        .map(|&tok| {
            // Toy projection: hash the token into a pseudo hidden state.
            let h: f32 = (0..hidden_dim)
                .map(|d| {
                    let x = ((tok as u64).wrapping_mul(2654435761) ^ (d as u64).wrapping_mul(40503))
                        as f32;
                    x / (u32::MAX as f32)
                })
                .sum();
            h / hidden_dim as f32
        })
        .collect()
}

/// Run the full prefill phase on the CPU, chunking the prompt according
/// to `config`. Returns the concatenated hidden-state scalars.
pub fn cpu_prefill(
    prompt_tokens: &[u32],
    config: &PrefillConfig,
    hidden_dim: usize,
) -> Result<Vec<f32>, PhaseError> {
    config.validate()?;
    if prompt_tokens.is_empty() {
        return Err(PhaseError::EmptyPrompt);
    }
    if prompt_tokens.len() > config.max_batch_tokens {
        return Err(PhaseError::InvalidConfig(format!(
            "prompt length {} exceeds max_batch_tokens {}",
            prompt_tokens.len(),
            config.max_batch_tokens,
        )));
    }

    let mut out = Vec::with_capacity(prompt_tokens.len());
    for chunk in prompt_tokens.chunks(config.chunk_size) {
        out.extend(cpu_prefill_chunk(chunk, hidden_dim));
    }
    Ok(out)
}

// ── CPU reference: decode ──────────────────────────────────────────

/// One autoregressive decode step on the CPU.
///
/// Takes the most recent token, the accumulated KV cache (as a flat
/// `f32` vector with `cache_len` entries of `hidden_dim` each), and
/// returns a logit vector of size `vocab_size`.
pub fn cpu_decode_step(
    token: u32,
    kv_cache: &[f32],
    cache_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    // Toy attention over KV cache entries.
    let query: f32 = (0..hidden_dim)
        .map(|d| {
            let x =
                ((token as u64).wrapping_mul(2654435761) ^ (d as u64).wrapping_mul(40503)) as f32;
            x / (u32::MAX as f32)
        })
        .sum::<f32>()
        / hidden_dim as f32;

    let actual_entries = if hidden_dim > 0 {
        (kv_cache.len() / hidden_dim).min(cache_len)
    } else {
        0
    };

    let attn_score: f32 = if actual_entries > 0 {
        (0..actual_entries)
            .map(|i| {
                let base = i * hidden_dim;
                let kv_sum: f32 = kv_cache[base..base + hidden_dim].iter().sum();
                (kv_sum / hidden_dim as f32) * query
            })
            .sum::<f32>()
            / actual_entries as f32
    } else {
        query
    };

    // Project to vocab logits.
    (0..vocab_size)
        .map(|v| {
            let seed = (v as u64).wrapping_mul(2654435761) ^ token as u64;
            attn_score + (seed as f32 / u64::MAX as f32)
        })
        .collect()
}

// ── PrefillOptimizer ───────────────────────────────────────────────

/// Decides work-group sizing and memory layout for prefill dispatch.
#[derive(Debug)]
pub struct PrefillOptimizer {
    config: PrefillConfig,
    hidden_dim: usize,
}

impl PrefillOptimizer {
    pub fn new(config: PrefillConfig, hidden_dim: usize) -> Self {
        Self { config, hidden_dim }
    }

    /// Optimal work-group size for a chunk of `n` tokens.
    ///
    /// Heuristic: min(parallel_heads, next power-of-two of n), clamped
    /// to 256 (A770 max workgroup).
    pub fn work_group_size(&self, n: usize) -> usize {
        let next_pow2 = n.next_power_of_two();
        next_pow2.min(self.config.parallel_heads).clamp(1, 256)
    }

    /// Execute the prefill phase on the CPU and return hidden-state
    /// scalars.
    pub fn execute_cpu(&self, prompt_tokens: &[u32]) -> Result<Vec<f32>, PhaseError> {
        cpu_prefill(prompt_tokens, &self.config, self.hidden_dim)
    }
}

// ── DecodeOptimizer ────────────────────────────────────────────────

/// Decides dispatch parameters for the decode phase.
#[derive(Debug)]
pub struct DecodeOptimizer {
    config: DecodeConfig,
    hidden_dim: usize,
    vocab_size: usize,
}

impl DecodeOptimizer {
    pub fn new(config: DecodeConfig, hidden_dim: usize, vocab_size: usize) -> Self {
        Self { config, hidden_dim, vocab_size }
    }

    /// Whether speculative decoding is enabled.
    pub fn is_speculative(&self) -> bool {
        self.config.speculative_tokens > 0
    }

    /// Execute one decode step on the CPU and return logits.
    pub fn execute_cpu_step(&self, token: u32, kv_cache: &[f32], cache_len: usize) -> Vec<f32> {
        cpu_decode_step(token, kv_cache, cache_len, self.hidden_dim, self.vocab_size)
    }
}

// ── PhaseScheduler ─────────────────────────────────────────────────

/// Manages transitions between prefill and decode phases.
#[derive(Debug)]
pub struct PhaseScheduler {
    phase: InferencePhase,
    prefill_optimizer: PrefillOptimizer,
    decode_optimizer: DecodeOptimizer,
    /// KV cache budget expressed as max entries.
    max_kv_entries: usize,
    /// Number of KV entries currently occupied.
    kv_entries_used: usize,
    /// Recorded phase transition (set once prefill completes).
    transition: Option<PhaseTransition>,
    /// Wall-clock instant when the scheduler was created (or prefill
    /// started).
    prefill_start: Instant,
    /// Instant when the first decode token was produced.
    decode_start: Option<Instant>,
    /// Total prompt tokens processed.
    prompt_tokens_processed: usize,
    /// Total tokens generated during decode.
    tokens_generated: usize,
}

impl PhaseScheduler {
    /// Create a new scheduler in the [`InferencePhase::Prefill`] state.
    ///
    /// `bytes_per_kv_entry` is the memory footprint of one KV-cache
    /// position across all layers (key + value).
    pub fn new(
        prefill_config: PrefillConfig,
        decode_config: DecodeConfig,
        hidden_dim: usize,
        vocab_size: usize,
        bytes_per_kv_entry: usize,
    ) -> Self {
        let max_kv = decode_config.max_kv_entries(bytes_per_kv_entry);
        Self {
            phase: InferencePhase::Prefill,
            prefill_optimizer: PrefillOptimizer::new(prefill_config, hidden_dim),
            decode_optimizer: DecodeOptimizer::new(decode_config, hidden_dim, vocab_size),
            max_kv_entries: max_kv,
            kv_entries_used: 0,
            transition: None,
            prefill_start: Instant::now(),
            decode_start: None,
            prompt_tokens_processed: 0,
            tokens_generated: 0,
        }
    }

    /// Current phase.
    pub fn phase(&self) -> InferencePhase {
        self.phase
    }

    /// Run the prefill phase on the CPU.
    ///
    /// On success the scheduler transitions to `Decode` and records a
    /// [`PhaseTransition`].
    pub fn run_prefill_cpu(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>, PhaseError> {
        if self.phase != InferencePhase::Prefill {
            return Err(PhaseError::WrongPhase {
                expected: InferencePhase::Prefill,
                actual: self.phase,
            });
        }

        let result = self.prefill_optimizer.execute_cpu(prompt_tokens)?;

        // Record prompt token count and fill KV cache.
        self.prompt_tokens_processed = prompt_tokens.len();
        self.kv_entries_used = prompt_tokens.len().min(self.max_kv_entries);

        // Transition to decode.
        let transition_start = Instant::now();
        self.phase = InferencePhase::Decode;
        let transition_latency_us = transition_start.elapsed().as_micros() as u64;

        self.transition = Some(PhaseTransition {
            tokens_processed: self.prompt_tokens_processed,
            kv_cache_filled: if self.max_kv_entries > 0 {
                self.kv_entries_used as f64 / self.max_kv_entries as f64
            } else {
                0.0
            },
            transition_latency_us,
        });

        Ok(result)
    }

    /// Run one decode step on the CPU.
    ///
    /// Returns logits for the next token. The caller is responsible for
    /// sampling.
    pub fn run_decode_step_cpu(
        &mut self,
        token: u32,
        kv_cache: &[f32],
    ) -> Result<Vec<f32>, PhaseError> {
        if self.phase != InferencePhase::Decode {
            return Err(PhaseError::WrongPhase {
                expected: InferencePhase::Decode,
                actual: self.phase,
            });
        }

        if self.decode_start.is_none() {
            self.decode_start = Some(Instant::now());
        }

        let logits = self.decode_optimizer.execute_cpu_step(token, kv_cache, self.kv_entries_used);

        self.tokens_generated += 1;

        // Update KV entry count (one new entry per step).
        if self.kv_entries_used < self.max_kv_entries {
            self.kv_entries_used += 1;
        }

        Ok(logits)
    }

    /// Whether the decoder should stop (early exit on EOS or KV cache
    /// exhaustion).
    pub fn should_stop(&self, last_token: u32, eos_token: u32) -> bool {
        if self.decode_optimizer.config.early_exit && last_token == eos_token {
            return true;
        }
        self.kv_entries_used >= self.max_kv_entries
    }

    /// Retrieve the phase transition record (available after prefill).
    pub fn transition(&self) -> Option<&PhaseTransition> {
        self.transition.as_ref()
    }

    /// Build a [`PhaseBenchmark`] from the recorded timings.
    pub fn benchmark(&self) -> PhaseBenchmark {
        let prefill_elapsed = match self.decode_start {
            Some(ds) => ds.duration_since(self.prefill_start),
            None => self.prefill_start.elapsed(),
        };
        let prefill_secs = prefill_elapsed.as_secs_f64().max(1e-9);
        let prefill_tok_per_s = self.prompt_tokens_processed as f64 / prefill_secs;

        let decode_secs =
            self.decode_start.map(|ds| ds.elapsed().as_secs_f64().max(1e-9)).unwrap_or(1e-9);
        let decode_tok_per_s = self.tokens_generated as f64 / decode_secs;

        let ttft_ms = prefill_elapsed.as_secs_f64() * 1000.0;

        PhaseBenchmark { prefill_tok_per_s, decode_tok_per_s, ttft_ms }
    }

    /// Total tokens processed (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens_processed + self.tokens_generated
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn default_prefill_config() -> PrefillConfig {
        PrefillConfig::default()
    }

    fn default_decode_config() -> DecodeConfig {
        DecodeConfig::default()
    }

    fn make_scheduler() -> PhaseScheduler {
        PhaseScheduler::new(
            default_prefill_config(),
            default_decode_config(),
            64,   // hidden_dim
            100,  // vocab_size
            1024, // bytes_per_kv_entry
        )
    }

    fn sample_prompt(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    // ── InferencePhase ────────────────────────────────────────────

    #[test]
    fn phase_display_prefill() {
        assert_eq!(InferencePhase::Prefill.to_string(), "Prefill");
    }

    #[test]
    fn phase_display_decode() {
        assert_eq!(InferencePhase::Decode.to_string(), "Decode");
    }

    #[test]
    fn phase_eq_and_clone() {
        let a = InferencePhase::Prefill;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, InferencePhase::Decode);
    }

    // ── PrefillConfig ─────────────────────────────────────────────

    #[test]
    fn prefill_config_defaults() {
        let cfg = PrefillConfig::default();
        assert_eq!(cfg.chunk_size, 512);
        assert_eq!(cfg.max_batch_tokens, 4096);
        assert_eq!(cfg.parallel_heads, 32);
    }

    #[test]
    fn prefill_config_validate_ok() {
        assert!(default_prefill_config().validate().is_ok());
    }

    #[test]
    fn prefill_config_zero_chunk_size() {
        let cfg = PrefillConfig { chunk_size: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn prefill_config_zero_max_batch() {
        let cfg = PrefillConfig { max_batch_tokens: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn prefill_config_zero_parallel_heads() {
        let cfg = PrefillConfig { parallel_heads: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn prefill_num_chunks_exact_division() {
        let cfg = PrefillConfig { chunk_size: 4, ..Default::default() };
        assert_eq!(cfg.num_chunks(8), 2);
    }

    #[test]
    fn prefill_num_chunks_remainder() {
        let cfg = PrefillConfig { chunk_size: 4, ..Default::default() };
        assert_eq!(cfg.num_chunks(9), 3);
    }

    #[test]
    fn prefill_num_chunks_zero_tokens() {
        assert_eq!(default_prefill_config().num_chunks(0), 0);
    }

    #[test]
    fn prefill_num_chunks_single_token() {
        let cfg = PrefillConfig { chunk_size: 512, ..Default::default() };
        assert_eq!(cfg.num_chunks(1), 1);
    }

    // ── DecodeConfig ──────────────────────────────────────────────

    #[test]
    fn decode_config_defaults() {
        let cfg = DecodeConfig::default();
        assert_eq!(cfg.kv_cache_budget_mb, 512);
        assert_eq!(cfg.speculative_tokens, 0);
        assert!(cfg.early_exit);
    }

    #[test]
    fn decode_config_validate_ok() {
        assert!(default_decode_config().validate().is_ok());
    }

    #[test]
    fn decode_config_zero_budget() {
        let cfg = DecodeConfig { kv_cache_budget_mb: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn decode_max_kv_entries() {
        let cfg = DecodeConfig { kv_cache_budget_mb: 1, ..Default::default() };
        // 1 MB = 1048576 bytes, entry = 1024 bytes → 1024 entries
        assert_eq!(cfg.max_kv_entries(1024), 1024);
    }

    #[test]
    fn decode_max_kv_entries_zero_entry_size() {
        assert_eq!(default_decode_config().max_kv_entries(0), 0);
    }

    // ── PhaseTransition ───────────────────────────────────────────

    #[test]
    fn phase_transition_fields() {
        let t = PhaseTransition {
            tokens_processed: 128,
            kv_cache_filled: 0.5,
            transition_latency_us: 42,
        };
        assert_eq!(t.tokens_processed, 128);
        assert!((t.kv_cache_filled - 0.5).abs() < f64::EPSILON);
        assert_eq!(t.transition_latency_us, 42);
    }

    // ── PhaseBenchmark ────────────────────────────────────────────

    #[test]
    fn phase_benchmark_display() {
        let b = PhaseBenchmark { prefill_tok_per_s: 1000.0, decode_tok_per_s: 50.0, ttft_ms: 12.5 };
        let s = b.to_string();
        assert!(s.contains("1000.0 tok/s"));
        assert!(s.contains("50.0 tok/s"));
        assert!(s.contains("12.50 ms"));
    }

    // ── CPU prefill reference ─────────────────────────────────────

    #[test]
    fn cpu_prefill_chunk_deterministic() {
        let a = cpu_prefill_chunk(&[1, 2, 3], 64);
        let b = cpu_prefill_chunk(&[1, 2, 3], 64);
        assert_eq!(a, b);
    }

    #[test]
    fn cpu_prefill_chunk_length() {
        let out = cpu_prefill_chunk(&[10, 20, 30, 40], 32);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn cpu_prefill_chunk_empty() {
        let out = cpu_prefill_chunk(&[], 64);
        assert!(out.is_empty());
    }

    #[test]
    fn cpu_prefill_full_prompt() {
        let cfg = PrefillConfig { chunk_size: 4, max_batch_tokens: 100, parallel_heads: 8 };
        let result = cpu_prefill(&[1, 2, 3, 4, 5], &cfg, 32).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn cpu_prefill_empty_prompt_error() {
        let cfg = default_prefill_config();
        let err = cpu_prefill(&[], &cfg, 32);
        assert!(err.is_err());
    }

    #[test]
    fn cpu_prefill_exceeds_max_batch() {
        let cfg = PrefillConfig { max_batch_tokens: 3, ..Default::default() };
        let err = cpu_prefill(&[1, 2, 3, 4], &cfg, 32);
        assert!(err.is_err());
    }

    #[test]
    fn cpu_prefill_chunking_preserves_all_tokens() {
        let cfg = PrefillConfig { chunk_size: 3, max_batch_tokens: 100, parallel_heads: 4 };
        let prompt: Vec<u32> = (0..10).collect();
        let result = cpu_prefill(&prompt, &cfg, 16).unwrap();
        // Every prompt token produces exactly one output scalar.
        assert_eq!(result.len(), prompt.len());
    }

    #[test]
    fn cpu_prefill_single_token() {
        let cfg = default_prefill_config();
        let result = cpu_prefill(&[42], &cfg, 64).unwrap();
        assert_eq!(result.len(), 1);
    }

    // ── CPU decode reference ──────────────────────────────────────

    #[test]
    fn cpu_decode_step_logit_length() {
        let logits = cpu_decode_step(7, &[], 0, 64, 100);
        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn cpu_decode_step_with_kv_cache() {
        let kv: Vec<f32> = vec![0.1; 64 * 3]; // 3 entries
        let logits = cpu_decode_step(5, &kv, 3, 64, 50);
        assert_eq!(logits.len(), 50);
    }

    #[test]
    fn cpu_decode_step_deterministic() {
        let kv: Vec<f32> = vec![0.5; 64 * 2];
        let a = cpu_decode_step(10, &kv, 2, 64, 50);
        let b = cpu_decode_step(10, &kv, 2, 64, 50);
        assert_eq!(a, b);
    }

    // ── PrefillOptimizer ──────────────────────────────────────────

    #[test]
    fn prefill_optimizer_work_group_size_small() {
        let opt =
            PrefillOptimizer::new(PrefillConfig { parallel_heads: 32, ..Default::default() }, 64);
        // n=3 → next_pow2=4, min(4,32)=4, clamped ≤256 → 4
        assert_eq!(opt.work_group_size(3), 4);
    }

    #[test]
    fn prefill_optimizer_work_group_size_large() {
        let opt =
            PrefillOptimizer::new(PrefillConfig { parallel_heads: 32, ..Default::default() }, 64);
        // n=1024 → next_pow2=1024, min(1024,32)=32, ≤256 → 32
        assert_eq!(opt.work_group_size(1024), 32);
    }

    #[test]
    fn prefill_optimizer_work_group_clamp_256() {
        let opt =
            PrefillOptimizer::new(PrefillConfig { parallel_heads: 1024, ..Default::default() }, 64);
        // n=512 → next_pow2=512, min(512,1024)=512, clamp 256
        assert_eq!(opt.work_group_size(512), 256);
    }

    #[test]
    fn prefill_optimizer_execute_cpu() {
        let opt = PrefillOptimizer::new(default_prefill_config(), 32);
        let out = opt.execute_cpu(&[1, 2, 3]).unwrap();
        assert_eq!(out.len(), 3);
    }

    // ── DecodeOptimizer ───────────────────────────────────────────

    #[test]
    fn decode_optimizer_not_speculative_by_default() {
        let opt = DecodeOptimizer::new(default_decode_config(), 64, 100);
        assert!(!opt.is_speculative());
    }

    #[test]
    fn decode_optimizer_speculative() {
        let cfg = DecodeConfig { speculative_tokens: 4, ..Default::default() };
        let opt = DecodeOptimizer::new(cfg, 64, 100);
        assert!(opt.is_speculative());
    }

    #[test]
    fn decode_optimizer_execute_cpu_step() {
        let opt = DecodeOptimizer::new(default_decode_config(), 64, 100);
        let logits = opt.execute_cpu_step(42, &[], 0);
        assert_eq!(logits.len(), 100);
    }

    // ── PhaseScheduler ────────────────────────────────────────────

    #[test]
    fn scheduler_starts_in_prefill() {
        let s = make_scheduler();
        assert_eq!(s.phase(), InferencePhase::Prefill);
    }

    #[test]
    fn scheduler_transitions_to_decode() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(8)).unwrap();
        assert_eq!(s.phase(), InferencePhase::Decode);
    }

    #[test]
    fn scheduler_prefill_records_transition() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(16)).unwrap();
        let t = s.transition().expect("transition should be recorded");
        assert_eq!(t.tokens_processed, 16);
        assert!(t.kv_cache_filled >= 0.0 && t.kv_cache_filled <= 1.0);
    }

    #[test]
    fn scheduler_cannot_prefill_twice() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(4)).unwrap();
        let err = s.run_prefill_cpu(&sample_prompt(4));
        assert!(err.is_err());
    }

    #[test]
    fn scheduler_cannot_decode_before_prefill() {
        let mut s = make_scheduler();
        let err = s.run_decode_step_cpu(0, &[]);
        assert!(err.is_err());
    }

    #[test]
    fn scheduler_decode_step() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(4)).unwrap();
        let logits = s.run_decode_step_cpu(99, &[]).unwrap();
        assert_eq!(logits.len(), 100);
        assert_eq!(s.tokens_generated, 1);
    }

    #[test]
    fn scheduler_multiple_decode_steps() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(4)).unwrap();
        for i in 0..5 {
            let _ = s.run_decode_step_cpu(i, &[]).unwrap();
        }
        assert_eq!(s.tokens_generated, 5);
    }

    #[test]
    fn scheduler_total_tokens() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(10)).unwrap();
        let _ = s.run_decode_step_cpu(0, &[]).unwrap();
        let _ = s.run_decode_step_cpu(1, &[]).unwrap();
        assert_eq!(s.total_tokens(), 12);
    }

    #[test]
    fn scheduler_should_stop_on_eos() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(2)).unwrap();
        assert!(s.should_stop(42, 42)); // last_token == eos
    }

    #[test]
    fn scheduler_should_not_stop_non_eos() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(2)).unwrap();
        assert!(!s.should_stop(7, 42));
    }

    #[test]
    fn scheduler_benchmark_after_prefill() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(32)).unwrap();
        let b = s.benchmark();
        assert!(b.prefill_tok_per_s > 0.0);
        assert!(b.ttft_ms >= 0.0);
    }

    #[test]
    fn scheduler_benchmark_after_decode() {
        let mut s = make_scheduler();
        let _ = s.run_prefill_cpu(&sample_prompt(16)).unwrap();
        for i in 0..4 {
            let _ = s.run_decode_step_cpu(i, &[]).unwrap();
        }
        let b = s.benchmark();
        assert!(b.prefill_tok_per_s > 0.0);
        assert!(b.decode_tok_per_s > 0.0);
        assert!(b.ttft_ms >= 0.0);
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn scheduler_prefill_empty_prompt_error() {
        let mut s = make_scheduler();
        assert!(s.run_prefill_cpu(&[]).is_err());
        // Should still be in Prefill phase.
        assert_eq!(s.phase(), InferencePhase::Prefill);
    }

    #[test]
    fn scheduler_max_context_length() {
        let cfg = PrefillConfig { chunk_size: 128, max_batch_tokens: 4096, parallel_heads: 16 };
        let mut s = PhaseScheduler::new(cfg, default_decode_config(), 32, 50, 512);
        let prompt = sample_prompt(4096);
        let out = s.run_prefill_cpu(&prompt).unwrap();
        assert_eq!(out.len(), 4096);
    }

    #[test]
    fn scheduler_single_token_prompt() {
        let mut s = make_scheduler();
        let out = s.run_prefill_cpu(&[7]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(s.phase(), InferencePhase::Decode);
    }

    // ── Property-style tests ──────────────────────────────────────

    #[test]
    fn property_prefill_output_count_matches_input() {
        for n in [1, 2, 7, 64, 100, 511, 512, 513, 1024] {
            let cfg = PrefillConfig { chunk_size: 64, max_batch_tokens: 2048, parallel_heads: 8 };
            let prompt = sample_prompt(n);
            let out = cpu_prefill(&prompt, &cfg, 16).unwrap();
            assert_eq!(out.len(), n, "mismatch for prompt len {n}");
        }
    }

    #[test]
    fn property_transition_preserves_token_count() {
        for n in [1, 5, 32, 128, 256] {
            let mut s = PhaseScheduler::new(
                PrefillConfig { chunk_size: 16, max_batch_tokens: 1024, parallel_heads: 4 },
                default_decode_config(),
                16,
                50,
                256,
            );
            let prompt = sample_prompt(n);
            let _ = s.run_prefill_cpu(&prompt).unwrap();
            let t = s.transition().unwrap();
            assert_eq!(t.tokens_processed, n, "transition lost tokens for n={n}");
        }
    }

    #[test]
    fn property_kv_cache_fill_bounded() {
        for n in [1, 64, 512, 2048] {
            let mut s = PhaseScheduler::new(
                PrefillConfig { chunk_size: 64, max_batch_tokens: 4096, parallel_heads: 4 },
                default_decode_config(),
                16,
                50,
                256,
            );
            let prompt = sample_prompt(n);
            let _ = s.run_prefill_cpu(&prompt).unwrap();
            let t = s.transition().unwrap();
            assert!(
                t.kv_cache_filled >= 0.0 && t.kv_cache_filled <= 1.0,
                "KV fill out of [0,1] for n={n}: {}",
                t.kv_cache_filled
            );
        }
    }

    #[test]
    fn property_decode_never_exceeds_kv_budget() {
        let mut s = PhaseScheduler::new(
            PrefillConfig { chunk_size: 4, max_batch_tokens: 100, parallel_heads: 2 },
            DecodeConfig { kv_cache_budget_mb: 1, speculative_tokens: 0, early_exit: false },
            8,
            10,
            // bytes_per_kv_entry = 1 MB → max_kv_entries = 1
            1024 * 1024,
        );
        let _ = s.run_prefill_cpu(&[1]).unwrap();
        // Attempt many decode steps — kv_entries_used should never
        // exceed max_kv_entries.
        for i in 0..20 {
            let _ = s.run_decode_step_cpu(i, &[]).unwrap();
        }
        assert!(s.kv_entries_used <= s.max_kv_entries);
    }

    // ── PhaseError Display ────────────────────────────────────────

    #[test]
    fn phase_error_display_invalid_config() {
        let e = PhaseError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn phase_error_display_wrong_phase() {
        let e = PhaseError::WrongPhase {
            expected: InferencePhase::Prefill,
            actual: InferencePhase::Decode,
        };
        let s = e.to_string();
        assert!(s.contains("Prefill"));
        assert!(s.contains("Decode"));
    }

    #[test]
    fn phase_error_display_empty_prompt() {
        let e = PhaseError::EmptyPrompt;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn phase_error_display_kernel_failure() {
        let e = PhaseError::KernelFailure("boom".into());
        assert!(e.to_string().contains("boom"));
    }
}
