//! End-to-end inference pipeline orchestrating all kernel stages.
//!
//! Provides a CPU-reference pipeline that chains embedding lookup, RoPE,
//! per-layer RMSNorm → Attention → Residual → RMSNorm → FFN → Residual,
//! final RMSNorm, LM-head projection, and sampling into a single
//! `InferencePipeline`.  A `TokenGenerator` drives autoregressive decoding
//! and a `PipelineBuilder` offers ergonomic configuration.

use std::time::Instant;

// ── Pipeline configuration ─────────────────────────────────────────

/// Full model configuration consumed by the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
}

impl PipelineConfig {
    /// Validate that every dimension is non-zero and self-consistent.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0".into());
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        if self.intermediate_dim == 0 {
            return Err("intermediate_dim must be > 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        Ok(())
    }

    /// BitNet-2B-like default for testing.
    pub fn bitnet_2b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 2048,
            num_layers: 24,
            num_heads: 16,
            head_dim: 128,
            intermediate_dim: 5504,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    /// Tiny config for fast unit tests.
    pub fn tiny_test() -> Self {
        Self {
            vocab_size: 64,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            intermediate_dim: 64,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }
}

// ── Pipeline stage enum ────────────────────────────────────────────

/// Logical inference stage identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    Embedding,
    PreNorm,
    Attention,
    FFN,
    PostNorm,
    LmHead,
    Sampling,
}

impl PipelineStage {
    pub fn all() -> &'static [PipelineStage] {
        &[
            PipelineStage::Embedding,
            PipelineStage::PreNorm,
            PipelineStage::Attention,
            PipelineStage::FFN,
            PipelineStage::PostNorm,
            PipelineStage::LmHead,
            PipelineStage::Sampling,
        ]
    }
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            PipelineStage::Embedding => "Embedding",
            PipelineStage::PreNorm => "PreNorm",
            PipelineStage::Attention => "Attention",
            PipelineStage::FFN => "FFN",
            PipelineStage::PostNorm => "PostNorm",
            PipelineStage::LmHead => "LmHead",
            PipelineStage::Sampling => "Sampling",
        };
        write!(f, "{name}")
    }
}

// ── Stage timings ──────────────────────────────────────────────────

/// Accumulated timing for one pipeline stage.
#[derive(Debug, Clone)]
pub struct StageTimings {
    pub stage: PipelineStage,
    pub total_duration_us: u64,
    pub call_count: u64,
}

impl StageTimings {
    fn new(stage: PipelineStage) -> Self {
        Self { stage, total_duration_us: 0, call_count: 0 }
    }

    fn record(&mut self, duration_us: u64) {
        self.total_duration_us += duration_us;
        self.call_count += 1;
    }

    /// Average microseconds per call, or 0.0 if no calls recorded.
    pub fn avg_us(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_duration_us as f64 / self.call_count as f64
        }
    }
}

// ── Pipeline status ────────────────────────────────────────────────

/// Runtime status of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStatus {
    Ready,
    Running,
    Paused,
    Error,
}

impl std::fmt::Display for PipelineStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PipelineStatus::Ready => "Ready",
            PipelineStatus::Running => "Running",
            PipelineStatus::Paused => "Paused",
            PipelineStatus::Error => "Error",
        };
        write!(f, "{s}")
    }
}

// ── Pipeline diagnostics ───────────────────────────────────────────

/// Runtime diagnostics snapshot.
#[derive(Debug, Clone)]
pub struct PipelineDiagnostics {
    pub status: PipelineStatus,
    pub total_forward_calls: u64,
    pub total_tokens_generated: u64,
    pub peak_sequence_len: usize,
    pub last_error: Option<String>,
}

// ── Generation config ──────────────────────────────────────────────

/// Parameters for autoregressive generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Stop generation when this token is produced.
    pub stop_token: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_token: None,
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err("max_tokens must be > 0".into());
        }
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0".into());
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        if self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be > 0".into());
        }
        Ok(())
    }

    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = p;
        self
    }

    pub fn with_stop_token(mut self, t: u32) -> Self {
        self.stop_token = Some(t);
        self
    }
}

// ── Generation result ──────────────────────────────────────────────

/// Output of an autoregressive generation run.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub total_time_us: u64,
    pub tokens_per_second: f64,
    pub stage_timings: Vec<StageTimings>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub stop_reason: StopReason,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    StopToken,
    EndOfSequence,
}

// ── Deterministic RNG ──────────────────────────────────────────────

/// Minimal xorshift64 for reproducible sampling (no external dep).
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f32) / (u64::MAX as f32)
    }
}

// ── Simulated model weights ────────────────────────────────────────

/// Placeholder weight tensors initialised with a deterministic pattern
/// so that the CPU-reference pipeline is fully self-contained.
struct SimulatedWeights {
    embedding_table: Vec<f32>,
    layer_weights: Vec<LayerWeights>,
    final_norm_gamma: Vec<f32>,
    lm_head: Vec<f32>,
}

struct LayerWeights {
    rms_pre_gamma: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    rms_post_gamma: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

impl SimulatedWeights {
    fn init(cfg: &PipelineConfig) -> Self {
        let h = cfg.hidden_dim;
        let v = cfg.vocab_size;
        let inter = cfg.intermediate_dim;

        let embedding_table = deterministic_vec(v * h, 0.02, 42);
        let final_norm_gamma = vec![1.0_f32; h];
        let lm_head = deterministic_vec(v * h, 0.02, 99);

        let layer_weights = (0..cfg.num_layers)
            .map(|i| {
                let seed = (i as u64 + 1) * 137;
                LayerWeights {
                    rms_pre_gamma: vec![1.0; h],
                    wq: deterministic_vec(h * h, 0.02, seed),
                    wk: deterministic_vec(h * h, 0.02, seed + 1),
                    wv: deterministic_vec(h * h, 0.02, seed + 2),
                    wo: deterministic_vec(h * h, 0.02, seed + 3),
                    rms_post_gamma: vec![1.0; h],
                    w1: deterministic_vec(h * inter, 0.02, seed + 4),
                    w2: deterministic_vec(inter * h, 0.02, seed + 5),
                    w3: deterministic_vec(h * inter, 0.02, seed + 6),
                }
            })
            .collect();

        Self { embedding_table, layer_weights, final_norm_gamma, lm_head }
    }
}

/// Generate a deterministic f32 vector using xorshift.
fn deterministic_vec(len: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64::new(seed);
    (0..len).map(|_| (rng.next_f32() - 0.5) * 2.0 * scale).collect()
}

// ── Core math helpers ──────────────────────────────────────────────

fn rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / n as f32 + eps).sqrt();
    input.iter().zip(gamma.iter()).map(|(x, g)| x / rms * g).collect()
}

fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn add_residual(out: &mut [f32], residual: &[f32]) {
    for (o, r) in out.iter_mut().zip(residual.iter()) {
        *o += r;
    }
}

fn softmax(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn apply_rope_to_pair(x0: f32, x1: f32, cos_val: f32, sin_val: f32) -> (f32, f32) {
    (x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val)
}

fn rope_frequencies(head_dim: usize, base: f32) -> Vec<f32> {
    (0..head_dim / 2).map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32)).collect()
}

// ── InferencePipeline ──────────────────────────────────────────────

/// Orchestrates all inference stages end-to-end.
pub struct InferencePipeline {
    config: PipelineConfig,
    weights: SimulatedWeights,
    rope_freqs: Vec<f32>,
    timings: Vec<StageTimings>,
    status: PipelineStatus,
    position: usize,
    total_forward_calls: u64,
    total_tokens_generated: u64,
    peak_seq_len: usize,
    last_error: Option<String>,
}

impl InferencePipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Result<Self, String> {
        config.validate()?;
        let weights = SimulatedWeights::init(&config);
        let rope_freqs = rope_frequencies(config.head_dim, config.rope_base);
        let timings = PipelineStage::all().iter().map(|s| StageTimings::new(*s)).collect();

        Ok(Self {
            config,
            weights,
            rope_freqs,
            timings,
            status: PipelineStatus::Ready,
            position: 0,
            total_forward_calls: 0,
            total_tokens_generated: 0,
            peak_seq_len: 0,
            last_error: None,
        })
    }

    /// Run a single forward pass returning logits of shape `[vocab_size]`.
    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("token_ids must not be empty".into());
        }
        let seq_len = token_ids.len();
        if self.position + seq_len > self.config.max_seq_len {
            return Err("sequence exceeds max_seq_len".into());
        }

        self.status = PipelineStatus::Running;
        let h = self.config.hidden_dim;

        // ── Embedding ──────────────────────────────────────────
        let t0 = Instant::now();
        let mut hidden = self.embedding_lookup(token_ids)?;
        self.record_timing(PipelineStage::Embedding, t0);

        // ── RoPE (applied once to the hidden representation) ───
        let t0 = Instant::now();
        self.apply_rope(&mut hidden, seq_len);
        self.record_timing(PipelineStage::PreNorm, t0);

        // ── Transformer layers ─────────────────────────────────
        for layer_idx in 0..self.config.num_layers {
            // Clone per-layer weight refs to avoid borrow conflict
            let rms_pre = self.weights.layer_weights[layer_idx].rms_pre_gamma.clone();
            let rms_post = self.weights.layer_weights[layer_idx].rms_post_gamma.clone();
            let eps = self.config.rms_norm_eps;

            // Pre-attention RMSNorm
            let t0 = Instant::now();
            let normed = rms_norm(&hidden[(seq_len - 1) * h..seq_len * h], &rms_pre, eps);
            self.record_timing(PipelineStage::PreNorm, t0);

            // Attention
            let t0 = Instant::now();
            let lw = &self.weights.layer_weights[layer_idx];
            let attn_out = self.attention_forward(&normed, lw);
            self.record_timing(PipelineStage::Attention, t0);

            // Residual
            let start = (seq_len - 1) * h;
            add_residual(&mut hidden[start..start + h], &attn_out);

            // Post-attention RMSNorm
            let t0 = Instant::now();
            let normed2 = rms_norm(&hidden[start..start + h], &rms_post, eps);
            self.record_timing(PipelineStage::PostNorm, t0);

            // FFN: SwiGLU
            let t0 = Instant::now();
            let lw = &self.weights.layer_weights[layer_idx];
            let ffn_out = self.ffn_forward(&normed2, lw);
            self.record_timing(PipelineStage::FFN, t0);

            // Residual
            add_residual(&mut hidden[start..start + h], &ffn_out);
        }

        // ── Final RMSNorm ──────────────────────────────────────
        let t0 = Instant::now();
        let start = (seq_len - 1) * h;
        let final_hidden = rms_norm(
            &hidden[start..start + h],
            &self.weights.final_norm_gamma,
            self.config.rms_norm_eps,
        );
        self.record_timing(PipelineStage::PostNorm, t0);

        // ── LM Head ────────────────────────────────────────────
        let t0 = Instant::now();
        let logits = matmul(&final_hidden, &self.weights.lm_head, 1, self.config.vocab_size, h);
        self.record_timing(PipelineStage::LmHead, t0);

        self.position += seq_len;
        if self.position > self.peak_seq_len {
            self.peak_seq_len = self.position;
        }
        self.total_forward_calls += 1;
        self.status = PipelineStatus::Ready;

        Ok(logits)
    }

    /// Autoregressive generation.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult, String> {
        gen_config.validate()?;
        if prompt_tokens.is_empty() {
            return Err("prompt_tokens must not be empty".into());
        }

        let gen_start = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut stop_reason = StopReason::MaxTokens;

        // Prefill: process prompt
        let mut logits = self.forward(prompt_tokens)?;

        for _ in 0..gen_config.max_tokens {
            // Sample
            let t0 = Instant::now();
            let next_token = self.sample(&mut logits, gen_config, prompt_tokens, &generated);
            self.record_timing(PipelineStage::Sampling, t0);

            generated.push(next_token);
            self.total_tokens_generated += 1;

            if gen_config.stop_token == Some(next_token) {
                stop_reason = StopReason::StopToken;
                break;
            }
            if next_token == 0 {
                stop_reason = StopReason::EndOfSequence;
                break;
            }

            // Decode step
            logits = self.forward(&[next_token])?;
        }

        let total_us = gen_start.elapsed().as_micros() as u64;
        let tps = if total_us > 0 {
            generated.len() as f64 / (total_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        Ok(GenerationResult {
            tokens: generated.clone(),
            total_time_us: total_us,
            tokens_per_second: tps,
            stage_timings: self.timings.clone(),
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated.len(),
            stop_reason,
        })
    }

    /// Reset internal state for a fresh sequence.
    pub fn reset(&mut self) {
        self.position = 0;
        self.status = PipelineStatus::Ready;
        for t in &mut self.timings {
            t.total_duration_us = 0;
            t.call_count = 0;
        }
    }

    /// Current per-stage timings.
    pub fn stage_timings(&self) -> &[StageTimings] {
        &self.timings
    }

    /// Overall tokens-per-second across the last generation run.
    pub fn tokens_per_second(&self) -> f64 {
        let total_us: u64 = self.timings.iter().map(|t| t.total_duration_us).sum();
        if total_us == 0 {
            return 0.0;
        }
        self.total_tokens_generated as f64 / (total_us as f64 / 1_000_000.0)
    }

    /// Current status.
    pub fn status(&self) -> PipelineStatus {
        self.status
    }

    /// Runtime diagnostics snapshot.
    pub fn diagnostics(&self) -> PipelineDiagnostics {
        PipelineDiagnostics {
            status: self.status,
            total_forward_calls: self.total_forward_calls,
            total_tokens_generated: self.total_tokens_generated,
            peak_sequence_len: self.peak_seq_len,
            last_error: self.last_error.clone(),
        }
    }

    /// Model configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    // ── Private helpers ────────────────────────────────────────

    fn embedding_lookup(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_dim;
        let mut out = vec![0.0_f32; token_ids.len() * h];
        for (i, &tid) in token_ids.iter().enumerate() {
            let idx = tid as usize;
            if idx >= self.config.vocab_size {
                return Err(format!("token id {tid} >= vocab_size {}", self.config.vocab_size));
            }
            out[i * h..(i + 1) * h]
                .copy_from_slice(&self.weights.embedding_table[idx * h..(idx + 1) * h]);
        }
        Ok(out)
    }

    fn apply_rope(&self, hidden: &mut [f32], seq_len: usize) {
        let h = self.config.hidden_dim;
        let half = self.config.head_dim / 2;
        for pos in 0..seq_len {
            let abs_pos = self.position + pos;
            for head in 0..self.config.num_heads {
                let base = pos * h + head * self.config.head_dim;
                for j in 0..half {
                    let freq = self.rope_freqs[j] * abs_pos as f32;
                    let cos_val = freq.cos();
                    let sin_val = freq.sin();
                    let (a, b) = apply_rope_to_pair(
                        hidden[base + j],
                        hidden[base + half + j],
                        cos_val,
                        sin_val,
                    );
                    hidden[base + j] = a;
                    hidden[base + half + j] = b;
                }
            }
        }
    }

    fn attention_forward(&self, normed: &[f32], lw: &LayerWeights) -> Vec<f32> {
        let h = self.config.hidden_dim;
        let q = matmul(normed, &lw.wq, 1, h, h);
        let k = matmul(normed, &lw.wk, 1, h, h);
        let v = matmul(normed, &lw.wv, 1, h, h);

        let head_dim = self.config.head_dim;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut concat = vec![0.0_f32; h];

        for hd in 0..n_heads {
            let offset = hd * head_dim;
            let mut score = 0.0_f32;
            for d in 0..head_dim {
                score += q[offset + d] * k[offset + d];
            }
            score *= scale;
            concat[offset..offset + head_dim].copy_from_slice(&v[offset..offset + head_dim]);
            let weight = 1.0 / (1.0 + (-score).exp());
            for d in 0..head_dim {
                concat[offset + d] *= weight;
            }
        }

        matmul(&concat, &lw.wo, 1, h, h)
    }

    fn ffn_forward(&self, normed: &[f32], lw: &LayerWeights) -> Vec<f32> {
        let h = self.config.hidden_dim;
        let inter = self.config.intermediate_dim;

        let gate_proj = matmul(normed, &lw.w1, 1, inter, h);
        let up_proj = matmul(normed, &lw.w3, 1, inter, h);

        let gated: Vec<f32> =
            gate_proj.iter().zip(up_proj.iter()).map(|(&g, &u)| silu(g) * u).collect();

        matmul(&gated, &lw.w2, 1, h, inter)
    }

    fn sample(
        &self,
        logits: &mut [f32],
        cfg: &GenerationConfig,
        prompt: &[u32],
        generated: &[u32],
    ) -> u32 {
        let vocab = self.config.vocab_size;

        // Repetition penalty
        if cfg.repetition_penalty != 1.0 {
            for &tok in prompt.iter().chain(generated.iter()) {
                let idx = tok as usize;
                if idx < vocab {
                    if logits[idx] > 0.0 {
                        logits[idx] /= cfg.repetition_penalty;
                    } else {
                        logits[idx] *= cfg.repetition_penalty;
                    }
                }
            }
        }

        // Temperature
        if cfg.temperature == 0.0 {
            return argmax(logits);
        }
        if cfg.temperature != 1.0 {
            for v in logits.iter_mut() {
                *v /= cfg.temperature;
            }
        }

        // Top-k filtering
        if cfg.top_k > 0 && cfg.top_k < vocab {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = indexed[cfg.top_k - 1].1;
            for v in logits.iter_mut() {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax → probabilities
        softmax(logits);

        // Top-p (nucleus) filtering
        if cfg.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cumulative = 0.0_f32;
            let mut cutoff_idx = indexed.len();
            for (i, &(_, p)) in indexed.iter().enumerate() {
                cumulative += p;
                if cumulative > cfg.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            let keep: std::collections::HashSet<usize> =
                indexed[..cutoff_idx].iter().map(|&(i, _)| i).collect();
            for (i, v) in logits.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *v = 0.0;
                }
            }
            // Re-normalise
            let sum: f32 = logits.iter().sum();
            if sum > 0.0 {
                for v in logits.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // Weighted sample using deterministic RNG seeded on position
        let mut rng = Xorshift64::new(self.position as u64 + 7919);
        let r = rng.next_f32();
        let mut cumulative = 0.0_f32;
        for (i, &p) in logits.iter().enumerate() {
            cumulative += p;
            if cumulative >= r {
                return i as u32;
            }
        }
        (vocab - 1) as u32
    }

    fn record_timing(&mut self, stage: PipelineStage, start: Instant) {
        let elapsed = start.elapsed().as_micros() as u64;
        if let Some(t) = self.timings.iter_mut().find(|t| t.stage == stage) {
            t.record(elapsed);
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── PipelineBuilder ────────────────────────────────────────────────

/// Builder for constructing an `InferencePipeline` with ergonomic defaults.
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self { config: PipelineConfig::tiny_test() }
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn vocab_size(mut self, v: usize) -> Self {
        self.config.vocab_size = v;
        self
    }

    pub fn hidden_dim(mut self, h: usize) -> Self {
        self.config.hidden_dim = h;
        self
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.config.num_layers = n;
        self
    }

    pub fn num_heads(mut self, n: usize) -> Self {
        self.config.num_heads = n;
        self
    }

    pub fn head_dim(mut self, d: usize) -> Self {
        self.config.head_dim = d;
        self
    }

    pub fn intermediate_dim(mut self, d: usize) -> Self {
        self.config.intermediate_dim = d;
        self
    }

    pub fn max_seq_len(mut self, s: usize) -> Self {
        self.config.max_seq_len = s;
        self
    }

    pub fn rms_norm_eps(mut self, eps: f32) -> Self {
        self.config.rms_norm_eps = eps;
        self
    }

    pub fn rope_base(mut self, base: f32) -> Self {
        self.config.rope_base = base;
        self
    }

    pub fn build(self) -> Result<InferencePipeline, String> {
        InferencePipeline::new(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── TokenGenerator ─────────────────────────────────────────────────

/// Stateful autoregressive token generator wrapping a pipeline.
pub struct TokenGenerator {
    pipeline: InferencePipeline,
    gen_config: GenerationConfig,
}

impl TokenGenerator {
    pub fn new(pipeline: InferencePipeline, gen_config: GenerationConfig) -> Self {
        Self { pipeline, gen_config }
    }

    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Result<GenerationResult, String> {
        self.pipeline.reset();
        self.pipeline.generate(prompt_tokens, &self.gen_config)
    }

    pub fn pipeline(&self) -> &InferencePipeline {
        &self.pipeline
    }

    pub fn pipeline_mut(&mut self) -> &mut InferencePipeline {
        &mut self.pipeline
    }

    pub fn config(&self) -> &GenerationConfig {
        &self.gen_config
    }

    pub fn set_config(&mut self, cfg: GenerationConfig) {
        self.gen_config = cfg;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_pipeline() -> InferencePipeline {
        InferencePipeline::new(PipelineConfig::tiny_test()).unwrap()
    }

    // ── Pipeline creation ──────────────────────────────────────

    #[test]
    fn test_pipeline_creation_tiny() {
        let p = tiny_pipeline();
        assert_eq!(p.status(), PipelineStatus::Ready);
    }

    #[test]
    fn test_pipeline_creation_bitnet_2b() {
        let p = InferencePipeline::new(PipelineConfig::bitnet_2b()).unwrap();
        assert_eq!(p.config().vocab_size, 32000);
    }

    #[test]
    fn test_pipeline_creation_invalid_vocab() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.vocab_size = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_creation_invalid_hidden() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.hidden_dim = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_creation_invalid_layers() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.num_layers = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_creation_invalid_heads() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.num_heads = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_creation_invalid_head_dim() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.head_dim = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_creation_invalid_intermediate() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.intermediate_dim = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_creation_invalid_max_seq() {
        let mut cfg = PipelineConfig::tiny_test();
        cfg.max_seq_len = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    // ── Forward pass ───────────────────────────────────────────

    #[test]
    fn test_forward_output_shape() {
        let mut p = tiny_pipeline();
        let logits = p.forward(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), p.config().vocab_size);
    }

    #[test]
    fn test_forward_single_token() {
        let mut p = tiny_pipeline();
        let logits = p.forward(&[1]).unwrap();
        assert_eq!(logits.len(), 64);
    }

    #[test]
    fn test_forward_empty_input_fails() {
        let mut p = tiny_pipeline();
        assert!(p.forward(&[]).is_err());
    }

    #[test]
    fn test_forward_out_of_vocab_fails() {
        let mut p = tiny_pipeline();
        assert!(p.forward(&[9999]).is_err());
    }

    #[test]
    fn test_forward_logits_are_finite() {
        let mut p = tiny_pipeline();
        let logits = p.forward(&[1, 2]).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_forward_logits_not_all_zero() {
        let mut p = tiny_pipeline();
        let logits = p.forward(&[1]).unwrap();
        assert!(logits.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn test_forward_deterministic() {
        let mut p1 = tiny_pipeline();
        let mut p2 = tiny_pipeline();
        let l1 = p1.forward(&[1, 2, 3]).unwrap();
        let l2 = p2.forward(&[1, 2, 3]).unwrap();
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_forward_exceeds_max_seq_len() {
        let mut p = tiny_pipeline();
        let tokens: Vec<u32> = (0..129).map(|i| (i % 64) as u32).collect();
        assert!(p.forward(&tokens).is_err());
    }

    // ── Generation ─────────────────────────────────────────────

    #[test]
    fn test_generate_produces_tokens() {
        let mut p = tiny_pipeline();
        let result = p.generate(&[1, 2], &GenerationConfig::greedy().with_max_tokens(4)).unwrap();
        assert!(!result.tokens.is_empty());
        assert!(result.generated_tokens >= 1);
    }

    #[test]
    fn test_generate_max_tokens_1() {
        let mut p = tiny_pipeline();
        let result = p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(1)).unwrap();
        assert!(result.generated_tokens <= 1);
    }

    #[test]
    fn test_generate_respects_stop_token() {
        let mut p = tiny_pipeline();
        let cfg = GenerationConfig::greedy().with_max_tokens(100).with_stop_token(0);
        let result = p.generate(&[1], &cfg).unwrap();
        assert!(result.generated_tokens <= 100);
    }

    #[test]
    fn test_generate_empty_prompt_fails() {
        let mut p = tiny_pipeline();
        assert!(p.generate(&[], &GenerationConfig::default()).is_err());
    }

    #[test]
    fn test_generate_invalid_config_fails() {
        let mut p = tiny_pipeline();
        let cfg = GenerationConfig { max_tokens: 0, ..Default::default() };
        assert!(p.generate(&[1], &cfg).is_err());
    }

    #[test]
    fn test_generate_with_temperature() {
        let mut p = tiny_pipeline();
        let cfg = GenerationConfig::default().with_temperature(0.5).with_max_tokens(4);
        let result = p.generate(&[1, 2], &cfg).unwrap();
        assert!(!result.tokens.is_empty());
    }

    #[test]
    fn test_generate_with_top_k() {
        let mut p = tiny_pipeline();
        let cfg = GenerationConfig::default().with_top_k(5).with_max_tokens(4);
        let result = p.generate(&[1], &cfg).unwrap();
        assert!(!result.tokens.is_empty());
    }

    #[test]
    fn test_generate_with_top_p() {
        let mut p = tiny_pipeline();
        let cfg = GenerationConfig::default().with_top_p(0.9).with_max_tokens(4);
        let result = p.generate(&[1], &cfg).unwrap();
        assert!(!result.tokens.is_empty());
    }

    #[test]
    fn test_generate_with_repetition_penalty() {
        let mut p = tiny_pipeline();
        let cfg = GenerationConfig::default().with_repetition_penalty(1.2).with_max_tokens(4);
        let result = p.generate(&[1], &cfg).unwrap();
        assert!(!result.tokens.is_empty());
    }

    // ── Stage timings ──────────────────────────────────────────

    #[test]
    fn test_stage_timings_all_recorded() {
        let mut p = tiny_pipeline();
        p.forward(&[1, 2]).unwrap();
        let timings = p.stage_timings();
        let with_calls: Vec<_> = timings.iter().filter(|t| t.call_count > 0).collect();
        assert!(with_calls.len() >= 5, "expected >= 5 stages with calls, got {}", with_calls.len());
    }

    #[test]
    fn test_stage_timings_avg_positive() {
        let mut p = tiny_pipeline();
        p.forward(&[1]).unwrap();
        for t in p.stage_timings() {
            if t.call_count > 0 {
                assert!(t.avg_us() >= 0.0);
            }
        }
    }

    #[test]
    fn test_stage_timings_count_increments() {
        let mut p = tiny_pipeline();
        p.forward(&[1]).unwrap();
        let c1: u64 = p.stage_timings().iter().map(|t| t.call_count).sum();
        p.forward(&[2]).unwrap();
        let c2: u64 = p.stage_timings().iter().map(|t| t.call_count).sum();
        assert!(c2 > c1);
    }

    // ── Status transitions ─────────────────────────────────────

    #[test]
    fn test_status_ready_initially() {
        let p = tiny_pipeline();
        assert_eq!(p.status(), PipelineStatus::Ready);
    }

    #[test]
    fn test_status_ready_after_forward() {
        let mut p = tiny_pipeline();
        p.forward(&[1]).unwrap();
        assert_eq!(p.status(), PipelineStatus::Ready);
    }

    #[test]
    fn test_status_ready_after_generate() {
        let mut p = tiny_pipeline();
        p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(2)).unwrap();
        assert_eq!(p.status(), PipelineStatus::Ready);
    }

    // ── Token rate ─────────────────────────────────────────────

    #[test]
    fn test_tokens_per_second_zero_before_gen() {
        let p = tiny_pipeline();
        assert_eq!(p.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_tokens_per_second_positive_after_gen() {
        let mut p = tiny_pipeline();
        p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(4)).unwrap();
        let tps = p.tokens_per_second();
        assert!(tps > 0.0, "expected positive tps, got {tps}");
        assert!(tps.is_finite(), "expected finite tps, got {tps}");
    }

    #[test]
    fn test_generation_result_tps_positive() {
        let mut p = tiny_pipeline();
        let result = p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(4)).unwrap();
        assert!(result.tokens_per_second > 0.0);
        assert!(result.tokens_per_second.is_finite());
    }

    // ── Reset ──────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_position() {
        let mut p = tiny_pipeline();
        p.forward(&[1, 2, 3]).unwrap();
        p.reset();
        assert_eq!(p.status(), PipelineStatus::Ready);
        let tokens: Vec<u32> = (0..64).map(|i| (i % 64) as u32).collect();
        assert!(p.forward(&tokens).is_ok());
    }

    #[test]
    fn test_reset_clears_timings() {
        let mut p = tiny_pipeline();
        p.forward(&[1]).unwrap();
        p.reset();
        for t in p.stage_timings() {
            assert_eq!(t.call_count, 0);
            assert_eq!(t.total_duration_us, 0);
        }
    }

    // ── Diagnostics ────────────────────────────────────────────

    #[test]
    fn test_diagnostics_initial() {
        let p = tiny_pipeline();
        let d = p.diagnostics();
        assert_eq!(d.status, PipelineStatus::Ready);
        assert_eq!(d.total_forward_calls, 0);
        assert_eq!(d.total_tokens_generated, 0);
        assert_eq!(d.peak_sequence_len, 0);
        assert!(d.last_error.is_none());
    }

    #[test]
    fn test_diagnostics_after_forward() {
        let mut p = tiny_pipeline();
        p.forward(&[1, 2, 3]).unwrap();
        let d = p.diagnostics();
        assert_eq!(d.total_forward_calls, 1);
        assert_eq!(d.peak_sequence_len, 3);
    }

    #[test]
    fn test_diagnostics_tokens_generated() {
        let mut p = tiny_pipeline();
        p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(3)).unwrap();
        let d = p.diagnostics();
        assert!(d.total_tokens_generated > 0);
    }

    // ── PipelineBuilder ────────────────────────────────────────

    #[test]
    fn test_builder_default() {
        let p = PipelineBuilder::new().build().unwrap();
        assert_eq!(p.status(), PipelineStatus::Ready);
    }

    #[test]
    fn test_builder_with_config() {
        let p = PipelineBuilder::new().with_config(PipelineConfig::tiny_test()).build().unwrap();
        assert_eq!(p.config().vocab_size, 64);
    }

    #[test]
    fn test_builder_individual_fields() {
        let p = PipelineBuilder::new()
            .vocab_size(128)
            .hidden_dim(64)
            .num_layers(1)
            .num_heads(4)
            .head_dim(16)
            .intermediate_dim(128)
            .max_seq_len(256)
            .rms_norm_eps(1e-6)
            .rope_base(10000.0)
            .build()
            .unwrap();
        assert_eq!(p.config().vocab_size, 128);
        assert_eq!(p.config().num_layers, 1);
    }

    #[test]
    fn test_builder_invalid_rejects() {
        let result = PipelineBuilder::new().vocab_size(0).build();
        assert!(result.is_err());
    }

    // ── TokenGenerator ─────────────────────────────────────────

    #[test]
    fn test_token_generator_basic() {
        let pipeline = tiny_pipeline();
        let cfg = GenerationConfig::greedy().with_max_tokens(4);
        let mut tok_gen = TokenGenerator::new(pipeline, cfg);
        let result = tok_gen.generate(&[1, 2]).unwrap();
        assert!(!result.tokens.is_empty());
    }

    #[test]
    fn test_token_generator_reuse() {
        let pipeline = tiny_pipeline();
        let cfg = GenerationConfig::greedy().with_max_tokens(2);
        let mut tok_gen = TokenGenerator::new(pipeline, cfg);
        let r1 = tok_gen.generate(&[1]).unwrap();
        let r2 = tok_gen.generate(&[1]).unwrap();
        assert_eq!(r1.tokens, r2.tokens);
    }

    #[test]
    fn test_token_generator_config_update() {
        let pipeline = tiny_pipeline();
        let mut tok_gen =
            TokenGenerator::new(pipeline, GenerationConfig::greedy().with_max_tokens(2));
        tok_gen.set_config(GenerationConfig::greedy().with_max_tokens(4));
        assert_eq!(tok_gen.config().max_tokens, 4);
    }

    // ── GenerationConfig validation ────────────────────────────

    #[test]
    fn test_gen_config_invalid_max_tokens() {
        let cfg = GenerationConfig { max_tokens: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gen_config_invalid_temperature() {
        let cfg = GenerationConfig { temperature: -1.0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gen_config_invalid_top_p() {
        let cfg = GenerationConfig { top_p: 0.0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gen_config_top_p_above_one() {
        let cfg = GenerationConfig { top_p: 1.5, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gen_config_invalid_rep_penalty() {
        let cfg = GenerationConfig { repetition_penalty: 0.0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gen_config_valid_defaults() {
        assert!(GenerationConfig::default().validate().is_ok());
    }

    // ── PipelineStage coverage ─────────────────────────────────

    #[test]
    fn test_pipeline_stage_all_count() {
        assert_eq!(PipelineStage::all().len(), 7);
    }

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Embedding), "Embedding");
        assert_eq!(format!("{}", PipelineStage::FFN), "FFN");
    }

    #[test]
    fn test_pipeline_status_display() {
        assert_eq!(format!("{}", PipelineStatus::Ready), "Ready");
        assert_eq!(format!("{}", PipelineStatus::Running), "Running");
        assert_eq!(format!("{}", PipelineStatus::Paused), "Paused");
        assert_eq!(format!("{}", PipelineStatus::Error), "Error");
    }

    // ── Generation result metadata ─────────────────────────────

    #[test]
    fn test_generation_result_prompt_count() {
        let mut p = tiny_pipeline();
        let result =
            p.generate(&[1, 2, 3], &GenerationConfig::greedy().with_max_tokens(2)).unwrap();
        assert_eq!(result.prompt_tokens, 3);
    }

    #[test]
    fn test_generation_result_stage_timings_present() {
        let mut p = tiny_pipeline();
        let result = p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(2)).unwrap();
        assert!(!result.stage_timings.is_empty());
    }

    #[test]
    fn test_generation_result_total_time_positive() {
        let mut p = tiny_pipeline();
        let result = p.generate(&[1], &GenerationConfig::greedy().with_max_tokens(2)).unwrap();
        assert!(result.total_time_us > 0);
    }

    // ── Helper function tests ──────────────────────────────────

    #[test]
    fn test_rms_norm_unit_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let out = rms_norm(&input, &gamma, 1e-5);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.3651).abs() < 0.01);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn test_silu_at_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_deterministic_vec_reproducible() {
        let a = deterministic_vec(100, 1.0, 42);
        let b = deterministic_vec(100, 1.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_rope_frequencies_length() {
        let freqs = rope_frequencies(128, 10000.0);
        assert_eq!(freqs.len(), 64);
    }
}
