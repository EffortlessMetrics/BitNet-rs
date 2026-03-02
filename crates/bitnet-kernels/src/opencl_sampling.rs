//! OpenCL-accelerated token sampling strategies for Intel Arc A770.
//!
//! Provides multiple sampling methods (greedy, temperature, top-k, top-p,
//! min-p, typical, mirostat, beam) with CPU reference implementations and
//! OpenCL kernel sources for GPU offload.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── OpenCL kernel sources ──────────────────────────────────────────────

/// OpenCL kernel for temperature-scaled logit transformation.
pub const OPENCL_TEMPERATURE_KERNEL: &str = r#"
__kernel void temperature_scale(
    __global float* logits,
    const uint n,
    const float inv_temperature)
{
    uint gid = get_global_id(0);
    if (gid < n) {
        logits[gid] *= inv_temperature;
    }
}
"#;

/// OpenCL kernel for partial-sort top-k selection using bitonic sort.
pub const OPENCL_TOP_K_KERNEL: &str = r#"
__kernel void top_k_partial_sort(
    __global const float* logits,
    __global float* out_values,
    __global uint*  out_indices,
    const uint n,
    const uint k)
{
    // Each work-group cooperatively finds the top-k elements.
    __local float  local_vals[256];
    __local uint   local_idxs[256];

    uint lid  = get_local_id(0);
    uint gid  = get_global_id(0);

    // Load value (or -INFINITY sentinel)
    float val = (gid < n) ? logits[gid] : -INFINITY;
    local_vals[lid]  = val;
    local_idxs[lid]  = gid;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Bitonic sort within the work-group (descending)
    uint local_size = get_local_size(0);
    for (uint size = 2; size <= local_size; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            uint ixj = lid ^ stride;
            if (ixj > lid) {
                bool ascending = ((lid & size) == 0);
                bool swap = ascending
                    ? (local_vals[lid] < local_vals[ixj])
                    : (local_vals[lid] > local_vals[ixj]);
                if (swap) {
                    float tv  = local_vals[lid];
                    local_vals[lid]  = local_vals[ixj];
                    local_vals[ixj]  = tv;
                    uint ti = local_idxs[lid];
                    local_idxs[lid]  = local_idxs[ixj];
                    local_idxs[ixj]  = ti;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // First k threads write results
    if (lid < k && lid < n) {
        out_values[lid]  = local_vals[lid];
        out_indices[lid] = local_idxs[lid];
    }
}
"#;

/// OpenCL kernel for numerically-stable softmax over a logit vector.
pub const OPENCL_SOFTMAX_KERNEL: &str = r#"
__kernel void softmax(
    __global const float* logits,
    __global float*       probs,
    __local  float*       scratch,
    const uint n)
{
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    // Phase 1: find max (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = lid; i < n; i += local_size) {
        local_max = fmax(local_max, logits[i]);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = local_size >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = lid; i < n; i += local_size) {
        float e = exp(logits[i] - max_val);
        probs[i] = e;
        local_sum += e;
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = local_size >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: normalize
    float inv_total = 1.0f / total;
    for (uint i = lid; i < n; i += local_size) {
        probs[i] *= inv_total;
    }
}
"#;

// ── Types ──────────────────────────────────────────────────────────────

/// Token sampling strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    Temperature(f32),
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
    MinP { p: f32, temperature: f32 },
    TypicalP { p: f32 },
    Mirostat { tau: f32, eta: f32 },
    Beam { width: usize },
}

/// Full sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub method: SamplingMethod,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: SamplingMethod::Greedy,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

/// Result of sampling a single token.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledToken {
    pub token_id: u32,
    pub probability: f32,
    pub log_prob: f32,
}

/// State for a single beam in beam search.
#[derive(Debug, Clone)]
pub struct BeamState {
    pub tokens: Vec<u32>,
    pub score: f64,
    pub finished: bool,
}

/// Adaptive state for Mirostat sampling.
#[derive(Debug, Clone)]
pub struct MirostatState {
    pub mu: f32,
    pub tau: f32,
    pub eta: f32,
}

/// Aggregate sampling statistics.
#[derive(Debug, Clone, Default)]
pub struct SamplingStats {
    pub total_samples: u64,
    pub greedy_count: u64,
    pub random_count: u64,
    pub beam_count: u64,
    pub avg_entropy: f64,
}

/// Main sampler holding config, RNG state, and statistics.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub config: SamplingConfig,
    pub rng_state: u64,
    pub mirostat_state: Option<MirostatState>,
    pub stats: SamplingStats,
}

/// Errors that can occur during sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingError {
    EmptyLogits,
    InvalidTemperature(String),
    InvalidTopK(usize),
    InvalidTopP(String),
    AllLogitsFiltered,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => write!(f, "empty logits"),
            Self::InvalidTemperature(v) => write!(f, "invalid temperature: {v}"),
            Self::InvalidTopK(k) => write!(f, "invalid top-k: {k}"),
            Self::InvalidTopP(v) => write!(f, "invalid top-p: {v}"),
            Self::AllLogitsFiltered => write!(f, "all logits filtered"),
        }
    }
}

impl std::error::Error for SamplingError {}

// ── Helpers ────────────────────────────────────────────────────────────

/// xorshift64 PRNG – returns value in [0, 1).
fn xorshift64_f32(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Map to [0, 1) using upper 24 bits for mantissa
    (s >> 40) as f32 / (1u64 << 24) as f32
}

/// Mix a seed into a well-distributed initial RNG state.
fn mix_seed(seed: u64) -> u64 {
    // splitmix64 to decorrelate similar seeds
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Numerically-stable softmax in-place, returns slice unchanged reference.
fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    let inv = 1.0 / sum as f32;
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

/// Sample an index from a probability distribution using the RNG.
fn sample_from_probs(probs: &[f32], rng: &mut u64) -> usize {
    let r = xorshift64_f32(rng);
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

// ── Public API ─────────────────────────────────────────────────────────

/// Create a new [`Sampler`] from the given configuration.
pub fn create_sampler(config: SamplingConfig) -> Sampler {
    let rng_state = mix_seed(config.seed.unwrap_or(0xDEAD_BEEF_CAFE_1234));
    let mirostat_state = match &config.method {
        SamplingMethod::Mirostat { tau, eta } => {
            Some(MirostatState { mu: 2.0 * *tau, tau: *tau, eta: *eta })
        }
        _ => None,
    };
    Sampler { config, rng_state, mirostat_state, stats: SamplingStats::default() }
}

/// Greedy (argmax) sampling.
pub fn cpu_sample_greedy(logits: &[f32]) -> SampledToken {
    let (idx, _) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    SampledToken { token_id: idx as u32, probability: probs[idx], log_prob: probs[idx].ln() }
}

/// Temperature sampling.
pub fn cpu_sample_temperature(logits: &[f32], temperature: f32, rng: &mut u64) -> SampledToken {
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    softmax_inplace(&mut scaled);
    let idx = sample_from_probs(&scaled, rng);
    SampledToken { token_id: idx as u32, probability: scaled[idx], log_prob: scaled[idx].ln() }
}

/// Top-k sampling: keep only the k highest-probability tokens.
pub fn cpu_sample_top_k(logits: &[f32], k: usize, temperature: f32, rng: &mut u64) -> SampledToken {
    let mut indexed: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v / temperature)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    let mut vals: Vec<f32> = indexed.iter().map(|&(_, v)| v).collect();
    softmax_inplace(&mut vals);
    let local_idx = sample_from_probs(&vals, rng);
    let token_id = indexed[local_idx].0 as u32;
    SampledToken { token_id, probability: vals[local_idx], log_prob: vals[local_idx].ln() }
}

/// Top-p (nucleus) sampling: keep smallest set whose cumulative prob ≥ p.
pub fn cpu_sample_top_p(logits: &[f32], p: f32, temperature: f32, rng: &mut u64) -> SampledToken {
    let indexed: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v / temperature)).collect();
    let mut probs: Vec<f32> = indexed.iter().map(|&(_, v)| v).collect();
    softmax_inplace(&mut probs);
    // Pair indices with probs, sort descending by prob
    let mut pairs: Vec<(usize, f32)> =
        indexed.iter().zip(probs.iter()).map(|(&(i, _), &pr)| (i, pr)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0f32;
    let mut kept = Vec::new();
    for (i, pr) in &pairs {
        kept.push((*i, *pr));
        cumulative += pr;
        if cumulative >= p {
            break;
        }
    }

    // Re-normalize
    let sum: f32 = kept.iter().map(|&(_, pr)| pr).sum();
    let inv = 1.0 / sum;
    let renorm: Vec<f32> = kept.iter().map(|&(_, pr)| pr * inv).collect();
    let local_idx = sample_from_probs(&renorm, rng);
    let token_id = kept[local_idx].0 as u32;
    SampledToken { token_id, probability: renorm[local_idx], log_prob: renorm[local_idx].ln() }
}

/// Min-p sampling: keep tokens with prob ≥ p × max_prob.
pub fn cpu_sample_min_p(logits: &[f32], p: f32, temperature: f32, rng: &mut u64) -> SampledToken {
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    softmax_inplace(&mut scaled);
    let max_prob = scaled.iter().cloned().fold(0.0f32, f32::max);
    let threshold = p * max_prob;

    let kept: Vec<(usize, f32)> = scaled
        .iter()
        .enumerate()
        .filter(|&(_, &pr)| pr >= threshold)
        .map(|(i, &pr)| (i, pr))
        .collect();

    if kept.is_empty() {
        // Fallback to greedy
        return cpu_sample_greedy(logits);
    }

    let sum: f32 = kept.iter().map(|&(_, pr)| pr).sum();
    let inv = 1.0 / sum;
    let renorm: Vec<f32> = kept.iter().map(|&(_, pr)| pr * inv).collect();
    let local_idx = sample_from_probs(&renorm, rng);
    let token_id = kept[local_idx].0 as u32;
    SampledToken { token_id, probability: renorm[local_idx], log_prob: renorm[local_idx].ln() }
}

/// Typical decoding: select tokens whose information content is close to
/// the expected entropy of the distribution.
pub fn cpu_sample_typical(logits: &[f32], p: f32, rng: &mut u64) -> SampledToken {
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let entropy = cpu_compute_entropy(&probs);

    // Compute |−log(p_i) − H| for each token
    let mut scored: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|&(_, &pr)| pr > 0.0)
        .map(|(i, &pr)| {
            let info = -pr.ln();
            let deviation = (info as f64 - entropy).abs() as f32;
            (i, pr, deviation)
        })
        .collect();

    // Sort by deviation ascending (most "typical" first)
    scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut cumulative = 0.0f32;
    let mut kept = Vec::new();
    for &(i, pr, _) in &scored {
        kept.push((i, pr));
        cumulative += pr;
        if cumulative >= p {
            break;
        }
    }

    let sum: f32 = kept.iter().map(|&(_, pr)| pr).sum();
    let inv = 1.0 / sum;
    let renorm: Vec<f32> = kept.iter().map(|&(_, pr)| pr * inv).collect();
    let local_idx = sample_from_probs(&renorm, rng);
    let token_id = kept[local_idx].0 as u32;
    SampledToken { token_id, probability: renorm[local_idx], log_prob: renorm[local_idx].ln() }
}

/// Mirostat adaptive sampling (v2 style).
pub fn cpu_sample_mirostat(
    logits: &[f32],
    state: &mut MirostatState,
    rng: &mut u64,
) -> SampledToken {
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);

    let mut sorted: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Truncate based on surprise threshold: keep tokens with −log2(p) ≤ mu
    let mut kept = Vec::new();
    for &(i, p) in &sorted {
        if p > 0.0 && (-p.log2() <= state.mu || kept.is_empty()) {
            kept.push((i, p));
        }
    }

    let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
    let inv = 1.0 / sum;
    let renorm: Vec<f32> = kept.iter().map(|&(_, p)| p * inv).collect();
    let local_idx = sample_from_probs(&renorm, rng);
    let token_id = kept[local_idx].0 as u32;
    let chosen_prob = kept[local_idx].1;

    // Update mu: move towards target surprise τ
    let surprise = -chosen_prob.log2();
    state.mu += state.eta * (state.tau - surprise);

    SampledToken { token_id, probability: renorm[local_idx], log_prob: renorm[local_idx].ln() }
}

/// Apply multiplicative repetition penalty to previously-seen tokens.
pub fn cpu_apply_repetition_penalty(logits: &mut [f32], tokens: &[u32], penalty: f32) {
    for &tok in tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply frequency penalty proportional to token counts.
pub fn cpu_apply_frequency_penalty(
    logits: &mut [f32],
    token_counts: &HashMap<u32, usize>,
    penalty: f32,
) {
    for (&tok, &count) in token_counts {
        let idx = tok as usize;
        if idx < logits.len() {
            logits[idx] -= penalty * count as f32;
        }
    }
}

/// Apply a flat presence penalty for every token that has appeared.
pub fn cpu_apply_presence_penalty(logits: &mut [f32], token_set: &HashSet<u32>, penalty: f32) {
    for &tok in token_set {
        let idx = tok as usize;
        if idx < logits.len() {
            logits[idx] -= penalty;
        }
    }
}

/// Shannon entropy (in nats) of a probability distribution.
pub fn cpu_compute_entropy(probs: &[f32]) -> f64 {
    let mut h = 0.0f64;
    for &p in probs {
        if p > 0.0 {
            h -= (p as f64) * (p as f64).ln();
        }
    }
    h
}

/// Unified sampling entry-point that dispatches on the configured method.
pub fn cpu_sample(sampler: &mut Sampler, logits: &[f32]) -> Result<SampledToken, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }

    let token = match &sampler.config.method {
        SamplingMethod::Greedy => {
            sampler.stats.greedy_count += 1;
            cpu_sample_greedy(logits)
        }
        SamplingMethod::Temperature(t) => {
            let t = *t;
            if t <= 0.0 {
                return Err(SamplingError::InvalidTemperature(format!("{t}")));
            }
            sampler.stats.random_count += 1;
            cpu_sample_temperature(logits, t, &mut sampler.rng_state)
        }
        SamplingMethod::TopK { k, temperature } => {
            let k = *k;
            let temp = *temperature;
            if k == 0 {
                return Err(SamplingError::InvalidTopK(k));
            }
            if temp <= 0.0 {
                return Err(SamplingError::InvalidTemperature(format!("{temp}")));
            }
            sampler.stats.random_count += 1;
            cpu_sample_top_k(logits, k, temp, &mut sampler.rng_state)
        }
        SamplingMethod::TopP { p, temperature } => {
            let p = *p;
            let temp = *temperature;
            if p <= 0.0 || p > 1.0 {
                return Err(SamplingError::InvalidTopP(format!("{p}")));
            }
            if temp <= 0.0 {
                return Err(SamplingError::InvalidTemperature(format!("{temp}")));
            }
            sampler.stats.random_count += 1;
            cpu_sample_top_p(logits, p, temp, &mut sampler.rng_state)
        }
        SamplingMethod::MinP { p, temperature } => {
            let p = *p;
            let temp = *temperature;
            if temp <= 0.0 {
                return Err(SamplingError::InvalidTemperature(format!("{temp}")));
            }
            sampler.stats.random_count += 1;
            cpu_sample_min_p(logits, p, temp, &mut sampler.rng_state)
        }
        SamplingMethod::TypicalP { p } => {
            let p = *p;
            sampler.stats.random_count += 1;
            cpu_sample_typical(logits, p, &mut sampler.rng_state)
        }
        SamplingMethod::Mirostat { .. } => {
            sampler.stats.random_count += 1;
            let state = sampler.mirostat_state.as_mut().unwrap();
            cpu_sample_mirostat(logits, state, &mut sampler.rng_state)
        }
        SamplingMethod::Beam { .. } => {
            sampler.stats.beam_count += 1;
            // Beam search degrades to greedy for single-token sampling
            cpu_sample_greedy(logits)
        }
    };

    sampler.stats.total_samples += 1;

    // Track running average entropy
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let h = cpu_compute_entropy(&probs);
    let n = sampler.stats.total_samples as f64;
    sampler.stats.avg_entropy = sampler.stats.avg_entropy * ((n - 1.0) / n) + h / n;

    Ok(token)
}

/// Human-readable description of a sampling configuration.
pub fn format_sampling_config(config: &SamplingConfig) -> String {
    let method = match &config.method {
        SamplingMethod::Greedy => "greedy".to_string(),
        SamplingMethod::Temperature(t) => format!("temperature({t})"),
        SamplingMethod::TopK { k, temperature } => {
            format!("top_k(k={k}, temp={temperature})")
        }
        SamplingMethod::TopP { p, temperature } => {
            format!("top_p(p={p}, temp={temperature})")
        }
        SamplingMethod::MinP { p, temperature } => {
            format!("min_p(p={p}, temp={temperature})")
        }
        SamplingMethod::TypicalP { p } => format!("typical(p={p})"),
        SamplingMethod::Mirostat { tau, eta } => {
            format!("mirostat(tau={tau}, eta={eta})")
        }
        SamplingMethod::Beam { width } => format!("beam(width={width})"),
    };
    let mut parts = vec![method];
    if config.repetition_penalty != 1.0 {
        parts.push(format!("rep_pen={}", config.repetition_penalty));
    }
    if config.frequency_penalty != 0.0 {
        parts.push(format!("freq_pen={}", config.frequency_penalty));
    }
    if config.presence_penalty != 0.0 {
        parts.push(format!("pres_pen={}", config.presence_penalty));
    }
    if let Some(seed) = config.seed {
        parts.push(format!("seed={seed}"));
    }
    parts.join(", ")
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(vals: &[f32]) -> Vec<f32> {
        vals.to_vec()
    }

    fn seed_rng() -> u64 {
        0xCAFE_BABE_1234_5678
    }

    // ── Greedy tests ───────────────────────────────────────────────

    #[test]
    fn greedy_returns_highest_logit() {
        let logits = make_logits(&[1.0, 3.0, 2.0, 0.5]);
        let tok = cpu_sample_greedy(&logits);
        assert_eq!(tok.token_id, 1);
    }

    #[test]
    fn greedy_returns_consistent_on_tie() {
        let logits = make_logits(&[5.0, 5.0, 5.0]);
        let tok = cpu_sample_greedy(&logits);
        // With equal logits, max_by returns the last maximum index
        assert!(tok.token_id < 3);
    }

    #[test]
    fn greedy_single_element() {
        let logits = make_logits(&[42.0]);
        let tok = cpu_sample_greedy(&logits);
        assert_eq!(tok.token_id, 0);
        assert!((tok.probability - 1.0).abs() < 1e-5);
    }

    #[test]
    fn greedy_always_picks_max() {
        for _ in 0..20 {
            let logits = make_logits(&[0.1, 0.2, 10.0, 0.3]);
            assert_eq!(cpu_sample_greedy(&logits).token_id, 2);
        }
    }

    // ── Temperature tests ──────────────────────────────────────────

    #[test]
    fn temperature_lower_is_more_deterministic() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 0.5]);
        let mut counts_low = [0u32; 4];
        let mut counts_high = [0u32; 4];
        let mut rng_low = seed_rng();
        let mut rng_high = seed_rng();

        for _ in 0..200 {
            let t1 = cpu_sample_temperature(&logits, 0.1, &mut rng_low);
            counts_low[t1.token_id as usize] += 1;
            let t2 = cpu_sample_temperature(&logits, 2.0, &mut rng_high);
            counts_high[t2.token_id as usize] += 1;
        }
        // Low temp should concentrate on the max logit (index 2)
        assert!(counts_low[2] > counts_high[2]);
    }

    #[test]
    fn temperature_near_zero_acts_greedy() {
        let logits = make_logits(&[1.0, 5.0, 2.0]);
        let mut rng = seed_rng();
        for _ in 0..50 {
            let tok = cpu_sample_temperature(&logits, 0.001, &mut rng);
            assert_eq!(tok.token_id, 1);
        }
    }

    #[test]
    fn temperature_produces_valid_probability() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let mut rng = seed_rng();
        let tok = cpu_sample_temperature(&logits, 1.0, &mut rng);
        assert!(tok.probability > 0.0 && tok.probability <= 1.0);
    }

    // ── Top-k tests ────────────────────────────────────────────────

    #[test]
    fn top_k_only_considers_k_tokens() {
        let logits = make_logits(&[0.1, 0.2, 10.0, 9.0, 0.3]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let tok = cpu_sample_top_k(&logits, 2, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        // Should only sample from the top-2 tokens (ids 2 and 3)
        assert!(seen.is_subset(&HashSet::from([2, 3])));
    }

    #[test]
    fn top_k_equals_one_is_greedy() {
        let logits = make_logits(&[1.0, 5.0, 3.0]);
        let mut rng = seed_rng();
        for _ in 0..20 {
            let tok = cpu_sample_top_k(&logits, 1, 1.0, &mut rng);
            assert_eq!(tok.token_id, 1);
        }
    }

    #[test]
    fn top_k_with_k_larger_than_vocab() {
        let logits = make_logits(&[1.0, 2.0]);
        let mut rng = seed_rng();
        let tok = cpu_sample_top_k(&logits, 100, 1.0, &mut rng);
        assert!(tok.token_id < 2);
    }

    // ── Top-p tests ────────────────────────────────────────────────

    #[test]
    fn top_p_respects_cumulative_threshold() {
        // Token 2 has ~90% probability after softmax with these logits
        let logits = make_logits(&[0.0, 0.0, 10.0, 0.0]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..100 {
            let tok = cpu_sample_top_p(&logits, 0.5, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        // With p=0.5, should mostly just pick the dominant token
        assert!(seen.contains(&2));
    }

    #[test]
    fn top_p_one_includes_all_tokens() {
        let logits = make_logits(&[1.0, 1.0, 1.0, 1.0]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let tok = cpu_sample_top_p(&logits, 1.0, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn top_p_produces_valid_probability() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let mut rng = seed_rng();
        let tok = cpu_sample_top_p(&logits, 0.9, 1.0, &mut rng);
        assert!(tok.probability > 0.0 && tok.probability <= 1.0);
    }

    // ── Min-p tests ────────────────────────────────────────────────

    #[test]
    fn min_p_filters_low_probability_tokens() {
        // With a large gap, min-p=0.5 should filter most tokens
        let logits = make_logits(&[0.0, 0.0, 10.0, 0.0, 0.0]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..100 {
            let tok = cpu_sample_min_p(&logits, 0.5, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        assert!(seen.contains(&2));
        assert!(seen.len() <= 2); // Dominant token + possibly one other
    }

    #[test]
    fn min_p_zero_includes_all() {
        let logits = make_logits(&[1.0, 1.0, 1.0, 1.0]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let tok = cpu_sample_min_p(&logits, 0.0, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn min_p_high_threshold_favors_max() {
        let logits = make_logits(&[1.0, 5.0, 2.0]);
        let mut rng = seed_rng();
        let mut max_count = 0u32;
        for _ in 0..100 {
            let tok = cpu_sample_min_p(&logits, 0.9, 1.0, &mut rng);
            if tok.token_id == 1 {
                max_count += 1;
            }
        }
        assert!(max_count > 80);
    }

    // ── Typical sampling tests ─────────────────────────────────────

    #[test]
    fn typical_sampling_produces_valid_token() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0]);
        let mut rng = seed_rng();
        let tok = cpu_sample_typical(&logits, 0.9, &mut rng);
        assert!(tok.token_id < 4);
        assert!(tok.probability > 0.0);
    }

    #[test]
    fn typical_p_one_explores_all() {
        let logits = make_logits(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..300 {
            let tok = cpu_sample_typical(&logits, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn typical_respects_information_content() {
        // Token at index 3 has highest logit → lowest surprise → most typical
        let logits = make_logits(&[0.1, 0.1, 0.1, 10.0]);
        let mut rng = seed_rng();
        let mut counts = [0u32; 4];
        for _ in 0..200 {
            let tok = cpu_sample_typical(&logits, 0.3, &mut rng);
            counts[tok.token_id as usize] += 1;
        }
        assert!(counts[3] > counts[0]);
    }

    // ── Mirostat tests ─────────────────────────────────────────────

    #[test]
    fn mirostat_adapts_mu() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0]);
        let mut state = MirostatState { mu: 6.0, tau: 3.0, eta: 0.1 };
        let mut rng = seed_rng();
        let initial_mu = state.mu;
        for _ in 0..20 {
            cpu_sample_mirostat(&logits, &mut state, &mut rng);
        }
        // mu should have moved towards tau
        assert!((state.mu - initial_mu).abs() > 0.0);
    }

    #[test]
    fn mirostat_produces_valid_token() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let mut state = MirostatState { mu: 6.0, tau: 3.0, eta: 0.1 };
        let mut rng = seed_rng();
        let tok = cpu_sample_mirostat(&logits, &mut state, &mut rng);
        assert!(tok.token_id < 3);
        assert!(tok.probability > 0.0);
    }

    #[test]
    fn mirostat_handles_single_token() {
        let logits = make_logits(&[5.0]);
        let mut state = MirostatState { mu: 6.0, tau: 3.0, eta: 0.1 };
        let mut rng = seed_rng();
        let tok = cpu_sample_mirostat(&logits, &mut state, &mut rng);
        assert_eq!(tok.token_id, 0);
    }

    // ── Penalty tests ──────────────────────────────────────────────

    #[test]
    fn repetition_penalty_lowers_positive_logits() {
        let mut logits = make_logits(&[1.0, 2.0, 3.0, 4.0]);
        let original = logits.clone();
        cpu_apply_repetition_penalty(&mut logits, &[1, 3], 2.0);
        assert!(logits[1] < original[1]);
        assert!(logits[3] < original[3]);
        assert_eq!(logits[0], original[0]); // Unchanged
        assert_eq!(logits[2], original[2]);
    }

    #[test]
    fn repetition_penalty_amplifies_negative_logits() {
        let mut logits = make_logits(&[-1.0, -2.0, 3.0]);
        cpu_apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!(logits[0] < -1.0); // More negative
        assert!(logits[1] < -2.0);
    }

    #[test]
    fn repetition_penalty_one_is_noop() {
        let mut logits = make_logits(&[1.0, 2.0, 3.0]);
        let original = logits.clone();
        cpu_apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn frequency_penalty_proportional_to_count() {
        let mut logits = make_logits(&[5.0, 5.0, 5.0]);
        let mut counts = HashMap::new();
        counts.insert(0u32, 1usize);
        counts.insert(1, 3);
        cpu_apply_frequency_penalty(&mut logits, &counts, 1.0);
        assert!((logits[0] - 4.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn frequency_penalty_zero_is_noop() {
        let mut logits = make_logits(&[5.0, 5.0]);
        let mut counts = HashMap::new();
        counts.insert(0u32, 10usize);
        let original = logits.clone();
        cpu_apply_frequency_penalty(&mut logits, &counts, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn presence_penalty_binary_for_seen_tokens() {
        let mut logits = make_logits(&[5.0, 5.0, 5.0, 5.0]);
        let set: HashSet<u32> = [1, 3].into_iter().collect();
        cpu_apply_presence_penalty(&mut logits, &set, 2.0);
        assert!((logits[0] - 5.0).abs() < 1e-6);
        assert!((logits[1] - 3.0).abs() < 1e-6);
        assert!((logits[2] - 5.0).abs() < 1e-6);
        assert!((logits[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn presence_penalty_zero_is_noop() {
        let mut logits = make_logits(&[3.0, 4.0]);
        let set: HashSet<u32> = [0, 1].into_iter().collect();
        let original = logits.clone();
        cpu_apply_presence_penalty(&mut logits, &set, 0.0);
        assert_eq!(logits, original);
    }

    // ── Entropy tests ──────────────────────────────────────────────

    #[test]
    fn entropy_deterministic_is_zero() {
        let probs = [1.0, 0.0, 0.0];
        let h = cpu_compute_entropy(&probs);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn entropy_uniform_is_ln_n() {
        let probs = [0.25f32, 0.25, 0.25, 0.25];
        let h = cpu_compute_entropy(&probs);
        let expected = (4.0f64).ln();
        assert!((h - expected).abs() < 1e-6);
    }

    #[test]
    fn entropy_non_negative() {
        let probs = [0.7, 0.2, 0.1];
        assert!(cpu_compute_entropy(&probs) >= 0.0);
    }

    // ── Determinism / seed tests ───────────────────────────────────

    #[test]
    fn same_seed_same_result() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = SamplingConfig {
            method: SamplingMethod::Temperature(0.8),
            seed: Some(42),
            ..Default::default()
        };
        let mut s1 = create_sampler(config.clone());
        let mut s2 = create_sampler(config);
        let t1 = cpu_sample(&mut s1, &logits).unwrap();
        let t2 = cpu_sample(&mut s2, &logits).unwrap();
        assert_eq!(t1.token_id, t2.token_id);
    }

    #[test]
    fn different_seeds_likely_differ() {
        let logits = make_logits(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut different = false;
        for offset in 1..50u64 {
            let c1 = SamplingConfig {
                method: SamplingMethod::Temperature(1.0),
                seed: Some(100),
                ..Default::default()
            };
            let c2 = SamplingConfig {
                method: SamplingMethod::Temperature(1.0),
                seed: Some(100 + offset),
                ..Default::default()
            };
            let mut s1 = create_sampler(c1);
            let mut s2 = create_sampler(c2);
            let t1 = cpu_sample(&mut s1, &logits).unwrap();
            let t2 = cpu_sample(&mut s2, &logits).unwrap();
            if t1.token_id != t2.token_id {
                different = true;
                break;
            }
        }
        assert!(different, "different seeds should eventually produce different tokens");
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn single_logit_all_methods() {
        let logits = make_logits(&[7.0]);
        assert_eq!(cpu_sample_greedy(&logits).token_id, 0);
        let mut rng = seed_rng();
        assert_eq!(cpu_sample_temperature(&logits, 1.0, &mut rng).token_id, 0);
        assert_eq!(cpu_sample_top_k(&logits, 1, 1.0, &mut rng).token_id, 0);
        assert_eq!(cpu_sample_top_p(&logits, 0.5, 1.0, &mut rng).token_id, 0);
        assert_eq!(cpu_sample_min_p(&logits, 0.1, 1.0, &mut rng).token_id, 0);
        assert_eq!(cpu_sample_typical(&logits, 0.9, &mut rng).token_id, 0);
    }

    #[test]
    fn all_equal_logits_uniform_sampling() {
        let logits = make_logits(&[1.0, 1.0, 1.0, 1.0]);
        let mut rng = seed_rng();
        let mut seen = HashSet::new();
        for _ in 0..400 {
            let tok = cpu_sample_temperature(&logits, 1.0, &mut rng);
            seen.insert(tok.token_id);
        }
        assert_eq!(seen.len(), 4, "uniform logits should visit all tokens");
    }

    #[test]
    fn very_large_logits_no_overflow() {
        let logits = make_logits(&[1000.0, 999.0, 998.0]);
        let tok = cpu_sample_greedy(&logits);
        assert_eq!(tok.token_id, 0);
        assert!(tok.probability.is_finite());
    }

    #[test]
    fn very_negative_logits() {
        let logits = make_logits(&[-1000.0, -999.0, -998.0]);
        let tok = cpu_sample_greedy(&logits);
        assert_eq!(tok.token_id, 2); // Least negative
        assert!(tok.probability.is_finite());
    }

    #[test]
    fn mixed_extreme_logits() {
        let logits = make_logits(&[-500.0, 500.0, -500.0]);
        let tok = cpu_sample_greedy(&logits);
        assert_eq!(tok.token_id, 1);
    }

    // ── Error handling ─────────────────────────────────────────────

    #[test]
    fn empty_logits_returns_error() {
        let config = SamplingConfig::default();
        let mut sampler = create_sampler(config);
        let result = cpu_sample(&mut sampler, &[]);
        assert_eq!(result.unwrap_err(), SamplingError::EmptyLogits);
    }

    #[test]
    fn invalid_temperature_zero() {
        let config =
            SamplingConfig { method: SamplingMethod::Temperature(0.0), ..Default::default() };
        let mut sampler = create_sampler(config);
        let result = cpu_sample(&mut sampler, &[1.0, 2.0]);
        assert!(matches!(result, Err(SamplingError::InvalidTemperature(_))));
    }

    #[test]
    fn invalid_temperature_negative() {
        let config =
            SamplingConfig { method: SamplingMethod::Temperature(-1.0), ..Default::default() };
        let mut sampler = create_sampler(config);
        let result = cpu_sample(&mut sampler, &[1.0, 2.0]);
        assert!(matches!(result, Err(SamplingError::InvalidTemperature(_))));
    }

    #[test]
    fn invalid_top_k_zero() {
        let config = SamplingConfig {
            method: SamplingMethod::TopK { k: 0, temperature: 1.0 },
            ..Default::default()
        };
        let mut sampler = create_sampler(config);
        let result = cpu_sample(&mut sampler, &[1.0, 2.0]);
        assert!(matches!(result, Err(SamplingError::InvalidTopK(0))));
    }

    #[test]
    fn invalid_top_p_zero() {
        let config = SamplingConfig {
            method: SamplingMethod::TopP { p: 0.0, temperature: 1.0 },
            ..Default::default()
        };
        let mut sampler = create_sampler(config);
        let result = cpu_sample(&mut sampler, &[1.0, 2.0]);
        assert!(matches!(result, Err(SamplingError::InvalidTopP(_))));
    }

    #[test]
    fn invalid_top_p_greater_than_one() {
        let config = SamplingConfig {
            method: SamplingMethod::TopP { p: 1.5, temperature: 1.0 },
            ..Default::default()
        };
        let mut sampler = create_sampler(config);
        let result = cpu_sample(&mut sampler, &[1.0, 2.0]);
        assert!(matches!(result, Err(SamplingError::InvalidTopP(_))));
    }

    // ── Property tests ─────────────────────────────────────────────

    #[test]
    fn probabilities_sum_to_one() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut probs = logits.clone();
        softmax_inplace(&mut probs);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sampled_token_has_nonzero_probability() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0]);
        let mut rng = seed_rng();
        for _ in 0..50 {
            let tok = cpu_sample_temperature(&logits, 1.0, &mut rng);
            assert!(tok.probability > 0.0);
        }
    }

    #[test]
    fn sampled_token_in_valid_range() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let mut rng = seed_rng();
        for _ in 0..100 {
            let tok = cpu_sample_temperature(&logits, 1.0, &mut rng);
            assert!(tok.token_id < 3);
        }
    }

    // ── Stats tracking ─────────────────────────────────────────────

    #[test]
    fn stats_track_greedy_count() {
        let config = SamplingConfig::default();
        let mut sampler = create_sampler(config);
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        for _ in 0..5 {
            cpu_sample(&mut sampler, &logits).unwrap();
        }
        assert_eq!(sampler.stats.total_samples, 5);
        assert_eq!(sampler.stats.greedy_count, 5);
    }

    #[test]
    fn stats_track_random_count() {
        let config = SamplingConfig {
            method: SamplingMethod::Temperature(1.0),
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = create_sampler(config);
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        for _ in 0..3 {
            cpu_sample(&mut sampler, &logits).unwrap();
        }
        assert_eq!(sampler.stats.total_samples, 3);
        assert_eq!(sampler.stats.random_count, 3);
    }

    #[test]
    fn stats_track_beam_count() {
        let config =
            SamplingConfig { method: SamplingMethod::Beam { width: 4 }, ..Default::default() };
        let mut sampler = create_sampler(config);
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        cpu_sample(&mut sampler, &logits).unwrap();
        assert_eq!(sampler.stats.beam_count, 1);
    }

    #[test]
    fn stats_avg_entropy_populated() {
        let config = SamplingConfig {
            method: SamplingMethod::Temperature(1.0),
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = create_sampler(config);
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0]);
        for _ in 0..10 {
            cpu_sample(&mut sampler, &logits).unwrap();
        }
        assert!(sampler.stats.avg_entropy > 0.0);
    }

    // ── Format / display ───────────────────────────────────────────

    #[test]
    fn format_greedy_config() {
        let config = SamplingConfig::default();
        let s = format_sampling_config(&config);
        assert!(s.contains("greedy"));
    }

    #[test]
    fn format_temperature_config() {
        let config =
            SamplingConfig { method: SamplingMethod::Temperature(0.7), ..Default::default() };
        let s = format_sampling_config(&config);
        assert!(s.contains("temperature"));
        assert!(s.contains("0.7"));
    }

    #[test]
    fn format_with_penalties() {
        let config = SamplingConfig {
            method: SamplingMethod::Greedy,
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            seed: Some(42),
        };
        let s = format_sampling_config(&config);
        assert!(s.contains("rep_pen"));
        assert!(s.contains("freq_pen"));
        assert!(s.contains("pres_pen"));
        assert!(s.contains("seed=42"));
    }

    // ── Beam state ─────────────────────────────────────────────────

    #[test]
    fn beam_state_construction() {
        let beam = BeamState { tokens: vec![1, 2, 3], score: -1.5, finished: false };
        assert_eq!(beam.tokens.len(), 3);
        assert!(!beam.finished);
    }

    #[test]
    fn beam_sampling_via_unified_api() {
        let config =
            SamplingConfig { method: SamplingMethod::Beam { width: 5 }, ..Default::default() };
        let mut sampler = create_sampler(config);
        let logits = make_logits(&[1.0, 3.0, 2.0]);
        let tok = cpu_sample(&mut sampler, &logits).unwrap();
        // Beam degrades to greedy for single token
        assert_eq!(tok.token_id, 1);
    }

    // ── OpenCL kernel source validation ────────────────────────────

    #[test]
    fn opencl_temperature_kernel_not_empty() {
        assert!(!OPENCL_TEMPERATURE_KERNEL.is_empty());
        assert!(OPENCL_TEMPERATURE_KERNEL.contains("temperature_scale"));
    }

    #[test]
    fn opencl_top_k_kernel_not_empty() {
        assert!(!OPENCL_TOP_K_KERNEL.is_empty());
        assert!(OPENCL_TOP_K_KERNEL.contains("top_k_partial_sort"));
    }

    #[test]
    fn opencl_softmax_kernel_not_empty() {
        assert!(!OPENCL_SOFTMAX_KERNEL.is_empty());
        assert!(OPENCL_SOFTMAX_KERNEL.contains("softmax"));
    }

    // ── Mirostat state via create_sampler ──────────────────────────

    #[test]
    fn create_sampler_inits_mirostat_state() {
        let config = SamplingConfig {
            method: SamplingMethod::Mirostat { tau: 5.0, eta: 0.1 },
            seed: Some(1),
            ..Default::default()
        };
        let sampler = create_sampler(config);
        let ms = sampler.mirostat_state.as_ref().unwrap();
        assert!((ms.mu - 10.0).abs() < 1e-6); // mu = 2*tau
        assert!((ms.tau - 5.0).abs() < 1e-6);
        assert!((ms.eta - 0.1).abs() < 1e-6);
    }

    #[test]
    fn create_sampler_no_mirostat_for_greedy() {
        let config = SamplingConfig::default();
        let sampler = create_sampler(config);
        assert!(sampler.mirostat_state.is_none());
    }

    // ── Penalty edge: out-of-range token indices ───────────────────

    #[test]
    fn repetition_penalty_ignores_out_of_range() {
        let mut logits = make_logits(&[1.0, 2.0]);
        let original = logits.clone();
        cpu_apply_repetition_penalty(&mut logits, &[99], 2.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn frequency_penalty_ignores_out_of_range() {
        let mut logits = make_logits(&[1.0, 2.0]);
        let original = logits.clone();
        let mut counts = HashMap::new();
        counts.insert(99u32, 5usize);
        cpu_apply_frequency_penalty(&mut logits, &counts, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn presence_penalty_ignores_out_of_range() {
        let mut logits = make_logits(&[1.0, 2.0]);
        let original = logits.clone();
        let set: HashSet<u32> = [99].into_iter().collect();
        cpu_apply_presence_penalty(&mut logits, &set, 1.0);
        assert_eq!(logits, original);
    }
}
