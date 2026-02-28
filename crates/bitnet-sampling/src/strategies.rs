//! Advanced sampling strategies: min-p, typical, mirostat, repetition penalty,
//! and composable sampler chains.
//!
//! These types build on the pure logits transforms in `bitnet-logits` and
//! integrate with the existing [`SamplingStrategy`](super::SamplingStrategy).

use anyhow::Result;
use bitnet_logits::{
    apply_min_p, apply_temperature, apply_top_k, apply_top_p, apply_typical, argmax,
    softmax_in_place,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::debug;

// ---------------------------------------------------------------------------
// MinPSampler
// ---------------------------------------------------------------------------

/// Min-p sampler: filters tokens below a dynamic probability threshold.
///
/// The threshold is `min_p * max_probability`, so the filter adapts to model
/// confidence — keeping more candidates when the model is uncertain and fewer
/// when it is confident.
///
/// # Examples
///
/// ```
/// use bitnet_sampling::MinPSampler;
///
/// let sampler = MinPSampler::new(0.1);
/// let mut probs = vec![0.5f32, 0.3, 0.1, 0.04, 0.04];
/// sampler.filter(&mut probs);
/// assert!(probs[3] == 0.0); // below 0.1 * 0.5 = 0.05 threshold
/// ```
#[derive(Debug, Clone)]
pub struct MinPSampler {
    pub min_p: f32,
}

impl MinPSampler {
    pub fn new(min_p: f32) -> Self {
        Self { min_p: min_p.clamp(0.0, 1.0) }
    }

    /// Filter probabilities in-place, zeroing entries below the threshold.
    pub fn filter(&self, probs: &mut [f32]) {
        apply_min_p(probs, self.min_p);
    }
}

// ---------------------------------------------------------------------------
// TypicalSampler
// ---------------------------------------------------------------------------

/// Locally typical sampler (Meister et al., 2023).
///
/// Keeps tokens whose information content (surprise) is closest to the
/// entropy of the distribution, up to a cumulative probability of `typical_p`.
///
/// # Examples
///
/// ```
/// use bitnet_sampling::TypicalSampler;
///
/// let sampler = TypicalSampler::new(0.9);
/// let mut probs = vec![0.4f32, 0.3, 0.2, 0.1];
/// sampler.filter(&mut probs);
/// let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
/// assert!(non_zero >= 1);
/// ```
#[derive(Debug, Clone)]
pub struct TypicalSampler {
    pub typical_p: f32,
}

impl TypicalSampler {
    pub fn new(typical_p: f32) -> Self {
        Self { typical_p: typical_p.clamp(0.0, 1.0) }
    }

    /// Filter probabilities in-place, keeping "typical" tokens.
    pub fn filter(&self, probs: &mut [f32]) {
        apply_typical(probs, self.typical_p);
    }
}

// ---------------------------------------------------------------------------
// MirostatSampler (v2)
// ---------------------------------------------------------------------------

/// Mirostat v2 adaptive sampler (Basu et al., 2021).
///
/// Targets a specific perplexity by dynamically adjusting a surprise threshold
/// (`mu`). Tokens with surprise exceeding `mu` are filtered out, and `mu` is
/// updated after each sample to track the target surprise `tau`.
///
/// # Examples
///
/// ```
/// use bitnet_sampling::MirostatSampler;
///
/// let mut sampler = MirostatSampler::new(5.0, 0.1, Some(42));
/// let logits = vec![2.0f32, 1.0, 0.5, -1.0, -2.0];
/// let token = sampler.sample(&logits).unwrap();
/// assert!((token as usize) < logits.len());
/// ```
#[derive(Debug, Clone)]
pub struct MirostatSampler {
    /// Target surprise (τ). Higher values allow more randomness.
    pub tau: f32,
    /// Learning rate (η) for mu updates.
    pub eta: f32,
    /// Current surprise threshold, initialised to `2 * tau`.
    pub mu: f32,
    rng: ChaCha8Rng,
}

impl MirostatSampler {
    pub fn new(tau: f32, eta: f32, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(&mut rand::rng()),
        };
        Self { tau, eta, mu: 2.0 * tau, rng }
    }

    /// Sample a token from logits using Mirostat v2 adaptive filtering.
    ///
    /// Returns the selected token ID. Updates internal `mu` state.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(anyhow::anyhow!("Empty logits slice"));
        }

        // Convert logits to probabilities
        let mut probs = logits.to_vec();
        softmax_in_place(&mut probs);

        // Filter: keep tokens whose surprise <= mu
        // surprise(i) = -ln(p(i))
        for p in probs.iter_mut() {
            if *p > 0.0 {
                let surprise = -p.ln();
                if surprise > self.mu {
                    *p = 0.0;
                }
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            // All tokens filtered — fall back to argmax of original logits
            let token = argmax(logits) as u32;
            // Update mu towards tau
            let surprise = {
                let mut orig = logits.to_vec();
                softmax_in_place(&mut orig);
                let p = orig[token as usize];
                if p > 0.0 { -p.ln() } else { self.tau }
            };
            self.mu -= self.eta * (surprise - self.tau);
            return Ok(token);
        }
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }

        // Sample from filtered distribution
        let random_val: f32 = self.rng.random();
        let mut cumulative = 0.0;
        let mut selected = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if random_val <= cumulative {
                selected = i;
                break;
            }
        }

        // Update mu: mu = mu - eta * (surprise - tau)
        let selected_prob = probs[selected];
        let surprise = if selected_prob > 0.0 { -selected_prob.ln() } else { self.tau };
        self.mu -= self.eta * (surprise - self.tau);

        debug!("Mirostat v2: token={}, surprise={:.3}, mu={:.3}", selected, surprise, self.mu);
        Ok(selected as u32)
    }

    /// Reset mu to its initial value (`2 * tau`).
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
    }
}

// ---------------------------------------------------------------------------
// RepetitionPenaltyConfig
// ---------------------------------------------------------------------------

/// Configurable repetition penalty with frequency, presence, and count modes.
///
/// - **Frequency penalty**: proportional to occurrence count
///   (`logit -= freq_penalty * count`).
/// - **Presence penalty**: flat penalty for any token seen at least once
///   (`logit -= presence_penalty` if count > 0).
/// - **Count penalty**: multiplicative penalty that increases with count
///   (`logit /= penalty ^ count` for positive logits).
///
/// # Examples
///
/// ```
/// use bitnet_sampling::RepetitionPenaltyConfig;
///
/// let config = RepetitionPenaltyConfig {
///     frequency_penalty: 0.5,
///     presence_penalty: 0.3,
///     count_penalty: 1.0, // 1.0 = disabled
/// };
/// let mut logits = vec![2.0f32, 1.0, 3.0, 0.5];
/// let token_counts = vec![(0u32, 3usize), (2, 1)];
/// config.apply(&mut logits, &token_counts);
/// // Token 0: 2.0 - 0.5*3 - 0.3 = 0.2
/// assert!((logits[0] - 0.2).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyConfig {
    /// Penalty per occurrence count (0.0 = disabled).
    pub frequency_penalty: f32,
    /// Flat penalty if token has been seen at all (0.0 = disabled).
    pub presence_penalty: f32,
    /// Multiplicative penalty base (1.0 = disabled). Applied as `logit / penalty^count`.
    pub count_penalty: f32,
}

impl Default for RepetitionPenaltyConfig {
    fn default() -> Self {
        Self { frequency_penalty: 0.0, presence_penalty: 0.0, count_penalty: 1.0 }
    }
}

impl RepetitionPenaltyConfig {
    /// Apply all configured penalties to logits in-place.
    ///
    /// `token_counts` is a slice of `(token_id, occurrence_count)` pairs.
    pub fn apply(&self, logits: &mut [f32], token_counts: &[(u32, usize)]) {
        for &(token_id, count) in token_counts {
            let idx = token_id as usize;
            if idx >= logits.len() || count == 0 {
                continue;
            }

            // Frequency penalty: proportional to count
            logits[idx] -= self.frequency_penalty * count as f32;

            // Presence penalty: flat if seen at all
            logits[idx] -= self.presence_penalty;

            // Count penalty: multiplicative
            #[allow(clippy::float_cmp)]
            if self.count_penalty != 1.0 {
                let penalty = self.count_penalty.powi(count as i32);
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SamplerChain
// ---------------------------------------------------------------------------

/// A composable chain of sampling transformers.
///
/// Stages are applied in order. Each stage transforms logits or probabilities
/// in-place. The final stage samples a token from the resulting distribution.
///
/// # Examples
///
/// ```
/// use bitnet_sampling::{SamplerChain, SamplerStage};
///
/// let chain = SamplerChain::builder()
///     .temperature(0.8)
///     .top_k(50)
///     .top_p(0.9)
///     .min_p(0.05)
///     .build(Some(42));
///
/// let logits = vec![2.0f32, 1.0, 0.5, -1.0];
/// let token = chain.sample(&logits).unwrap();
/// assert!((token as usize) < logits.len());
/// ```
pub struct SamplerChain {
    stages: Vec<SamplerStage>,
    rng: std::cell::RefCell<ChaCha8Rng>,
}

/// Individual sampling stage in a [`SamplerChain`].
#[derive(Debug, Clone)]
pub enum SamplerStage {
    /// Temperature scaling (logit domain).
    Temperature(f32),
    /// Top-k filtering (logit domain).
    TopK(usize),
    /// Top-p / nucleus filtering (probability domain, applied after softmax).
    TopP(f32),
    /// Min-p filtering (probability domain, applied after softmax).
    MinP(f32),
    /// Typical sampling filter (probability domain, applied after softmax).
    Typical(f32),
    /// Repetition penalty (logit domain).
    RepetitionPenalty(RepetitionPenaltyConfig, Vec<(u32, usize)>),
}

impl SamplerChain {
    /// Create a new chain from explicit stages.
    pub fn new(stages: Vec<SamplerStage>, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(&mut rand::rng()),
        };
        Self { stages, rng: std::cell::RefCell::new(rng) }
    }

    /// Start building a chain with a fluent API.
    pub fn builder() -> SamplerChainBuilder {
        SamplerChainBuilder { stages: Vec::new() }
    }

    /// Sample a token by running all stages in order.
    ///
    /// Logit-domain stages run first, then softmax converts to probabilities,
    /// then probability-domain stages run, then a token is sampled.
    pub fn sample(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(anyhow::anyhow!("Empty logits slice"));
        }

        let mut buf = logits.to_vec();

        // Partition stages: logit-domain first, then probability-domain
        let mut needs_softmax = false;
        for stage in &self.stages {
            match stage {
                SamplerStage::Temperature(t) => {
                    apply_temperature(&mut buf, *t);
                }
                SamplerStage::TopK(k) => {
                    apply_top_k(&mut buf, *k);
                }
                SamplerStage::RepetitionPenalty(config, counts) => {
                    config.apply(&mut buf, counts);
                }
                SamplerStage::TopP(_) | SamplerStage::MinP(_) | SamplerStage::Typical(_) => {
                    if !needs_softmax {
                        softmax_in_place(&mut buf);
                        needs_softmax = true;
                    }
                    match stage {
                        SamplerStage::TopP(p) => apply_top_p(&mut buf, *p),
                        SamplerStage::MinP(p) => apply_min_p(&mut buf, *p),
                        SamplerStage::Typical(p) => apply_typical(&mut buf, *p),
                        _ => unreachable!(),
                    }
                }
            }
        }

        // Convert to probabilities if not already done
        if !needs_softmax {
            softmax_in_place(&mut buf);
        }

        // Renormalize
        let sum: f32 = buf.iter().sum();
        if sum > 0.0 && (sum - 1.0).abs() > 1e-6 {
            let inv_sum = 1.0 / sum;
            for p in buf.iter_mut() {
                *p *= inv_sum;
            }
        }

        // Sample
        let random_val: f32 = self.rng.borrow_mut().random();
        let mut cumulative = 0.0;
        for (i, &p) in buf.iter().enumerate() {
            cumulative += p;
            if random_val <= cumulative {
                return Ok(i as u32);
            }
        }
        Ok((buf.len() - 1) as u32)
    }

    /// Get the list of stages.
    pub fn stages(&self) -> &[SamplerStage] {
        &self.stages
    }
}

/// Builder for [`SamplerChain`].
pub struct SamplerChainBuilder {
    stages: Vec<SamplerStage>,
}

impl SamplerChainBuilder {
    pub fn temperature(mut self, t: f32) -> Self {
        self.stages.push(SamplerStage::Temperature(t));
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.stages.push(SamplerStage::TopK(k));
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.stages.push(SamplerStage::TopP(p));
        self
    }

    pub fn min_p(mut self, p: f32) -> Self {
        self.stages.push(SamplerStage::MinP(p));
        self
    }

    pub fn typical(mut self, p: f32) -> Self {
        self.stages.push(SamplerStage::Typical(p));
        self
    }

    pub fn repetition_penalty(
        mut self,
        config: RepetitionPenaltyConfig,
        counts: Vec<(u32, usize)>,
    ) -> Self {
        self.stages.push(SamplerStage::RepetitionPenalty(config, counts));
        self
    }

    pub fn build(self, seed: Option<u64>) -> SamplerChain {
        SamplerChain::new(self.stages, seed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- MinPSampler -------------------------------------------------------

    #[test]
    fn min_p_sampler_basic_filtering() {
        let sampler = MinPSampler::new(0.2);
        let mut probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        sampler.filter(&mut probs);
        // Threshold = 0.2 * 0.5 = 0.1
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert!(probs[2] > 0.0);
        assert_eq!(probs[3], 0.0);
        assert_eq!(probs[4], 0.0);
    }

    #[test]
    fn min_p_sampler_clamps_value() {
        let sampler = MinPSampler::new(1.5);
        assert_eq!(sampler.min_p, 1.0);
        let sampler = MinPSampler::new(-0.5);
        assert_eq!(sampler.min_p, 0.0);
    }

    #[test]
    fn min_p_sampler_zero_is_noop() {
        let sampler = MinPSampler::new(0.0);
        let mut probs = vec![0.5, 0.3, 0.2];
        let original = probs.clone();
        sampler.filter(&mut probs);
        assert_eq!(probs, original);
    }

    // --- TypicalSampler ----------------------------------------------------

    #[test]
    fn typical_sampler_filters_atypical() {
        let sampler = TypicalSampler::new(0.5);
        let mut probs = vec![0.5, 0.25, 0.15, 0.07, 0.03];
        sampler.filter(&mut probs);
        let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(non_zero >= 1);
        assert!(non_zero < 5);
    }

    #[test]
    fn typical_sampler_one_is_noop() {
        let sampler = TypicalSampler::new(1.0);
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        let original = probs.clone();
        sampler.filter(&mut probs);
        assert_eq!(probs, original);
    }

    #[test]
    fn typical_sampler_clamps_value() {
        let sampler = TypicalSampler::new(1.5);
        assert_eq!(sampler.typical_p, 1.0);
    }

    // --- MirostatSampler ---------------------------------------------------

    #[test]
    fn mirostat_samples_valid_token() {
        let mut sampler = MirostatSampler::new(5.0, 0.1, Some(42));
        let logits = vec![2.0f32, 1.0, 0.5, -1.0, -2.0];
        let token = sampler.sample(&logits).unwrap();
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn mirostat_mu_converges() {
        let mut sampler = MirostatSampler::new(3.0, 0.1, Some(123));
        let logits = vec![5.0f32, 1.0, 0.5, 0.1, -1.0, -2.0, -3.0, -4.0];

        let initial_mu = sampler.mu;
        for _ in 0..50 {
            let _ = sampler.sample(&logits).unwrap();
        }
        // After many samples, mu should have changed from initial
        assert!((sampler.mu - initial_mu).abs() > 0.001);
    }

    #[test]
    fn mirostat_reset_restores_mu() {
        let mut sampler = MirostatSampler::new(5.0, 0.1, Some(42));
        let logits = vec![2.0, 1.0, 0.5];
        let _ = sampler.sample(&logits).unwrap();
        let _ = sampler.sample(&logits).unwrap();
        sampler.reset();
        assert_eq!(sampler.mu, 2.0 * sampler.tau);
    }

    #[test]
    fn mirostat_deterministic_with_seed() {
        let logits = vec![2.0f32, 1.0, 0.5, -1.0, -2.0];
        let mut s1 = MirostatSampler::new(5.0, 0.1, Some(42));
        let mut s2 = MirostatSampler::new(5.0, 0.1, Some(42));
        let t1 = s1.sample(&logits).unwrap();
        let t2 = s2.sample(&logits).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn mirostat_empty_logits_errors() {
        let mut sampler = MirostatSampler::new(5.0, 0.1, Some(42));
        assert!(sampler.sample(&[]).is_err());
    }

    // --- RepetitionPenaltyConfig -------------------------------------------

    #[test]
    fn repetition_penalty_frequency() {
        let config = RepetitionPenaltyConfig { frequency_penalty: 0.5, ..Default::default() };
        let mut logits = vec![2.0f32, 1.0, 3.0];
        config.apply(&mut logits, &[(0, 3)]);
        // 2.0 - 0.5 * 3 = 0.5
        assert!((logits[0] - 0.5).abs() < 1e-6);
        // Untouched
        assert!((logits[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_presence() {
        let config = RepetitionPenaltyConfig { presence_penalty: 0.3, ..Default::default() };
        let mut logits = vec![2.0f32, 1.0, 3.0];
        config.apply(&mut logits, &[(0, 1), (2, 5)]);
        // Both get -0.3 regardless of count
        assert!((logits[0] - 1.7).abs() < 1e-6);
        assert!((logits[2] - 2.7).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_count_multiplicative() {
        let config = RepetitionPenaltyConfig { count_penalty: 2.0, ..Default::default() };
        let mut logits = vec![8.0f32, 1.0];
        config.apply(&mut logits, &[(0, 3)]);
        // 8.0 / 2.0^3 = 1.0
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_combined() {
        let config = RepetitionPenaltyConfig {
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            count_penalty: 1.0,
        };
        let mut logits = vec![2.0f32, 1.0, 3.0, 0.5];
        config.apply(&mut logits, &[(0, 3), (2, 1)]);
        // Token 0: 2.0 - 0.5*3 - 0.3 = 0.2
        assert!((logits[0] - 0.2).abs() < 1e-6);
        // Token 2: 3.0 - 0.5*1 - 0.3 = 2.2
        assert!((logits[2] - 2.2).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_default_is_noop() {
        let config = RepetitionPenaltyConfig::default();
        let mut logits = vec![2.0f32, 1.0, 3.0];
        let original = logits.clone();
        config.apply(&mut logits, &[(0, 5), (1, 3)]);
        assert_eq!(logits, original);
    }

    #[test]
    fn repetition_penalty_zero_count_is_noop() {
        let config = RepetitionPenaltyConfig {
            frequency_penalty: 1.0,
            presence_penalty: 1.0,
            count_penalty: 2.0,
        };
        let mut logits = vec![2.0f32, 1.0];
        let original = logits.clone();
        config.apply(&mut logits, &[(0, 0)]);
        assert_eq!(logits, original);
    }

    // --- SamplerChain ------------------------------------------------------

    #[test]
    fn chain_temperature_only() {
        let chain = SamplerChain::builder().temperature(0.5).build(Some(42));
        let logits = vec![2.0f32, 1.0, 0.5, -1.0];
        let token = chain.sample(&logits).unwrap();
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn chain_full_pipeline() {
        let chain = SamplerChain::builder()
            .temperature(0.8)
            .top_k(3)
            .top_p(0.9)
            .min_p(0.05)
            .build(Some(42));
        let logits = vec![2.0f32, 1.0, 0.5, -1.0, -2.0];
        let token = chain.sample(&logits).unwrap();
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn chain_deterministic_with_seed() {
        let logits = vec![2.0f32, 1.0, 0.5, -1.0, -2.0];
        let c1 = SamplerChain::builder().temperature(0.8).top_k(3).build(Some(42));
        let c2 = SamplerChain::builder().temperature(0.8).top_k(3).build(Some(42));
        assert_eq!(c1.sample(&logits).unwrap(), c2.sample(&logits).unwrap());
    }

    #[test]
    fn chain_empty_logits_errors() {
        let chain = SamplerChain::builder().temperature(0.8).build(Some(42));
        assert!(chain.sample(&[]).is_err());
    }

    #[test]
    fn chain_with_typical() {
        let chain = SamplerChain::builder().temperature(0.8).typical(0.9).build(Some(42));
        let logits = vec![3.0f32, 1.0, 0.5, -1.0, -2.0];
        let token = chain.sample(&logits).unwrap();
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn chain_stages_count() {
        let chain = SamplerChain::builder()
            .temperature(0.8)
            .top_k(50)
            .top_p(0.9)
            .min_p(0.05)
            .build(Some(42));
        assert_eq!(chain.stages().len(), 4);
    }

    #[test]
    fn chain_with_repetition_penalty() {
        let config = RepetitionPenaltyConfig {
            frequency_penalty: 0.5,
            presence_penalty: 0.0,
            count_penalty: 1.0,
        };
        let chain = SamplerChain::builder()
            .repetition_penalty(config, vec![(0, 5)])
            .temperature(0.8)
            .build(Some(42));
        let logits = vec![2.0f32, 1.0, 0.5];
        let token = chain.sample(&logits).unwrap();
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn chain_single_token_vocab() {
        let chain = SamplerChain::builder().temperature(0.8).top_k(5).top_p(0.9).build(Some(42));
        let logits = vec![1.0f32];
        let token = chain.sample(&logits).unwrap();
        assert_eq!(token, 0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mirostat_always_returns_valid_token(
            logits in proptest::collection::vec(-10.0f32..10.0f32, 2..64),
            seed in 0u64..1000u64,
        ) {
            let mut sampler = MirostatSampler::new(5.0, 0.1, Some(seed));
            let token = sampler.sample(&logits).unwrap();
            prop_assert!((token as usize) < logits.len());
        }

        #[test]
        fn chain_always_returns_valid_token(
            logits in proptest::collection::vec(-10.0f32..10.0f32, 2..64),
            seed in 0u64..1000u64,
        ) {
            let chain = SamplerChain::builder()
                .temperature(0.8)
                .top_k(10)
                .top_p(0.9)
                .min_p(0.05)
                .build(Some(seed));
            let token = chain.sample(&logits).unwrap();
            prop_assert!((token as usize) < logits.len());
        }
    }
}
