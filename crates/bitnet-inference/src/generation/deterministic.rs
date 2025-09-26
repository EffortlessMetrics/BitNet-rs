//! Deterministic Generation Support
//!
//! Provides deterministic text generation for reproducible results,
//! controlled by environment variables BITNET_DETERMINISTIC and BITNET_SEED.

use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Tensor};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Mutex;

/// Global deterministic state
static DETERMINISTIC_STATE: Mutex<Option<DeterministicState>> = Mutex::new(None);

#[derive(Debug)]
struct DeterministicState {
    seed: u64,
    #[allow(dead_code)]
    rng: ChaCha8Rng,
    enabled: bool,
}

/// Deterministic generator for reproducible inference
#[derive(Debug)]
pub struct DeterministicGenerator {
    rng: ChaCha8Rng,
    seed: u64,
    step_count: usize,
}

impl DeterministicGenerator {
    /// Create new deterministic generator with seed
    pub fn new(seed: u64) -> Result<Self> {
        let rng = ChaCha8Rng::seed_from_u64(seed);

        Ok(Self { rng, seed, step_count: 0 })
    }

    /// Sample token deterministically
    pub async fn sample_deterministic(
        &mut self,
        logits: &BitNetTensor,
        step: usize,
    ) -> Result<(usize, f32)> {
        // For deterministic generation, we use a combination of:
        // 1. Seed-based RNG state
        // 2. Step-dependent behavior
        // 3. Argmax sampling with tie-breaking

        let logits_candle = logits.to_candle()?;

        // Get the last token's logits
        let last_logits = if logits_candle.dims().len() == 3 {
            let (batch, seq_len, vocab_size) = logits_candle.dims3()?;
            logits_candle.narrow(1, seq_len - 1, 1)?.reshape(&[batch, vocab_size])?
        } else {
            logits_candle.clone()
        };

        // Convert to probabilities for deterministic sampling
        let probabilities = candle_nn::ops::softmax(&last_logits, candle_core::D::Minus1)?;
        let prob_vec = probabilities.flatten_all()?.to_vec1::<f32>()?;

        // Deterministic sampling with tie-breaking
        let token_id = self.deterministic_argmax(&prob_vec, step)?;
        let probability = prob_vec[token_id];

        self.step_count += 1;
        Ok((token_id, probability))
    }

    /// Deterministic argmax with reproducible tie-breaking
    fn deterministic_argmax(&mut self, probabilities: &[f32], step: usize) -> Result<usize> {
        if probabilities.is_empty() {
            return Err(anyhow::anyhow!("Empty probability distribution"));
        }

        // Find maximum probability
        let max_prob = probabilities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0);

        // Find all indices with maximum probability
        let max_indices: Vec<usize> = probabilities
            .iter()
            .enumerate()
            .filter(|&(_, &prob)| (prob - max_prob).abs() < 1e-9)
            .map(|(idx, _)| idx)
            .collect();

        if max_indices.len() == 1 {
            return Ok(max_indices[0]);
        }

        // Deterministic tie-breaking using seed and step
        let tie_breaker = self.seed.wrapping_mul(step as u64 + 1) % max_indices.len() as u64;
        Ok(max_indices[tie_breaker as usize])
    }

    /// Reset RNG to initial seed state
    pub fn reset(&mut self) {
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.step_count = 0;
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Update seed and reset
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.reset();
    }
}

/// Set global deterministic seed from environment variables
pub fn set_deterministic_seed() -> Result<()> {
    let enabled = std::env::var("BITNET_DETERMINISTIC").is_ok();

    if !enabled {
        return Ok(());
    }

    let seed = std::env::var("BITNET_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut state = DETERMINISTIC_STATE
        .lock()
        .map_err(|_| anyhow::anyhow!("Failed to lock deterministic state"))?;

    *state = Some(DeterministicState { seed, rng: ChaCha8Rng::seed_from_u64(seed), enabled: true });

    // Also set RAYON thread count to 1 for deterministic parallel execution
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }

    log::info!("Deterministic mode enabled with seed: {}", seed);
    Ok(())
}

/// Check if deterministic mode is enabled
pub fn is_deterministic_enabled() -> bool {
    DETERMINISTIC_STATE
        .lock()
        .ok()
        .and_then(|state| state.as_ref().map(|s| s.enabled))
        .unwrap_or(false)
}

/// Get deterministic seed
pub fn get_deterministic_seed() -> Option<u64> {
    DETERMINISTIC_STATE.lock().ok().and_then(|state| state.as_ref().map(|s| s.seed))
}

/// Sample deterministically from global state
pub fn sample_with_global_deterministic(probabilities: &[f32], step: usize) -> Result<usize> {
    let mut state = DETERMINISTIC_STATE
        .lock()
        .map_err(|_| anyhow::anyhow!("Failed to lock deterministic state"))?;

    if let Some(ref mut det_state) = state.as_mut() {
        if !det_state.enabled {
            return Err(anyhow::anyhow!("Deterministic mode not enabled"));
        }

        // Find maximum probability
        let max_prob = probabilities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0);

        // Find all indices with maximum probability
        let max_indices: Vec<usize> = probabilities
            .iter()
            .enumerate()
            .filter(|&(_, &prob)| (prob - max_prob).abs() < 1e-9)
            .map(|(idx, _)| idx)
            .collect();

        if max_indices.len() == 1 {
            return Ok(max_indices[0]);
        }

        // Deterministic tie-breaking
        let tie_breaker = det_state.seed.wrapping_mul(step as u64 + 1) % max_indices.len() as u64;
        Ok(max_indices[tie_breaker as usize])
    } else {
        Err(anyhow::anyhow!("Deterministic state not initialized"))
    }
}

/// Initialize deterministic state if environment variables are set
pub fn init_deterministic_from_env() -> Result<()> {
    if std::env::var("BITNET_DETERMINISTIC").is_ok() {
        set_deterministic_seed()?;

        // Additional deterministic settings
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            unsafe {
                std::env::set_var("RAYON_NUM_THREADS", "1");
            }
        }

        log::info!("Deterministic inference mode initialized");
    }

    Ok(())
}

/// Deterministic random number generator for testing
pub struct DeterministicRng {
    seed: u64,
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { seed, state: seed }
    }

    pub fn reset(&mut self) {
        self.state = self.seed;
    }
}

impl RngCore for DeterministicRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        // Simple linear congruential generator for deterministic behavior
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            for (i, &byte) in bytes.iter().enumerate() {
                if i < chunk.len() {
                    chunk[i] = byte;
                }
            }
        }
    }

    // try_fill_bytes is not part of RngCore trait in this version
}

/// Validate deterministic behavior
pub fn validate_deterministic_consistency<F>(test_fn: F, iterations: usize) -> Result<bool>
where
    F: Fn() -> Result<Vec<usize>>,
{
    if !is_deterministic_enabled() {
        return Ok(false); // Can't validate if not deterministic
    }

    let mut results = Vec::new();

    for i in 0..iterations {
        // Reset deterministic state before each iteration
        set_deterministic_seed()?;

        let result =
            test_fn().with_context(|| format!("Test function failed on iteration {}", i))?;

        if i == 0 {
            results = result;
        } else if results != result {
            log::error!("Deterministic consistency check failed on iteration {}", i);
            log::error!("Expected: {:?}", results);
            log::error!("Got: {:?}", result);
            return Ok(false);
        }
    }

    log::info!("Deterministic consistency validated across {} iterations", iterations);
    Ok(true)
}
