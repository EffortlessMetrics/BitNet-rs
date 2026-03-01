#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PipelineConfigInput {
    hidden_size: u32,
    n_heads: u16,
    n_kv_heads: u16,
    n_layers: u16,
    intermediate_size: u32,
    head_dim: u16,
    vocab_size: u32,
    max_seq_len: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
}

/// Minimal transformer layer config for validation fuzzing.
#[derive(Debug)]
struct TransformerLayerConfig {
    hidden_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_layers: usize,
    intermediate_size: usize,
    head_dim: usize,
    vocab_size: usize,
    max_seq_len: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

#[derive(Debug)]
enum ConfigError {
    ZeroField,
    HeadDimMismatch,
    KvHeadsExceedHeads,
    HeadsDivisibility,
    InvalidEps,
    InvalidTheta,
    TooLarge,
}

impl TransformerLayerConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 {
            return Err(ConfigError::ZeroField);
        }
        if self.n_heads == 0 {
            return Err(ConfigError::ZeroField);
        }
        if self.n_kv_heads == 0 {
            return Err(ConfigError::ZeroField);
        }
        if self.n_layers == 0 {
            return Err(ConfigError::ZeroField);
        }
        if self.vocab_size == 0 {
            return Err(ConfigError::ZeroField);
        }
        if self.max_seq_len == 0 {
            return Err(ConfigError::ZeroField);
        }

        // Head dimension must divide hidden_size evenly.
        let expected_head_dim = self.hidden_size / self.n_heads;
        if expected_head_dim == 0 || self.hidden_size % self.n_heads != 0 {
            return Err(ConfigError::HeadDimMismatch);
        }
        if self.head_dim != expected_head_dim {
            return Err(ConfigError::HeadDimMismatch);
        }

        // KV heads must not exceed attention heads.
        if self.n_kv_heads > self.n_heads {
            return Err(ConfigError::KvHeadsExceedHeads);
        }

        // n_heads must be divisible by n_kv_heads (GQA constraint).
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(ConfigError::HeadsDivisibility);
        }

        // Epsilon must be positive and finite.
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ConfigError::InvalidEps);
        }

        // Rope theta must be positive and finite.
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(ConfigError::InvalidTheta);
        }

        // Sanity bounds.
        if self.hidden_size > 65536 {
            return Err(ConfigError::TooLarge);
        }
        if self.n_layers > 1024 {
            return Err(ConfigError::TooLarge);
        }

        Ok(())
    }

    /// Compute the total parameter count estimate for the transformer stack.
    fn estimated_params(&self) -> u64 {
        let hs = self.hidden_size as u64;
        let ff = self.intermediate_size as u64;
        let nl = self.n_layers as u64;
        let vs = self.vocab_size as u64;
        // Rough estimate: embedding + n_layers * (4*hs*hs + 2*hs*ff) + output
        vs * hs + nl * (4 * hs * hs + 2 * hs * ff) + vs * hs
    }
}

fuzz_target!(|input: PipelineConfigInput| {
    let config = TransformerLayerConfig {
        hidden_size: input.hidden_size as usize,
        n_heads: input.n_heads as usize,
        n_kv_heads: input.n_kv_heads as usize,
        n_layers: input.n_layers as usize,
        intermediate_size: input.intermediate_size as usize,
        head_dim: input.head_dim as usize,
        vocab_size: input.vocab_size as usize,
        max_seq_len: input.max_seq_len as usize,
        rms_norm_eps: input.rms_norm_eps,
        rope_theta: input.rope_theta,
    };

    match config.validate() {
        Ok(()) => {
            // Invariant 1: Valid config has non-zero fields.
            assert!(config.hidden_size > 0);
            assert!(config.n_heads > 0);
            assert!(config.n_kv_heads > 0);
            assert!(config.n_layers > 0);
            assert!(config.vocab_size > 0);

            // Invariant 2: hidden_size divisible by n_heads.
            assert_eq!(
                config.hidden_size % config.n_heads,
                0,
                "hidden_size must be divisible by n_heads"
            );

            // Invariant 3: head_dim matches hidden_size / n_heads.
            assert_eq!(config.head_dim, config.hidden_size / config.n_heads, "head_dim mismatch");

            // Invariant 4: GQA constraint — n_heads divisible by n_kv_heads.
            assert_eq!(
                config.n_heads % config.n_kv_heads,
                0,
                "n_heads must be divisible by n_kv_heads"
            );

            // Invariant 5: n_kv_heads <= n_heads.
            assert!(config.n_kv_heads <= config.n_heads);

            // Invariant 6: Param estimate is non-zero for valid config.
            let params = config.estimated_params();
            assert!(params > 0, "valid config should have non-zero estimated params");

            // Invariant 7: eps and theta are positive finite.
            assert!(config.rms_norm_eps > 0.0 && config.rms_norm_eps.is_finite());
            assert!(config.rope_theta > 0.0 && config.rope_theta.is_finite());
        }
        Err(_) => {
            // Validation error is expected for bad inputs — no panic.
        }
    }

    // Also fuzz the BitNetConfig builder with the same dimensions.
    let result = bitnet_common::BitNetConfig::builder()
        .hidden_size(input.hidden_size as usize)
        .num_heads(input.n_heads as usize)
        .num_key_value_heads(input.n_kv_heads as usize)
        .num_layers(input.n_layers as usize)
        .vocab_size(input.vocab_size as usize)
        .max_length(input.max_seq_len as usize)
        .build();

    match result {
        Ok(cfg) => {
            let _ = cfg.validate();
        }
        Err(_) => {}
    }
});
