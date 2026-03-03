//! RoPE (Rotary Position Embedding) configuration.
//!
//! Configure RoPE parameters for different model architectures:
//! base frequency, scaling, NTK-aware extensions, and YaRN.


/// RoPE scaling strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// No scaling (standard RoPE).
    None,
    /// Linear frequency scaling.
    Linear { factor: f64 },
    /// Dynamic NTK-aware scaling.
    DynamicNtk { factor: f64, original_max_pos: usize },
    /// YaRN (Yet another RoPE extension).
    Yarn {
        factor: f64,
        attention_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
    },
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self::None
    }
}

/// Full RoPE configuration.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub base_freq: f64,
    pub max_seq_len: usize,
    pub scaling: RopeScaling,
}

impl RopeConfig {
    pub fn new(head_dim: usize) -> Self {
        Self {
            head_dim,
            base_freq: 10000.0,
            max_seq_len: 4096,
            scaling: RopeScaling::None,
        }
    }

    pub fn with_base_freq(mut self, freq: f64) -> Self {
        self.base_freq = freq;
        self
    }

    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    pub fn with_scaling(mut self, scaling: RopeScaling) -> Self {
        self.scaling = scaling;
        self
    }

    /// Compute inverse frequency table for the given config.
    pub fn inv_freq(&self) -> Vec<f64> {
        let dim = self.head_dim;
        let mut freqs = Vec::with_capacity(dim / 2);
        let base = self.effective_base();

        for i in (0..dim).step_by(2) {
            let freq = 1.0 / base.powf(i as f64 / dim as f64);
            freqs.push(freq);
        }
        freqs
    }

    /// Effective base frequency after scaling.
    pub fn effective_base(&self) -> f64 {
        match &self.scaling {
            RopeScaling::None => self.base_freq,
            RopeScaling::Linear { factor } => self.base_freq * factor,
            RopeScaling::DynamicNtk {
                factor,
                original_max_pos,
            } => {
                let dim = self.head_dim as f64;
                self.base_freq
                    * ((factor * self.max_seq_len as f64 / *original_max_pos as f64)
                        .powf(dim / (dim - 2.0))
                        - 1.0)
                        .max(1.0)
            }
            RopeScaling::Yarn { factor, .. } => {
                self.base_freq * factor
            }
        }
    }

    /// Compute cos/sin tables for positions [0, max_pos).
    pub fn compute_cos_sin(&self, max_pos: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let inv_freq = self.inv_freq();
        let half_dim = inv_freq.len();
        let mut cos_table = Vec::with_capacity(max_pos);
        let mut sin_table = Vec::with_capacity(max_pos);

        for pos in 0..max_pos {
            let mut cos_row = Vec::with_capacity(half_dim);
            let mut sin_row = Vec::with_capacity(half_dim);
            for &freq in &inv_freq {
                let angle = pos as f64 * freq;
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
            }
            cos_table.push(cos_row);
            sin_table.push(sin_row);
        }

        (cos_table, sin_table)
    }
}

/// Preset: BitNet-2B RoPE (standard, 4K context).
pub fn bitnet_rope(head_dim: usize) -> RopeConfig {
    RopeConfig::new(head_dim)
        .with_base_freq(10000.0)
        .with_max_seq_len(4096)
}

/// Preset: Phi-4 RoPE (extended 16K context).
pub fn phi4_rope(head_dim: usize) -> RopeConfig {
    RopeConfig::new(head_dim)
        .with_base_freq(10000.0)
        .with_max_seq_len(16384)
        .with_scaling(RopeScaling::DynamicNtk {
            factor: 4.0,
            original_max_pos: 4096,
        })
}

/// Preset: LLaMA-3 RoPE (extended 8K context).
pub fn llama3_rope(head_dim: usize) -> RopeConfig {
    RopeConfig::new(head_dim)
        .with_base_freq(500000.0)
        .with_max_seq_len(8192)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_config() {
        let config = RopeConfig::new(128);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.base_freq, 10000.0);
        assert_eq!(config.max_seq_len, 4096);
    }

    #[test]
    fn test_inv_freq_length() {
        let config = RopeConfig::new(64);
        let freqs = config.inv_freq();
        assert_eq!(freqs.len(), 32); // dim/2
    }

    #[test]
    fn test_inv_freq_monotonic() {
        let config = RopeConfig::new(128);
        let freqs = config.inv_freq();
        for i in 1..freqs.len() {
            assert!(freqs[i] < freqs[i - 1], "frequencies should decrease");
        }
    }

    #[test]
    fn test_cos_sin_tables() {
        let config = RopeConfig::new(64);
        let (cos, sin) = config.compute_cos_sin(10);
        assert_eq!(cos.len(), 10);
        assert_eq!(sin.len(), 10);
        assert_eq!(cos[0].len(), 32);
        // Position 0: cos=1, sin=0
        for &c in &cos[0] {
            assert!((c - 1.0).abs() < 1e-10);
        }
        for &s in &sin[0] {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_scaling() {
        let base = RopeConfig::new(128);
        let scaled = RopeConfig::new(128)
            .with_scaling(RopeScaling::Linear { factor: 2.0 });
        assert!((scaled.effective_base() - base.effective_base() * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dynamic_ntk_scaling() {
        let config = phi4_rope(128);
        let effective = config.effective_base();
        assert!(effective > 10000.0, "NTK scaling should increase base freq");
    }

    #[test]
    fn test_bitnet_preset() {
        let config = bitnet_rope(128);
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.scaling, RopeScaling::None);
    }

    #[test]
    fn test_phi4_preset() {
        let config = phi4_rope(128);
        assert_eq!(config.max_seq_len, 16384);
        matches!(config.scaling, RopeScaling::DynamicNtk { .. });
    }

    #[test]
    fn test_llama3_preset() {
        let config = llama3_rope(128);
        assert_eq!(config.max_seq_len, 8192);
        assert_eq!(config.base_freq, 500000.0);
    }

    #[test]
    fn test_cos_sin_range() {
        let config = RopeConfig::new(64);
        let (cos, sin) = config.compute_cos_sin(100);
        for pos in 0..100 {
            for i in 0..32 {
                assert!(cos[pos][i] >= -1.0 && cos[pos][i] <= 1.0);
                assert!(sin[pos][i] >= -1.0 && sin[pos][i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_builder_chaining() {
        let config = RopeConfig::new(64)
            .with_base_freq(50000.0)
            .with_max_seq_len(32768)
            .with_scaling(RopeScaling::Linear { factor: 4.0 });
        assert_eq!(config.base_freq, 50000.0);
        assert_eq!(config.max_seq_len, 32768);
    }

    #[test]
    fn test_no_scaling_effective_base() {
        let config = RopeConfig::new(128).with_base_freq(10000.0);
        assert!((config.effective_base() - 10000.0).abs() < 1e-6);
    }
}
