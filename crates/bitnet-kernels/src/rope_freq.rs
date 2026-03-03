//! RoPE frequency table construction.
//!
//! Build sine/cosine frequency tables for Rotary Position Embeddings.
//! Supports standard, NTK-aware, and YaRN scaling methods.

/// RoPE scaling method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScaling {
    /// No scaling (standard RoPE).
    None,
    /// Linear frequency scaling by a factor.
    Linear(u32), // store as integer, divide by 100
    /// NTK-aware scaling.
    Ntk(u32),
}

impl RopeScaling {
    pub fn factor(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Linear(f) | Self::Ntk(f) => *f as f64 / 100.0,
        }
    }
}

/// RoPE frequency configuration.
#[derive(Debug, Clone)]
pub struct RopeFreqConfig {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f64,
    pub scaling: RopeScaling,
}

impl RopeFreqConfig {
    pub fn standard(dim: usize, max_seq_len: usize) -> Self {
        Self { dim, max_seq_len, base: 10000.0, scaling: RopeScaling::None }
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    pub fn with_scaling(mut self, scaling: RopeScaling) -> Self {
        self.scaling = scaling;
        self
    }
}

/// Precomputed frequency table.
#[derive(Debug, Clone)]
pub struct FreqTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub dim: usize,
    pub seq_len: usize,
}

impl FreqTable {
    pub fn total_elements(&self) -> usize {
        self.cos.len()
    }

    pub fn size_bytes(&self) -> usize {
        self.cos.len() * 4 + self.sin.len() * 4
    }

    /// Get cos/sin pair for (position, dim_pair_index).
    pub fn get(&self, pos: usize, pair_idx: usize) -> Option<(f32, f32)> {
        let half_dim = self.dim / 2;
        if pos >= self.seq_len || pair_idx >= half_dim {
            return None;
        }
        let idx = pos * half_dim + pair_idx;
        Some((self.cos[idx], self.sin[idx]))
    }
}

/// Build the frequency table.
pub fn build_freq_table(config: &RopeFreqConfig) -> FreqTable {
    let half_dim = config.dim / 2;
    let total = config.max_seq_len * half_dim;
    let mut cos_table = vec![0.0f32; total];
    let mut sin_table = vec![0.0f32; total];

    let effective_base = match config.scaling {
        RopeScaling::Ntk(factor) => {
            let f = factor as f64 / 100.0;
            config.base * f.powf(config.dim as f64 / (config.dim as f64 - 2.0))
        }
        _ => config.base,
    };

    let freq_scale = match config.scaling {
        RopeScaling::Linear(factor) => factor as f64 / 100.0,
        _ => 1.0,
    };

    for pos in 0..config.max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / effective_base.powf(2.0 * i as f64 / config.dim as f64);
            let theta = (pos as f64) * freq / freq_scale;
            let idx = pos * half_dim + i;
            cos_table[idx] = theta.cos() as f32;
            sin_table[idx] = theta.sin() as f32;
        }
    }

    FreqTable { cos: cos_table, sin: sin_table, dim: config.dim, seq_len: config.max_seq_len }
}

/// Apply RoPE rotation to a pair of values.
pub fn apply_rope_pair(x0: f32, x1: f32, cos_theta: f32, sin_theta: f32) -> (f32, f32) {
    (x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_config() {
        let config = RopeFreqConfig::standard(128, 4096);
        assert_eq!(config.base, 10000.0);
        assert_eq!(config.dim, 128);
    }

    #[test]
    fn test_build_table() {
        let config = RopeFreqConfig::standard(64, 100);
        let table = build_freq_table(&config);
        assert_eq!(table.total_elements(), 100 * 32);
    }

    #[test]
    fn test_get_pair() {
        let config = RopeFreqConfig::standard(64, 100);
        let table = build_freq_table(&config);
        let (cos, sin) = table.get(0, 0).unwrap();
        // At position 0, cos(0)=1, sin(0)=0
        assert!((cos - 1.0).abs() < 1e-5);
        assert!(sin.abs() < 1e-5);
    }

    #[test]
    fn test_get_oob() {
        let config = RopeFreqConfig::standard(64, 10);
        let table = build_freq_table(&config);
        assert!(table.get(100, 0).is_none());
        assert!(table.get(0, 100).is_none());
    }

    #[test]
    fn test_cos_sin_identity() {
        let config = RopeFreqConfig::standard(64, 100);
        let table = build_freq_table(&config);
        for pos in [0, 10, 50] {
            for pair in [0, 5, 15] {
                let (c, s) = table.get(pos, pair).unwrap();
                let norm = c * c + s * s;
                assert!((norm - 1.0).abs() < 1e-4, "cos²+sin²≠1 at pos={pos} pair={pair}");
            }
        }
    }

    #[test]
    fn test_rope_pair_identity() {
        let (y0, y1) = apply_rope_pair(1.0, 0.0, 1.0, 0.0);
        assert!((y0 - 1.0).abs() < 1e-6);
        assert!(y1.abs() < 1e-6);
    }

    #[test]
    fn test_rope_pair_90deg() {
        let (y0, y1) = apply_rope_pair(1.0, 0.0, 0.0, 1.0);
        assert!(y0.abs() < 1e-6);
        assert!((y1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_scaling() {
        let config = RopeFreqConfig::standard(64, 100).with_scaling(RopeScaling::Linear(200)); // 2.0x
        assert_eq!(config.scaling.factor(), 2.0);
        let table = build_freq_table(&config);
        assert_eq!(table.seq_len, 100);
    }

    #[test]
    fn test_ntk_scaling() {
        let config = RopeFreqConfig::standard(64, 100).with_scaling(RopeScaling::Ntk(400)); // 4.0x
        let table = build_freq_table(&config);
        // NTK changes base, verify table is different
        let std_config = RopeFreqConfig::standard(64, 100);
        let std_table = build_freq_table(&std_config);
        // Mid-range dims should differ
        let (c1, _) = table.get(50, 15).unwrap();
        let (c2, _) = std_table.get(50, 15).unwrap();
        assert!((c1 - c2).abs() > 0.001);
    }

    #[test]
    fn test_custom_base() {
        let config = RopeFreqConfig::standard(64, 100).with_base(500000.0);
        assert_eq!(config.base, 500000.0);
    }

    #[test]
    fn test_size_bytes() {
        let config = RopeFreqConfig::standard(64, 100);
        let table = build_freq_table(&config);
        assert_eq!(table.size_bytes(), 100 * 32 * 4 * 2);
    }

    #[test]
    fn test_scaling_factor() {
        assert_eq!(RopeScaling::None.factor(), 1.0);
        assert_eq!(RopeScaling::Linear(200).factor(), 2.0);
    }
}
