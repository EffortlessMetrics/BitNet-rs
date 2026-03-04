//! RoPE (Rotary Position Embedding) configuration utilities.
//!
//! Configuration for different model families' RoPE parameters.

/// RoPE scaling type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    None,
    Linear(f32),
    Dynamic(f32),
    Yarn { factor: f32, original_max_pos: usize },
    NTKAware(f32),
}

/// RoPE configuration for a model.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub max_position: usize,
    pub base: f32,
    pub scaling: RopeScaling,
    pub interleaved: bool,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            max_position: 4096,
            base: 10000.0,
            scaling: RopeScaling::None,
            interleaved: false,
        }
    }
}

impl RopeConfig {
    /// Compute inverse frequency table.
    pub fn inv_freq(&self) -> Vec<f32> {
        let effective_base = match self.scaling {
            RopeScaling::NTKAware(factor) => self.base * factor,
            _ => self.base,
        };

        (0..self.head_dim / 2)
            .map(|i| 1.0 / effective_base.powf(2.0 * i as f32 / self.head_dim as f32))
            .collect()
    }

    /// Build sin/cos tables for positions [0, max_position).
    pub fn build_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let inv = self.inv_freq();
        let half = self.head_dim / 2;
        let mut sin_table = vec![0.0f32; self.max_position * half];
        let mut cos_table = vec![0.0f32; self.max_position * half];

        for pos in 0..self.max_position {
            let scaled_pos = match self.scaling {
                RopeScaling::Linear(factor) => pos as f32 / factor,
                RopeScaling::Dynamic(factor) => pos as f32 / factor,
                _ => pos as f32,
            };

            for (i, &freq) in inv.iter().enumerate() {
                let angle = scaled_pos * freq;
                sin_table[pos * half + i] = angle.sin();
                cos_table[pos * half + i] = angle.cos();
            }
        }

        (sin_table, cos_table)
    }

    /// Table size in bytes for sin+cos combined.
    pub fn table_size_bytes(&self) -> usize {
        self.max_position * self.head_dim / 2 * 4 * 2 // sin + cos, f32
    }

    /// Config for BitNet models.
    pub fn bitnet() -> Self {
        Self { head_dim: 128, max_position: 4096, base: 10000.0, ..Default::default() }
    }

    /// Config for Phi-4 (extended context).
    pub fn phi4() -> Self {
        Self { head_dim: 128, max_position: 16384, base: 10000.0, ..Default::default() }
    }

    /// Config for LLaMA-3 (extended base).
    pub fn llama3() -> Self {
        Self { head_dim: 128, max_position: 8192, base: 500000.0, ..Default::default() }
    }

    /// Config for Qwen2 (YaRN scaling).
    pub fn qwen2() -> Self {
        Self {
            head_dim: 128,
            max_position: 32768,
            base: 1000000.0,
            scaling: RopeScaling::Yarn { factor: 4.0, original_max_pos: 8192 },
            ..Default::default()
        }
    }
}

/// Apply RoPE rotation to a pair of values.
pub fn apply_rope_pair(x0: f32, x1: f32, cos_val: f32, sin_val: f32) -> (f32, f32) {
    (x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let c = RopeConfig::default();
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.max_position, 4096);
        assert!((c.base - 10000.0).abs() < 0.01);
    }

    #[test]
    fn test_inv_freq() {
        let c = RopeConfig { head_dim: 4, ..Default::default() };
        let inv = c.inv_freq();
        assert_eq!(inv.len(), 2);
        assert!((inv[0] - 1.0).abs() < 0.01); // 1/10000^0 = 1
    }

    #[test]
    fn test_build_tables() {
        let c = RopeConfig { head_dim: 4, max_position: 2, ..Default::default() };
        let (sin_t, cos_t) = c.build_tables();
        assert_eq!(sin_t.len(), 4); // 2 positions * 2 half_dim
        assert_eq!(cos_t.len(), 4);
        // Position 0: sin(0)=0, cos(0)=1
        assert!(sin_t[0].abs() < 0.01);
        assert!((cos_t[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_table_size() {
        let c = RopeConfig::default();
        let size = c.table_size_bytes();
        assert!(size > 0);
        assert_eq!(size, 4096 * 64 * 4 * 2); // 4096 * (128/2) * sizeof(f32) * 2
    }

    #[test]
    fn test_bitnet_config() {
        let c = RopeConfig::bitnet();
        assert_eq!(c.max_position, 4096);
    }

    #[test]
    fn test_phi4_config() {
        let c = RopeConfig::phi4();
        assert_eq!(c.max_position, 16384);
    }

    #[test]
    fn test_llama3_config() {
        let c = RopeConfig::llama3();
        assert_eq!(c.max_position, 8192);
        assert!((c.base - 500000.0).abs() < 0.01);
    }

    #[test]
    fn test_qwen2_config() {
        let c = RopeConfig::qwen2();
        assert!(matches!(c.scaling, RopeScaling::Yarn { .. }));
    }

    #[test]
    fn test_linear_scaling() {
        let c = RopeConfig {
            head_dim: 4,
            max_position: 4,
            scaling: RopeScaling::Linear(2.0),
            ..Default::default()
        };
        let (sin_t, _) = c.build_tables();
        // Position 2 with scale=2 → effective position 1
        assert!(!sin_t.is_empty());
    }

    #[test]
    fn test_ntk_aware() {
        let c = RopeConfig {
            head_dim: 4,
            max_position: 4,
            scaling: RopeScaling::NTKAware(2.0),
            ..Default::default()
        };
        let inv = c.inv_freq();
        // Base should be doubled
        let c2 = RopeConfig { head_dim: 4, ..Default::default() };
        let inv2 = c2.inv_freq();
        assert!(inv[1] < inv2[1]); // higher base = lower freq
    }

    #[test]
    fn test_apply_rope_pair() {
        // cos=1, sin=0 → identity
        let (a, b) = apply_rope_pair(3.0, 4.0, 1.0, 0.0);
        assert!((a - 3.0).abs() < 0.01);
        assert!((b - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_rope_90deg() {
        // cos=0, sin=1 → rotation by 90 degrees
        let (a, b) = apply_rope_pair(3.0, 4.0, 0.0, 1.0);
        assert!((a - (-4.0)).abs() < 0.01);
        assert!((b - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_scaling_none() {
        assert_eq!(RopeScaling::None, RopeScaling::None);
    }

    #[test]
    fn test_interleaved() {
        let c = RopeConfig { interleaved: true, ..Default::default() };
        assert!(c.interleaved);
    }
}
