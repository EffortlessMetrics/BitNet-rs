//! Sampling configuration for text generation.
//!
//! Defines sampling strategies (greedy, top-k, top-p, temperature)
//! and their composition for controllable generation.

/// Sampling method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    TopK { k: usize },
    TopP { p: f32 },
    Temperature { temp: f32 },
}

impl SamplingMethod {
    pub fn name(&self) -> &'static str {
        match self {
            SamplingMethod::Greedy => "greedy",
            SamplingMethod::TopK { .. } => "top_k",
            SamplingMethod::TopP { .. } => "top_p",
            SamplingMethod::Temperature { .. } => "temperature",
        }
    }
}

/// Complete sampler configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }

    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            repetition_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }

    pub fn deterministic(seed: u64) -> Self {
        Self { seed: Some(seed), ..Self::greedy() }
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn with_repetition_penalty(mut self, rp: f32) -> Self {
        self.repetition_penalty = rp;
        self
    }

    pub fn with_frequency_penalty(mut self, fp: f32) -> Self {
        self.frequency_penalty = fp;
        self
    }

    pub fn with_presence_penalty(mut self, pp: f32) -> Self {
        self.presence_penalty = pp;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Check if this config uses greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Check if this config is deterministic.
    pub fn is_deterministic(&self) -> bool {
        self.is_greedy() || self.seed.is_some()
    }

    /// Validate the configuration. Returns issues if any.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.temperature < 0.0 {
            issues.push("temperature must be >= 0".into());
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                issues.push("top_k must be > 0".into());
            }
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                issues.push("top_p must be in [0.0, 1.0]".into());
            }
        }
        if self.repetition_penalty < 0.0 {
            issues.push("repetition_penalty must be >= 0".into());
        }
        issues
    }

    /// Get the primary sampling method.
    pub fn primary_method(&self) -> SamplingMethod {
        if self.is_greedy() {
            return SamplingMethod::Greedy;
        }
        if let Some(k) = self.top_k {
            return SamplingMethod::TopK { k };
        }
        if let Some(p) = self.top_p {
            return SamplingMethod::TopP { p };
        }
        SamplingMethod::Temperature { temp: self.temperature }
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let c = SamplerConfig::greedy();
        assert!(c.is_greedy());
        assert!(c.is_deterministic());
        assert_eq!(c.primary_method(), SamplingMethod::Greedy);
    }

    #[test]
    fn test_creative() {
        let c = SamplerConfig::creative();
        assert!(!c.is_greedy());
        assert_eq!(c.temperature, 0.9);
        assert_eq!(c.top_k, Some(50));
    }

    #[test]
    fn test_balanced() {
        let c = SamplerConfig::balanced();
        assert_eq!(c.temperature, 0.7);
    }

    #[test]
    fn test_deterministic() {
        let c = SamplerConfig::deterministic(42);
        assert!(c.is_deterministic());
        assert_eq!(c.seed, Some(42));
    }

    #[test]
    fn test_builder() {
        let c = SamplerConfig::greedy()
            .with_temperature(0.5)
            .with_top_k(10)
            .with_top_p(0.9)
            .with_repetition_penalty(1.2)
            .with_seed(123);
        assert_eq!(c.temperature, 0.5);
        assert_eq!(c.top_k, Some(10));
        assert_eq!(c.seed, Some(123));
    }

    #[test]
    fn test_validate_ok() {
        let c = SamplerConfig::balanced();
        assert!(c.validate().is_empty());
    }

    #[test]
    fn test_validate_bad_temp() {
        let c = SamplerConfig::greedy().with_temperature(-1.0);
        assert!(!c.validate().is_empty());
    }

    #[test]
    fn test_validate_bad_top_k() {
        let c = SamplerConfig::greedy().with_top_k(0);
        assert!(!c.validate().is_empty());
    }

    #[test]
    fn test_validate_bad_top_p() {
        let c = SamplerConfig::greedy().with_top_p(1.5);
        assert!(!c.validate().is_empty());
    }

    #[test]
    fn test_primary_method_top_k() {
        let c = SamplerConfig::creative();
        assert!(matches!(c.primary_method(), SamplingMethod::TopK { .. }));
    }

    #[test]
    fn test_sampling_method_name() {
        assert_eq!(SamplingMethod::Greedy.name(), "greedy");
        assert_eq!(SamplingMethod::TopK { k: 5 }.name(), "top_k");
    }

    #[test]
    fn test_default() {
        let c = SamplerConfig::default();
        assert_eq!(c.temperature, 0.7); // balanced default
    }
}
