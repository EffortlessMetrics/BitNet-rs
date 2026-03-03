//! Predefined sampling strategy presets.
//!
//! Convenient sampling configurations for common use cases
//! (greedy, creative, balanced, code generation, etc.).

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding (deterministic, always pick argmax).
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0, top_k: 1, ..Default::default() }
    }

    /// Creative: higher temperature, wider sampling.
    pub fn creative() -> Self {
        Self { temperature: 0.9, top_p: 0.95, top_k: 50, ..Default::default() }
    }

    /// Balanced: moderate temperature.
    pub fn balanced() -> Self {
        Self { temperature: 0.7, top_p: 0.9, top_k: 40, ..Default::default() }
    }

    /// Code generation: low temperature for precision.
    pub fn code() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.95,
            top_k: 10,
            repetition_penalty: 1.1,
            ..Default::default()
        }
    }

    /// Chat: slightly creative with repetition control.
    pub fn chat() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.1,
            ..Default::default()
        }
    }

    /// Deterministic with a seed.
    pub fn deterministic(seed: u64) -> Self {
        Self { temperature: 0.0, top_k: 1, seed: Some(seed), ..Default::default() }
    }

    // Builder methods

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_repetition_penalty(mut self, rp: f32) -> Self {
        self.repetition_penalty = rp;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Whether this is greedy/deterministic decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }

    /// Whether a random seed is set.
    pub fn is_seeded(&self) -> bool {
        self.seed.is_some()
    }

    /// Validate parameters are in reasonable ranges.
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0".into());
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        if self.repetition_penalty < 1.0 {
            return Err("repetition_penalty must be >= 1.0".into());
        }
        Ok(())
    }
}

impl std::fmt::Display for SamplingParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_greedy() {
            write!(f, "greedy")?;
        } else {
            write!(
                f,
                "temp={:.1}, top_p={:.2}, top_k={}",
                self.temperature, self.top_p, self.top_k
            )?;
        }
        if self.repetition_penalty != 1.0 {
            write!(f, ", rep_pen={:.2}", self.repetition_penalty)?;
        }
        if let Some(seed) = self.seed {
            write!(f, ", seed={seed}")?;
        }
        Ok(())
    }
}

/// Named preset for display/selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Greedy,
    Creative,
    Balanced,
    Code,
    Chat,
}

impl Preset {
    pub fn to_params(self) -> SamplingParams {
        match self {
            Self::Greedy => SamplingParams::greedy(),
            Self::Creative => SamplingParams::creative(),
            Self::Balanced => SamplingParams::balanced(),
            Self::Code => SamplingParams::code(),
            Self::Chat => SamplingParams::chat(),
        }
    }

    pub fn all() -> &'static [Preset] {
        &[Self::Greedy, Self::Creative, Self::Balanced, Self::Code, Self::Chat]
    }
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Greedy => write!(f, "greedy"),
            Self::Creative => write!(f, "creative"),
            Self::Balanced => write!(f, "balanced"),
            Self::Code => write!(f, "code"),
            Self::Chat => write!(f, "chat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, 1.0);
        assert!(!p.is_greedy());
    }

    #[test]
    fn test_greedy() {
        let p = SamplingParams::greedy();
        assert!(p.is_greedy());
        assert_eq!(p.temperature, 0.0);
    }

    #[test]
    fn test_creative() {
        let p = SamplingParams::creative();
        assert!(!p.is_greedy());
        assert!(p.temperature > 0.5);
    }

    #[test]
    fn test_balanced() {
        let p = SamplingParams::balanced();
        assert!((p.temperature - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_code() {
        let p = SamplingParams::code();
        assert!(p.temperature < 0.5);
        assert!(p.repetition_penalty > 1.0);
    }

    #[test]
    fn test_chat() {
        let p = SamplingParams::chat();
        assert!(p.frequency_penalty > 0.0);
    }

    #[test]
    fn test_deterministic() {
        let p = SamplingParams::deterministic(42);
        assert!(p.is_greedy());
        assert!(p.is_seeded());
        assert_eq!(p.seed, Some(42));
    }

    #[test]
    fn test_builder() {
        let p = SamplingParams::default()
            .with_temperature(0.5)
            .with_top_p(0.9)
            .with_top_k(20)
            .with_repetition_penalty(1.2)
            .with_seed(123);
        assert!((p.temperature - 0.5).abs() < 0.01);
        assert_eq!(p.top_k, 20);
        assert_eq!(p.seed, Some(123));
    }

    #[test]
    fn test_validate_ok() {
        assert!(SamplingParams::greedy().validate().is_ok());
        assert!(SamplingParams::creative().validate().is_ok());
    }

    #[test]
    fn test_validate_bad_temp() {
        let p = SamplingParams::default().with_temperature(-1.0);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_bad_top_p() {
        let p = SamplingParams { top_p: 0.0, ..SamplingParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_bad_rep_pen() {
        let p = SamplingParams { repetition_penalty: 0.5, ..SamplingParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_display_greedy() {
        let s = format!("{}", SamplingParams::greedy());
        assert!(s.contains("greedy"));
    }

    #[test]
    fn test_display_creative() {
        let s = format!("{}", SamplingParams::creative());
        assert!(s.contains("temp="));
    }

    #[test]
    fn test_preset_all() {
        assert_eq!(Preset::all().len(), 5);
    }

    #[test]
    fn test_preset_to_params() {
        let p = Preset::Code.to_params();
        assert!(p.temperature < 0.5);
    }

    #[test]
    fn test_preset_display() {
        assert_eq!(format!("{}", Preset::Greedy), "greedy");
        assert_eq!(format!("{}", Preset::Chat), "chat");
    }

    #[test]
    fn test_not_seeded_by_default() {
        assert!(!SamplingParams::default().is_seeded());
    }
}
