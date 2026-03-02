//! Core BitNet feature taxonomy + feature-set operations.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

/// Canonical feature axes for feature-flag contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BitnetFeature {
    Cpu,
    Gpu,
    Cuda,
    Metal,
    Vulkan,
    Oneapi,
    Inference,
    Kernels,
    Tokenizers,
    Quantization,
    Cli,
    Server,
    Ffi,
    Python,
    Wasm,
    CrossValidation,
    Trace,
    Iq2sFfi,
    CppFfi,
    Fixtures,
    Reporting,
    Trend,
    IntegrationTests,
}

impl fmt::Display for BitnetFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Metal => write!(f, "metal"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::Oneapi => write!(f, "oneapi"),
            Self::Inference => write!(f, "inference"),
            Self::Kernels => write!(f, "kernels"),
            Self::Tokenizers => write!(f, "tokenizers"),
            Self::Quantization => write!(f, "quantization"),
            Self::Cli => write!(f, "cli"),
            Self::Server => write!(f, "server"),
            Self::Ffi => write!(f, "ffi"),
            Self::Python => write!(f, "python"),
            Self::Wasm => write!(f, "wasm"),
            Self::CrossValidation => write!(f, "crossval"),
            Self::Trace => write!(f, "trace"),
            Self::Iq2sFfi => write!(f, "iq2s-ffi"),
            Self::CppFfi => write!(f, "cpp-ffi"),
            Self::Fixtures => write!(f, "fixtures"),
            Self::Reporting => write!(f, "reporting"),
            Self::Trend => write!(f, "trend"),
            Self::IntegrationTests => write!(f, "integration-tests"),
        }
    }
}

impl FromStr for BitnetFeature {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "vulkan" => Ok(Self::Vulkan),
            "oneapi" => Ok(Self::Oneapi),
            "inference" => Ok(Self::Inference),
            "kernels" => Ok(Self::Kernels),
            "tokenizers" => Ok(Self::Tokenizers),
            "quantization" => Ok(Self::Quantization),
            "cli" => Ok(Self::Cli),
            "server" => Ok(Self::Server),
            "ffi" => Ok(Self::Ffi),
            "python" => Ok(Self::Python),
            "wasm" => Ok(Self::Wasm),
            "crossval" | "cross-validation" => Ok(Self::CrossValidation),
            "trace" => Ok(Self::Trace),
            "iq2s-ffi" => Ok(Self::Iq2sFfi),
            "cpp-ffi" => Ok(Self::CppFfi),
            "fixtures" => Ok(Self::Fixtures),
            "reporting" => Ok(Self::Reporting),
            "trend" => Ok(Self::Trend),
            "integration-tests" => Ok(Self::IntegrationTests),
            _ => Err("unknown feature"),
        }
    }
}

/// Ordered set of supported features.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeatureSet(BTreeSet<BitnetFeature>);

impl FeatureSet {
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    pub fn insert(&mut self, feature: BitnetFeature) -> bool {
        self.0.insert(feature)
    }

    pub fn contains(&self, feature: BitnetFeature) -> bool {
        self.0.contains(&feature)
    }

    pub fn extend<I>(&mut self, features: I)
    where
        I: IntoIterator<Item = BitnetFeature>,
    {
        self.0.extend(features);
    }

    pub fn from_names<I, S>(features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set = Self::new();
        for feature in features {
            if let Ok(feature) = feature.as_ref().parse() {
                set.insert(feature);
            }
        }
        set
    }

    pub fn labels(&self) -> Vec<String> {
        self.0.iter().map(ToString::to_string).collect()
    }

    pub fn missing_required(&self, required: &Self) -> Self {
        Self(required.0.difference(&self.0).copied().collect())
    }

    pub fn forbidden_overlap(&self, forbidden: &Self) -> Self {
        Self(self.0.intersection(&forbidden.0).copied().collect())
    }

    pub fn satisfies(&self, required: &Self, forbidden: &Self) -> bool {
        self.missing_required(required).is_empty() && self.forbidden_overlap(forbidden).is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &BitnetFeature> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<&[BitnetFeature]> for FeatureSet {
    fn from(value: &[BitnetFeature]) -> Self {
        Self(value.iter().copied().collect())
    }
}

impl From<&[&str]> for FeatureSet {
    fn from(value: &[&str]) -> Self {
        Self::from_names(value.iter().copied())
    }
}

/// Canonical helper for mapping runtime feature selections to `FeatureSet`.
pub fn feature_set_from_names(features: &[&str]) -> FeatureSet {
    FeatureSet::from_names(features.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parser_ignores_unknown_features() {
        let set = feature_set_from_names(&["inference", "unknown-feature", "kernels"]);
        assert!(set.contains(BitnetFeature::Inference));
        assert!(set.contains(BitnetFeature::Kernels));
        assert!(!set.contains(BitnetFeature::Gpu));
    }
}
