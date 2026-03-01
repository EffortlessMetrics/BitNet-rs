//! Shared CLI device preference parser.
//!
//! This crate intentionally stays small and dependency-light so multiple CLI
//! command implementations can consistently parse aliases like `gpu`/`opencl`.

use core::fmt;

/// Canonical device preference requested from CLI flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliDevicePreference {
    Cpu,
    Gpu,
    Metal,
    Auto,
}

impl CliDevicePreference {
    /// Parse a user-provided CLI device string.
    pub fn parse(value: &str) -> Result<Self, ParseDevicePreferenceError> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" | "vulkan" | "opencl" | "ocl" => Ok(Self::Gpu),
            "metal" | "npu" => Ok(Self::Metal),
            "auto" => Ok(Self::Auto),
            _ => Err(ParseDevicePreferenceError { input: value.to_owned() }),
        }
    }

    /// Display stable user-facing allowed values.
    pub const fn allowed_values() -> &'static str {
        "cpu, cuda, gpu, vulkan, opencl, ocl, metal, npu, auto"
    }
}

/// Parsing error for unsupported device values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDevicePreferenceError {
    input: String,
}

impl ParseDevicePreferenceError {
    /// The original user input.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }
}

impl fmt::Display for ParseDevicePreferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Invalid device: {}. Must be one of: {}",
            self.input,
            CliDevicePreference::allowed_values()
        )
    }
}

impl std::error::Error for ParseDevicePreferenceError {}

#[cfg(test)]
mod tests {
    use super::CliDevicePreference;

    #[test]
    fn parses_cpu() {
        assert_eq!(CliDevicePreference::parse("cpu").unwrap(), CliDevicePreference::Cpu);
    }

    #[test]
    fn parses_gpu_aliases() {
        for value in ["cuda", "gpu", "vulkan", "opencl", "ocl"] {
            assert_eq!(CliDevicePreference::parse(value).unwrap(), CliDevicePreference::Gpu);
        }
    }

    #[test]
    fn parses_metal_aliases() {
        for value in ["metal", "npu"] {
            assert_eq!(CliDevicePreference::parse(value).unwrap(), CliDevicePreference::Metal);
        }
    }

    #[test]
    fn parses_auto() {
        assert_eq!(CliDevicePreference::parse("auto").unwrap(), CliDevicePreference::Auto);
    }

    #[test]
    fn rejects_invalid_values() {
        let err = CliDevicePreference::parse("tpu").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid device: tpu. Must be one of: cpu, cuda, gpu, vulkan, opencl, ocl, metal, npu, auto"
        );
    }
}
