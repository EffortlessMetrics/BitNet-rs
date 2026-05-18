//! Shared compatibility value types.

use super::labels::normalize_label;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostArch {
    X86,
    Arm,
    Unknown,
}

impl HostArch {
    pub fn from_label(label: &str) -> Self {
        let label = normalize_label(label);
        if label.contains("x86") || label.contains("amd64") || label.contains("x64") {
            Self::X86
        } else if label.contains("arm") || label.contains("aarch64") {
            Self::Arm
        } else {
            Self::Unknown
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86 => "x86",
            Self::Arm => "arm",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitnetKernel {
    I2S,
    Tl1,
    Tl2,
    Unknown,
}

impl BitnetKernel {
    pub fn from_label(label: &str) -> Self {
        let label = normalize_label(label);
        if label.contains("i2_s") || label.contains("i2s") {
            Self::I2S
        } else if label.contains("tl1") {
            Self::Tl1
        } else if label.contains("tl2") {
            Self::Tl2
        } else {
            Self::Unknown
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I2S => "i2_s",
            Self::Tl1 => "tl1",
            Self::Tl2 => "tl2",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKernelSupport {
    SupportedReference,
    Supported,
    ListedSupportedVerifyRunner,
    UnsupportedUpstream,
    Unknown,
}

impl ModelKernelSupport {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SupportedReference => "supported_reference",
            Self::Supported => "supported",
            Self::ListedSupportedVerifyRunner => "listed_supported_verify_runner",
            Self::UnsupportedUpstream => "unsupported_upstream",
            Self::Unknown => "unknown",
        }
    }
}
