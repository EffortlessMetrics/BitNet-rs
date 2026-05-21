use std::fmt;

/// An API version (major.minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApiVersion {
    pub major: u16,
    pub minor: u16,
}

impl ApiVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Current API version.
    pub const CURRENT: Self = Self::new(1, 0);

    /// Minimum supported API version.
    pub const MIN_SUPPORTED: Self = Self::new(1, 0);

    /// Check if this version is compatible with another.
    /// Same major version and >= minor version means compatible.
    pub const fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major && self.minor >= other.minor
    }

    /// Check if this version is deprecated.
    pub const fn is_deprecated(&self, min_supported: &Self) -> bool {
        self.major < min_supported.major
            || (self.major == min_supported.major && self.minor < min_supported.minor)
    }

    /// Parse from "v1.0" or "1.0" format.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.strip_prefix('v').unwrap_or(s);
        let mut parts = s.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        Some(Self { major, minor })
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}
