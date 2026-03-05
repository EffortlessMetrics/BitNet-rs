//! Reusable API versioning primitives for `BitNet` services.

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
    pub fn is_deprecated(&self, min_supported: &Self) -> bool {
        self < min_supported
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

/// Version negotiation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationResult {
    /// Exact match or compatible version found.
    Accepted(ApiVersion),
    /// Client version is deprecated but still functional.
    Deprecated { accepted: ApiVersion, sunset_version: ApiVersion },
    /// No compatible version found.
    Rejected { requested: ApiVersion, supported: Vec<ApiVersion> },
}

/// Supported version range.
#[derive(Debug, Clone)]
pub struct VersionRange {
    pub versions: Vec<ApiVersion>,
    pub current: ApiVersion,
    pub min_supported: ApiVersion,
}

impl VersionRange {
    pub const fn new(
        versions: Vec<ApiVersion>,
        current: ApiVersion,
        min_supported: ApiVersion,
    ) -> Self {
        Self { versions, current, min_supported }
    }

    pub fn default_range() -> Self {
        Self {
            versions: vec![ApiVersion::new(1, 0)],
            current: ApiVersion::CURRENT,
            min_supported: ApiVersion::MIN_SUPPORTED,
        }
    }

    /// Negotiate a version with the client.
    pub fn negotiate(&self, requested: &ApiVersion) -> NegotiationResult {
        // Find best compatible version
        let compatible: Vec<_> = self
            .versions
            .iter()
            .filter(|v| v.major == requested.major && v.minor <= requested.minor)
            .copied()
            .collect();

        if let Some(&best) = compatible.last() {
            if best.is_deprecated(&self.min_supported) {
                NegotiationResult::Deprecated { accepted: best, sunset_version: self.min_supported }
            } else {
                NegotiationResult::Accepted(best)
            }
        } else {
            NegotiationResult::Rejected { requested: *requested, supported: self.versions.clone() }
        }
    }

    pub fn is_supported(&self, version: &ApiVersion) -> bool {
        self.versions.iter().any(|v| v.is_compatible_with(version))
    }
}

/// Extract API version from a URL path prefix like "/v1/..." or "/api/v1.0/...".
#[must_use]
pub fn extract_version_from_path(path: &str) -> Option<ApiVersion> {
    for segment in path.split('/') {
        if let Some(version) = ApiVersion::parse(segment)
            && version.major > 0
        {
            return Some(version);
        }
    }
    None
}

/// Format a version header value.
#[must_use]
pub fn version_header(version: &ApiVersion) -> String {
    format!("application/json; version={version}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let v = ApiVersion::new(1, 2);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(format!("{v}"), "v1.2");
    }

    #[test]
    fn test_version_parse() {
        assert_eq!(ApiVersion::parse("v1.0"), Some(ApiVersion::new(1, 0)));
        assert_eq!(ApiVersion::parse("2.3"), Some(ApiVersion::new(2, 3)));
        assert_eq!(ApiVersion::parse("v1"), Some(ApiVersion::new(1, 0)));
    }

    #[test]
    fn test_version_compatibility() {
        let v1_2 = ApiVersion::new(1, 2);
        let v1_0 = ApiVersion::new(1, 0);
        assert!(v1_2.is_compatible_with(&v1_0));
        assert!(!v1_0.is_compatible_with(&v1_2));
    }

    #[test]
    fn test_version_ordering() {
        assert!(ApiVersion::new(1, 0) < ApiVersion::new(1, 1));
        assert!(ApiVersion::new(1, 1) < ApiVersion::new(2, 0));
    }

    #[test]
    fn test_negotiate_accepted() {
        let range = VersionRange::default_range();
        let result = range.negotiate(&ApiVersion::new(1, 0));
        assert_eq!(result, NegotiationResult::Accepted(ApiVersion::new(1, 0)));
    }

    #[test]
    fn test_negotiate_rejected() {
        let range = VersionRange::default_range();
        let result = range.negotiate(&ApiVersion::new(2, 0));
        assert!(matches!(result, NegotiationResult::Rejected { .. }));
    }

    #[test]
    fn test_extract_version_from_path() {
        assert_eq!(extract_version_from_path("/v1/chat"), Some(ApiVersion::new(1, 0)));
        assert_eq!(extract_version_from_path("/api/v2.1/models"), Some(ApiVersion::new(2, 1)));
        assert_eq!(extract_version_from_path("/health"), None);
    }

    #[test]
    fn test_version_header() {
        let h = version_header(&ApiVersion::new(1, 0));
        assert_eq!(h, "application/json; version=v1.0");
    }

    #[test]
    fn test_is_deprecated() {
        let v = ApiVersion::new(0, 9);
        let min = ApiVersion::new(1, 0);
        assert!(v.is_deprecated(&min));
        assert!(!min.is_deprecated(&min));
    }

    #[test]
    fn test_is_supported() {
        let range = VersionRange::default_range();
        assert!(range.is_supported(&ApiVersion::new(1, 0)));
        assert!(!range.is_supported(&ApiVersion::new(2, 0)));
    }

    #[test]
    fn test_parse_invalid() {
        assert!(ApiVersion::parse("abc").is_none());
        assert!(ApiVersion::parse("").is_none());
    }

    #[test]
    fn test_current_version() {
        assert_eq!(ApiVersion::CURRENT, ApiVersion::new(1, 0));
    }
}
