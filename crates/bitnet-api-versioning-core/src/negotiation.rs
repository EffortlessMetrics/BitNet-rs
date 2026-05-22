use crate::api_version::ApiVersion;

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
        let compatible = self.compatible_versions(requested);

        if let Some(&best) = compatible.last() {
            self.acceptance_result(best)
        } else {
            NegotiationResult::Rejected { requested: *requested, supported: self.versions.clone() }
        }
    }

    pub fn is_supported(&self, version: &ApiVersion) -> bool {
        self.versions.iter().any(|v| v.is_compatible_with(version))
    }

    fn compatible_versions(&self, requested: &ApiVersion) -> Vec<ApiVersion> {
        self.versions
            .iter()
            .filter(|v| v.major == requested.major && v.minor <= requested.minor)
            .copied()
            .collect()
    }

    fn acceptance_result(&self, best: ApiVersion) -> NegotiationResult {
        if best.is_deprecated(&self.min_supported) {
            NegotiationResult::Deprecated { accepted: best, sunset_version: self.min_supported }
        } else {
            NegotiationResult::Accepted(best)
        }
    }
}
