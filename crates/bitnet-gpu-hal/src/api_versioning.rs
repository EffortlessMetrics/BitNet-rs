//! API versioning with deprecation tracking for the GPU HAL layer.
//!
//! Provides semver-based version routing, backward compatibility checking,
//! deprecation tracking with sunset dates, version negotiation, and
//! migration guide generation.

use std::collections::HashMap;
use std::fmt;

// ── ApiVersion ────────────────────────────────────────────────────────────

/// Semantic version for an API endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ApiVersion {
    /// Create a new API version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Parse a version string like `"1.2.3"`.
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        match parts.len() {
            3 => Some(Self {
                major: parts[0].parse().ok()?,
                minor: parts[1].parse().ok()?,
                patch: parts[2].parse().ok()?,
            }),
            2 => Some(Self {
                major: parts[0].parse().ok()?,
                minor: parts[1].parse().ok()?,
                patch: 0,
            }),
            1 => Some(Self {
                major: parts[0].parse().ok()?,
                minor: 0,
                patch: 0,
            }),
            _ => None,
        }
    }

    /// Check if this version is compatible with `other` under semver rules.
    ///
    /// Same major version and `self >= other` means compatible.
    pub const fn is_compatible_with(&self, other: &Self) -> bool {
        if self.major != other.major {
            return false;
        }
        if self.minor > other.minor {
            return true;
        }
        if self.minor == other.minor {
            return self.patch >= other.patch;
        }
        false
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for ApiVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ApiVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

// ── VersionConstraint ─────────────────────────────────────────────────────

/// A semver range constraint for matching API versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionConstraint {
    /// Exact match: `=1.2.3`
    Exact(ApiVersion),
    /// Greater-than-or-equal: `>=1.0.0`
    Gte(ApiVersion),
    /// Less-than: `<2.0.0`
    Lt(ApiVersion),
    /// Tilde: `~1.2` matches `>=1.2.0, <1.3.0`
    Tilde(ApiVersion),
    /// Caret: `^1.2` matches `>=1.2.0, <2.0.0`
    Caret(ApiVersion),
    /// Intersection of two constraints.
    And(Box<VersionConstraint>, Box<VersionConstraint>),
}

impl VersionConstraint {
    /// Parse a constraint string.
    ///
    /// Supports `>=`, `<`, `~`, `^`, `=`, and bare versions (exact).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix(">=") {
            ApiVersion::parse(rest.trim()).map(Self::Gte)
        } else if let Some(rest) = s.strip_prefix('<') {
            ApiVersion::parse(rest.trim()).map(Self::Lt)
        } else if let Some(rest) = s.strip_prefix('~') {
            ApiVersion::parse(rest.trim()).map(Self::Tilde)
        } else if let Some(rest) = s.strip_prefix('^') {
            ApiVersion::parse(rest.trim()).map(Self::Caret)
        } else if let Some(rest) = s.strip_prefix('=') {
            ApiVersion::parse(rest.trim()).map(Self::Exact)
        } else {
            ApiVersion::parse(s).map(Self::Exact)
        }
    }

    /// Check whether a version satisfies this constraint.
    pub fn matches(&self, v: &ApiVersion) -> bool {
        match self {
            Self::Exact(target) => v == target,
            Self::Gte(min) => v >= min,
            Self::Lt(max) => v < max,
            Self::Tilde(base) => {
                v.major == base.major
                    && v.minor == base.minor
                    && v.patch >= base.patch
            }
            Self::Caret(base) => {
                v.major == base.major && v >= base
            }
            Self::And(a, b) => a.matches(v) && b.matches(v),
        }
    }
}

impl fmt::Display for VersionConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(v) => write!(f, "={v}"),
            Self::Gte(v) => write!(f, ">={v}"),
            Self::Lt(v) => write!(f, "<{v}"),
            Self::Tilde(v) => write!(f, "~{v}"),
            Self::Caret(v) => write!(f, "^{v}"),
            Self::And(a, b) => write!(f, "{a}, {b}"),
        }
    }
}

// ── VersionedEndpoint ─────────────────────────────────────────────────────

/// Metadata for a versioned API endpoint.
#[derive(Debug, Clone)]
pub struct VersionedEndpoint {
    /// Endpoint path or identifier.
    pub path: String,
    /// Human-readable description.
    pub description: String,
    /// Version when this endpoint was introduced.
    pub since: ApiVersion,
    /// Version constraint for supported clients.
    pub supported_versions: VersionConstraint,
    /// Whether the endpoint is currently deprecated.
    pub deprecated: bool,
}

impl VersionedEndpoint {
    /// Create a new versioned endpoint.
    pub fn new(
        path: impl Into<String>,
        description: impl Into<String>,
        since: ApiVersion,
        supported: VersionConstraint,
    ) -> Self {
        Self {
            path: path.into(),
            description: description.into(),
            since,
            supported_versions: supported,
            deprecated: false,
        }
    }

    /// Check whether a given version can use this endpoint.
    pub fn supports_version(&self, v: &ApiVersion) -> bool {
        self.supported_versions.matches(v)
    }
}

// ── DeprecationWarning ────────────────────────────────────────────────────

/// A deprecation warning for an API endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeprecationWarning {
    /// The endpoint that is deprecated.
    pub endpoint: String,
    /// Version since which it was deprecated.
    pub deprecated_since: ApiVersion,
    /// Date (ISO 8601) after which the endpoint will be removed.
    pub sunset_date: String,
    /// Guidance for migrating to the replacement.
    pub migration_guide: String,
}

impl fmt::Display for DeprecationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DEPRECATED: '{}' (since {}). Sunset: {}. {}",
            self.endpoint,
            self.deprecated_since,
            self.sunset_date,
            self.migration_guide,
        )
    }
}

// ── DeprecationTracker ────────────────────────────────────────────────────

/// Tracks deprecated API endpoints with sunset dates.
#[derive(Debug, Default)]
pub struct DeprecationTracker {
    warnings: Vec<DeprecationWarning>,
}

impl DeprecationTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a deprecation warning.
    pub fn deprecate(
        &mut self,
        endpoint: impl Into<String>,
        deprecated_since: ApiVersion,
        sunset_date: impl Into<String>,
        migration_guide: impl Into<String>,
    ) {
        self.warnings.push(DeprecationWarning {
            endpoint: endpoint.into(),
            deprecated_since,
            sunset_date: sunset_date.into(),
            migration_guide: migration_guide.into(),
        });
    }

    /// Get all warnings for a specific endpoint.
    pub fn warnings_for(&self, endpoint: &str) -> Vec<&DeprecationWarning> {
        self.warnings.iter().filter(|w| w.endpoint == endpoint).collect()
    }

    /// Get all active warnings.
    pub fn all_warnings(&self) -> &[DeprecationWarning] {
        &self.warnings
    }

    /// Check whether a specific endpoint is deprecated.
    pub fn is_deprecated(&self, endpoint: &str) -> bool {
        self.warnings.iter().any(|w| w.endpoint == endpoint)
    }

    /// Total number of tracked deprecations.
    pub fn count(&self) -> usize {
        self.warnings.len()
    }
}

// ── CompatibilityChecker ──────────────────────────────────────────────────

/// Result of a compatibility check between two API versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compatibility {
    /// Fully compatible (same major, server >= client).
    Full,
    /// Backward-compatible (additive changes only).
    Backward,
    /// Incompatible (major version mismatch or server < client).
    Incompatible { reason: String },
}

/// Checks compatibility between API versions.
pub struct CompatibilityChecker;

impl CompatibilityChecker {
    /// Check whether `server` is compatible with `client`.
    ///
    /// # Rules
    /// - Same major + same minor + same patch → `Full`
    /// - Same major + server >= client → `Backward`
    /// - Different major → `Incompatible`
    pub fn check(server: &ApiVersion, client: &ApiVersion) -> Compatibility {
        if server.major != client.major {
            return Compatibility::Incompatible {
                reason: format!(
                    "major version mismatch: server={}, client={}",
                    server.major, client.major,
                ),
            };
        }
        if server < client {
            return Compatibility::Incompatible {
                reason: format!(
                    "server version {server} is older than client {client}"
                ),
            };
        }
        if server == client {
            Compatibility::Full
        } else {
            Compatibility::Backward
        }
    }
}

// ── VersionNegotiator ─────────────────────────────────────────────────────

/// Negotiates the best API version between client and server.
pub struct VersionNegotiator;

impl VersionNegotiator {
    /// Pick the best mutually-supported version.
    ///
    /// `server_versions` and `client_versions` should each be sorted
    /// ascending. Returns the highest version present in both lists.
    pub fn negotiate(
        server_versions: &[ApiVersion],
        client_versions: &[ApiVersion],
    ) -> Option<ApiVersion> {
        let mut best: Option<ApiVersion> = None;
        for sv in server_versions {
            for cv in client_versions {
                if sv == cv {
                    if best.is_none_or(|b| *sv > b) {
                        best = Some(*sv);
                    }
                }
            }
        }
        best
    }

    /// Find the best compatible version from the server list for a client.
    ///
    /// Returns the highest server version with the same major as the client
    /// and >= the client version.
    pub fn best_compatible(
        server_versions: &[ApiVersion],
        client: &ApiVersion,
    ) -> Option<ApiVersion> {
        server_versions
            .iter()
            .filter(|sv| sv.is_compatible_with(client))
            .max()
            .copied()
    }
}

// ── MigrationGuide ────────────────────────────────────────────────────────

/// A single migration step between consecutive versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationStep {
    /// Source version.
    pub from: ApiVersion,
    /// Target version.
    pub to: ApiVersion,
    /// Human-readable instructions.
    pub instructions: String,
}

/// Generates upgrade instructions between API versions.
#[derive(Debug, Default)]
pub struct MigrationGuide {
    steps: Vec<MigrationStep>,
}

impl MigrationGuide {
    /// Create an empty migration guide.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a migration step.
    pub fn add_step(
        &mut self,
        from: ApiVersion,
        to: ApiVersion,
        instructions: impl Into<String>,
    ) {
        self.steps.push(MigrationStep {
            from,
            to,
            instructions: instructions.into(),
        });
    }

    /// Get the migration path from `from` to `to`.
    ///
    /// Returns an ordered list of steps, or `None` if no path exists.
    pub fn path(
        &self,
        from: &ApiVersion,
        to: &ApiVersion,
    ) -> Option<Vec<&MigrationStep>> {
        if from >= to {
            return None;
        }
        let mut result = Vec::new();
        let mut current = *from;
        while current < *to {
            let step = self.steps.iter().find(|s| s.from == current)?;
            result.push(step);
            current = step.to;
        }
        if current == *to { Some(result) } else { None }
    }

    /// Total number of registered steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

// ── ApiVersionRouter ──────────────────────────────────────────────────────

/// Routes requests to the correct handler based on API version.
///
/// Each handler is identified by a version constraint and a name string.
#[derive(Debug, Default)]
pub struct ApiVersionRouter {
    routes: Vec<(VersionConstraint, String)>,
}

impl ApiVersionRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a handler for a version constraint.
    pub fn add_route(
        &mut self,
        constraint: VersionConstraint,
        handler: impl Into<String>,
    ) {
        self.routes.push((constraint, handler.into()));
    }

    /// Resolve the handler name for a given API version.
    ///
    /// Returns the first matching handler.
    pub fn resolve(&self, version: &ApiVersion) -> Option<&str> {
        self.routes
            .iter()
            .find(|(c, _)| c.matches(version))
            .map(|(_, h)| h.as_str())
    }

    /// Return all handlers matching a version.
    pub fn resolve_all(&self, version: &ApiVersion) -> Vec<&str> {
        self.routes
            .iter()
            .filter(|(c, _)| c.matches(version))
            .map(|(_, h)| h.as_str())
            .collect()
    }

    /// Number of registered routes.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

// ── ApiCatalog ────────────────────────────────────────────────────────────

/// Registry of all API endpoints with version information.
#[derive(Debug, Default)]
pub struct ApiCatalog {
    endpoints: HashMap<String, VersionedEndpoint>,
    deprecations: DeprecationTracker,
}

impl ApiCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an endpoint.
    pub fn register(&mut self, endpoint: VersionedEndpoint) {
        self.endpoints.insert(endpoint.path.clone(), endpoint);
    }

    /// Look up an endpoint by path.
    pub fn get(&self, path: &str) -> Option<&VersionedEndpoint> {
        self.endpoints.get(path)
    }

    /// List all endpoint paths.
    pub fn list_paths(&self) -> Vec<&str> {
        let mut paths: Vec<&str> =
            self.endpoints.keys().map(String::as_str).collect();
        paths.sort_unstable();
        paths
    }

    /// Find all endpoints compatible with a version.
    pub fn endpoints_for_version(
        &self,
        version: &ApiVersion,
    ) -> Vec<&VersionedEndpoint> {
        self.endpoints
            .values()
            .filter(|ep| ep.supports_version(version))
            .collect()
    }

    /// Access the deprecation tracker.
    pub fn deprecations(&self) -> &DeprecationTracker {
        &self.deprecations
    }

    /// Access the deprecation tracker mutably.
    pub fn deprecations_mut(&mut self) -> &mut DeprecationTracker {
        &mut self.deprecations
    }

    /// Number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ApiVersion parsing ────────────────────────────────────────────

    #[test]
    fn parse_full_version() {
        let v = ApiVersion::parse("1.2.3").unwrap();
        assert_eq!(v, ApiVersion::new(1, 2, 3));
    }

    #[test]
    fn parse_major_minor() {
        let v = ApiVersion::parse("2.1").unwrap();
        assert_eq!(v, ApiVersion::new(2, 1, 0));
    }

    #[test]
    fn parse_major_only() {
        let v = ApiVersion::parse("3").unwrap();
        assert_eq!(v, ApiVersion::new(3, 0, 0));
    }

    #[test]
    fn parse_invalid_returns_none() {
        assert!(ApiVersion::parse("").is_none());
        assert!(ApiVersion::parse("abc").is_none());
        assert!(ApiVersion::parse("1.2.3.4").is_none());
    }

    #[test]
    fn version_display() {
        assert_eq!(ApiVersion::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn version_ordering() {
        let v1 = ApiVersion::new(1, 0, 0);
        let v1_1 = ApiVersion::new(1, 1, 0);
        let v2 = ApiVersion::new(2, 0, 0);
        assert!(v1 < v1_1);
        assert!(v1_1 < v2);
        assert!(v2 > v1);
    }

    #[test]
    fn version_equality() {
        assert_eq!(ApiVersion::new(1, 0, 0), ApiVersion::new(1, 0, 0));
        assert_ne!(ApiVersion::new(1, 0, 0), ApiVersion::new(1, 0, 1));
    }

    #[test]
    fn version_compatible_same() {
        let v = ApiVersion::new(1, 2, 3);
        assert!(v.is_compatible_with(&v));
    }

    #[test]
    fn version_compatible_newer_minor() {
        let server = ApiVersion::new(1, 3, 0);
        let client = ApiVersion::new(1, 2, 0);
        assert!(server.is_compatible_with(&client));
    }

    #[test]
    fn version_incompatible_different_major() {
        let v1 = ApiVersion::new(1, 0, 0);
        let v2 = ApiVersion::new(2, 0, 0);
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn version_incompatible_older_minor() {
        let server = ApiVersion::new(1, 1, 0);
        let client = ApiVersion::new(1, 2, 0);
        assert!(!server.is_compatible_with(&client));
    }

    // ── VersionConstraint ─────────────────────────────────────────────

    #[test]
    fn constraint_exact_match() {
        let c = VersionConstraint::parse("=1.2.3").unwrap();
        assert!(c.matches(&ApiVersion::new(1, 2, 3)));
        assert!(!c.matches(&ApiVersion::new(1, 2, 4)));
    }

    #[test]
    fn constraint_gte() {
        let c = VersionConstraint::parse(">=1.0.0").unwrap();
        assert!(c.matches(&ApiVersion::new(1, 0, 0)));
        assert!(c.matches(&ApiVersion::new(2, 0, 0)));
        assert!(!c.matches(&ApiVersion::new(0, 9, 0)));
    }

    #[test]
    fn constraint_lt() {
        let c = VersionConstraint::parse("<2.0.0").unwrap();
        assert!(c.matches(&ApiVersion::new(1, 9, 9)));
        assert!(!c.matches(&ApiVersion::new(2, 0, 0)));
    }

    #[test]
    fn constraint_tilde() {
        let c = VersionConstraint::parse("~1.2.0").unwrap();
        assert!(c.matches(&ApiVersion::new(1, 2, 0)));
        assert!(c.matches(&ApiVersion::new(1, 2, 9)));
        assert!(!c.matches(&ApiVersion::new(1, 3, 0)));
        assert!(!c.matches(&ApiVersion::new(2, 2, 0)));
    }

    #[test]
    fn constraint_caret() {
        let c = VersionConstraint::parse("^1.2.0").unwrap();
        assert!(c.matches(&ApiVersion::new(1, 2, 0)));
        assert!(c.matches(&ApiVersion::new(1, 9, 0)));
        assert!(!c.matches(&ApiVersion::new(2, 0, 0)));
        assert!(!c.matches(&ApiVersion::new(1, 1, 0)));
    }

    #[test]
    fn constraint_and() {
        let c = VersionConstraint::And(
            Box::new(VersionConstraint::Gte(ApiVersion::new(1, 0, 0))),
            Box::new(VersionConstraint::Lt(ApiVersion::new(2, 0, 0))),
        );
        assert!(c.matches(&ApiVersion::new(1, 5, 0)));
        assert!(!c.matches(&ApiVersion::new(0, 9, 0)));
        assert!(!c.matches(&ApiVersion::new(2, 0, 0)));
    }

    #[test]
    fn constraint_display() {
        assert_eq!(
            VersionConstraint::Gte(ApiVersion::new(1, 0, 0)).to_string(),
            ">=1.0.0"
        );
        assert_eq!(
            VersionConstraint::Lt(ApiVersion::new(2, 0, 0)).to_string(),
            "<2.0.0"
        );
    }

    #[test]
    fn constraint_parse_bare_version() {
        let c = VersionConstraint::parse("1.2.3").unwrap();
        assert!(c.matches(&ApiVersion::new(1, 2, 3)));
        assert!(!c.matches(&ApiVersion::new(1, 2, 4)));
    }

    #[test]
    fn constraint_tilde_display() {
        assert_eq!(
            VersionConstraint::Tilde(ApiVersion::new(1, 2, 0)).to_string(),
            "~1.2.0"
        );
    }

    #[test]
    fn constraint_caret_display() {
        assert_eq!(
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)).to_string(),
            "^1.0.0"
        );
    }

    // ── ApiVersionRouter ──────────────────────────────────────────────

    #[test]
    fn router_empty_resolves_none() {
        let router = ApiVersionRouter::new();
        assert!(router.resolve(&ApiVersion::new(1, 0, 0)).is_none());
    }

    #[test]
    fn router_single_route() {
        let mut router = ApiVersionRouter::new();
        router.add_route(
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
            "v1_handler",
        );
        assert_eq!(
            router.resolve(&ApiVersion::new(1, 5, 0)),
            Some("v1_handler")
        );
    }

    #[test]
    fn router_multiple_routes() {
        let mut router = ApiVersionRouter::new();
        router.add_route(
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
            "v1_handler",
        );
        router.add_route(
            VersionConstraint::Caret(ApiVersion::new(2, 0, 0)),
            "v2_handler",
        );
        assert_eq!(
            router.resolve(&ApiVersion::new(1, 3, 0)),
            Some("v1_handler")
        );
        assert_eq!(
            router.resolve(&ApiVersion::new(2, 1, 0)),
            Some("v2_handler")
        );
    }

    #[test]
    fn router_no_match() {
        let mut router = ApiVersionRouter::new();
        router.add_route(
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
            "v1_handler",
        );
        assert!(router.resolve(&ApiVersion::new(2, 0, 0)).is_none());
    }

    #[test]
    fn router_resolve_all() {
        let mut router = ApiVersionRouter::new();
        router.add_route(
            VersionConstraint::Gte(ApiVersion::new(1, 0, 0)),
            "catch_all",
        );
        router.add_route(
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
            "v1_specific",
        );
        let handlers = router.resolve_all(&ApiVersion::new(1, 5, 0));
        assert_eq!(handlers.len(), 2);
    }

    #[test]
    fn router_route_count() {
        let mut router = ApiVersionRouter::new();
        assert_eq!(router.route_count(), 0);
        router.add_route(
            VersionConstraint::Exact(ApiVersion::new(1, 0, 0)),
            "h1",
        );
        assert_eq!(router.route_count(), 1);
    }

    #[test]
    fn router_first_match_wins() {
        let mut router = ApiVersionRouter::new();
        router.add_route(
            VersionConstraint::Gte(ApiVersion::new(1, 0, 0)),
            "first",
        );
        router.add_route(
            VersionConstraint::Gte(ApiVersion::new(1, 0, 0)),
            "second",
        );
        assert_eq!(
            router.resolve(&ApiVersion::new(1, 0, 0)),
            Some("first")
        );
    }

    // ── DeprecationTracker ────────────────────────────────────────────

    #[test]
    fn tracker_empty() {
        let tracker = DeprecationTracker::new();
        assert_eq!(tracker.count(), 0);
        assert!(!tracker.is_deprecated("/v1/infer"));
    }

    #[test]
    fn tracker_add_deprecation() {
        let mut tracker = DeprecationTracker::new();
        tracker.deprecate(
            "/v1/infer",
            ApiVersion::new(1, 5, 0),
            "2025-12-31",
            "Use /v2/infer instead",
        );
        assert!(tracker.is_deprecated("/v1/infer"));
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn tracker_warnings_for_endpoint() {
        let mut tracker = DeprecationTracker::new();
        tracker.deprecate(
            "/v1/infer",
            ApiVersion::new(1, 5, 0),
            "2025-12-31",
            "Use /v2/infer",
        );
        tracker.deprecate(
            "/v1/embed",
            ApiVersion::new(1, 6, 0),
            "2026-03-31",
            "Use /v2/embed",
        );
        let warnings = tracker.warnings_for("/v1/infer");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].sunset_date, "2025-12-31");
    }

    #[test]
    fn tracker_all_warnings() {
        let mut tracker = DeprecationTracker::new();
        tracker.deprecate(
            "/v1/a",
            ApiVersion::new(1, 0, 0),
            "2025-06-30",
            "migrate",
        );
        tracker.deprecate(
            "/v1/b",
            ApiVersion::new(1, 1, 0),
            "2025-09-30",
            "migrate",
        );
        assert_eq!(tracker.all_warnings().len(), 2);
    }

    #[test]
    fn deprecation_warning_display() {
        let w = DeprecationWarning {
            endpoint: "/v1/infer".into(),
            deprecated_since: ApiVersion::new(1, 5, 0),
            sunset_date: "2025-12-31".into(),
            migration_guide: "Use /v2/infer".into(),
        };
        let s = w.to_string();
        assert!(s.contains("DEPRECATED"));
        assert!(s.contains("/v1/infer"));
        assert!(s.contains("2025-12-31"));
    }

    #[test]
    fn tracker_not_deprecated_unregistered() {
        let mut tracker = DeprecationTracker::new();
        tracker.deprecate(
            "/v1/infer",
            ApiVersion::new(1, 5, 0),
            "2025-12-31",
            "migrate",
        );
        assert!(!tracker.is_deprecated("/v2/infer"));
    }

    // ── CompatibilityChecker ──────────────────────────────────────────

    #[test]
    fn compat_full() {
        let v = ApiVersion::new(1, 2, 3);
        assert_eq!(
            CompatibilityChecker::check(&v, &v),
            Compatibility::Full,
        );
    }

    #[test]
    fn compat_backward() {
        let server = ApiVersion::new(1, 3, 0);
        let client = ApiVersion::new(1, 2, 0);
        assert_eq!(
            CompatibilityChecker::check(&server, &client),
            Compatibility::Backward,
        );
    }

    #[test]
    fn compat_incompatible_major() {
        let server = ApiVersion::new(2, 0, 0);
        let client = ApiVersion::new(1, 0, 0);
        assert!(matches!(
            CompatibilityChecker::check(&server, &client),
            Compatibility::Incompatible { .. },
        ));
    }

    #[test]
    fn compat_incompatible_server_older() {
        let server = ApiVersion::new(1, 1, 0);
        let client = ApiVersion::new(1, 2, 0);
        assert!(matches!(
            CompatibilityChecker::check(&server, &client),
            Compatibility::Incompatible { .. },
        ));
    }

    #[test]
    fn compat_backward_patch() {
        let server = ApiVersion::new(1, 2, 5);
        let client = ApiVersion::new(1, 2, 3);
        assert_eq!(
            CompatibilityChecker::check(&server, &client),
            Compatibility::Backward,
        );
    }

    #[test]
    fn compat_incompatible_reason_message() {
        let server = ApiVersion::new(2, 0, 0);
        let client = ApiVersion::new(1, 0, 0);
        if let Compatibility::Incompatible { reason } =
            CompatibilityChecker::check(&server, &client)
        {
            assert!(reason.contains("major version mismatch"));
        } else {
            panic!("expected incompatible");
        }
    }

    // ── VersionNegotiator ─────────────────────────────────────────────

    #[test]
    fn negotiate_exact_overlap() {
        let server = [ApiVersion::new(1, 0, 0), ApiVersion::new(2, 0, 0)];
        let client = [ApiVersion::new(1, 0, 0), ApiVersion::new(2, 0, 0)];
        assert_eq!(
            VersionNegotiator::negotiate(&server, &client),
            Some(ApiVersion::new(2, 0, 0)),
        );
    }

    #[test]
    fn negotiate_partial_overlap() {
        let server = [ApiVersion::new(1, 0, 0), ApiVersion::new(2, 0, 0)];
        let client = [ApiVersion::new(2, 0, 0), ApiVersion::new(3, 0, 0)];
        assert_eq!(
            VersionNegotiator::negotiate(&server, &client),
            Some(ApiVersion::new(2, 0, 0)),
        );
    }

    #[test]
    fn negotiate_no_overlap() {
        let server = [ApiVersion::new(1, 0, 0)];
        let client = [ApiVersion::new(2, 0, 0)];
        assert!(VersionNegotiator::negotiate(&server, &client).is_none());
    }

    #[test]
    fn negotiate_empty_server() {
        let client = [ApiVersion::new(1, 0, 0)];
        assert!(VersionNegotiator::negotiate(&[], &client).is_none());
    }

    #[test]
    fn negotiate_empty_client() {
        let server = [ApiVersion::new(1, 0, 0)];
        assert!(VersionNegotiator::negotiate(&server, &[]).is_none());
    }

    #[test]
    fn best_compatible_found() {
        let server = [
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 1, 0),
            ApiVersion::new(1, 2, 0),
            ApiVersion::new(2, 0, 0),
        ];
        let client = ApiVersion::new(1, 1, 0);
        assert_eq!(
            VersionNegotiator::best_compatible(&server, &client),
            Some(ApiVersion::new(1, 2, 0)),
        );
    }

    #[test]
    fn best_compatible_none() {
        let server = [ApiVersion::new(2, 0, 0)];
        let client = ApiVersion::new(1, 0, 0);
        assert!(
            VersionNegotiator::best_compatible(&server, &client).is_none()
        );
    }

    // ── MigrationGuide ────────────────────────────────────────────────

    #[test]
    fn migration_empty() {
        let guide = MigrationGuide::new();
        assert_eq!(guide.step_count(), 0);
    }

    #[test]
    fn migration_single_step() {
        let mut guide = MigrationGuide::new();
        guide.add_step(
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 1, 0),
            "Add new field to request",
        );
        let path = guide
            .path(&ApiVersion::new(1, 0, 0), &ApiVersion::new(1, 1, 0))
            .unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].instructions, "Add new field to request");
    }

    #[test]
    fn migration_multi_step() {
        let mut guide = MigrationGuide::new();
        guide.add_step(
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 1, 0),
            "Step 1",
        );
        guide.add_step(
            ApiVersion::new(1, 1, 0),
            ApiVersion::new(2, 0, 0),
            "Step 2",
        );
        let path = guide
            .path(&ApiVersion::new(1, 0, 0), &ApiVersion::new(2, 0, 0))
            .unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn migration_no_path() {
        let guide = MigrationGuide::new();
        assert!(
            guide
                .path(
                    &ApiVersion::new(1, 0, 0),
                    &ApiVersion::new(2, 0, 0)
                )
                .is_none()
        );
    }

    #[test]
    fn migration_reverse_returns_none() {
        let mut guide = MigrationGuide::new();
        guide.add_step(
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 1, 0),
            "upgrade",
        );
        assert!(
            guide
                .path(
                    &ApiVersion::new(1, 1, 0),
                    &ApiVersion::new(1, 0, 0)
                )
                .is_none()
        );
    }

    #[test]
    fn migration_same_version_returns_none() {
        let guide = MigrationGuide::new();
        let v = ApiVersion::new(1, 0, 0);
        assert!(guide.path(&v, &v).is_none());
    }

    // ── VersionedEndpoint ─────────────────────────────────────────────

    #[test]
    fn endpoint_supports_version() {
        let ep = VersionedEndpoint::new(
            "/infer",
            "Run inference",
            ApiVersion::new(1, 0, 0),
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
        );
        assert!(ep.supports_version(&ApiVersion::new(1, 5, 0)));
        assert!(!ep.supports_version(&ApiVersion::new(2, 0, 0)));
    }

    #[test]
    fn endpoint_deprecated_flag() {
        let mut ep = VersionedEndpoint::new(
            "/old",
            "Old endpoint",
            ApiVersion::new(0, 1, 0),
            VersionConstraint::Exact(ApiVersion::new(0, 1, 0)),
        );
        assert!(!ep.deprecated);
        ep.deprecated = true;
        assert!(ep.deprecated);
    }

    // ── ApiCatalog ────────────────────────────────────────────────────

    #[test]
    fn catalog_empty() {
        let catalog = ApiCatalog::new();
        assert_eq!(catalog.endpoint_count(), 0);
        assert!(catalog.list_paths().is_empty());
    }

    #[test]
    fn catalog_register_and_get() {
        let mut catalog = ApiCatalog::new();
        catalog.register(VersionedEndpoint::new(
            "/infer",
            "Inference",
            ApiVersion::new(1, 0, 0),
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
        ));
        assert_eq!(catalog.endpoint_count(), 1);
        assert!(catalog.get("/infer").is_some());
        assert!(catalog.get("/missing").is_none());
    }

    #[test]
    fn catalog_list_paths_sorted() {
        let mut catalog = ApiCatalog::new();
        catalog.register(VersionedEndpoint::new(
            "/z",
            "Z",
            ApiVersion::new(1, 0, 0),
            VersionConstraint::Gte(ApiVersion::new(1, 0, 0)),
        ));
        catalog.register(VersionedEndpoint::new(
            "/a",
            "A",
            ApiVersion::new(1, 0, 0),
            VersionConstraint::Gte(ApiVersion::new(1, 0, 0)),
        ));
        assert_eq!(catalog.list_paths(), vec!["/a", "/z"]);
    }

    #[test]
    fn catalog_endpoints_for_version() {
        let mut catalog = ApiCatalog::new();
        catalog.register(VersionedEndpoint::new(
            "/v1",
            "V1",
            ApiVersion::new(1, 0, 0),
            VersionConstraint::Caret(ApiVersion::new(1, 0, 0)),
        ));
        catalog.register(VersionedEndpoint::new(
            "/v2",
            "V2",
            ApiVersion::new(2, 0, 0),
            VersionConstraint::Caret(ApiVersion::new(2, 0, 0)),
        ));
        let eps = catalog.endpoints_for_version(&ApiVersion::new(1, 5, 0));
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].path, "/v1");
    }

    #[test]
    fn catalog_deprecations() {
        let mut catalog = ApiCatalog::new();
        catalog.deprecations_mut().deprecate(
            "/v1/infer",
            ApiVersion::new(1, 5, 0),
            "2025-12-31",
            "Use /v2/infer",
        );
        assert!(catalog.deprecations().is_deprecated("/v1/infer"));
    }

    // ── Integration / edge cases ──────────────────────────────────────

    #[test]
    fn constraint_gte_boundary() {
        let c = VersionConstraint::Gte(ApiVersion::new(1, 0, 0));
        assert!(c.matches(&ApiVersion::new(1, 0, 0)));
        assert!(!c.matches(&ApiVersion::new(0, 99, 99)));
    }

    #[test]
    fn constraint_lt_boundary() {
        let c = VersionConstraint::Lt(ApiVersion::new(2, 0, 0));
        assert!(c.matches(&ApiVersion::new(1, 99, 99)));
        assert!(!c.matches(&ApiVersion::new(2, 0, 0)));
    }

    #[test]
    fn negotiate_picks_highest_common() {
        let server = [
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 1, 0),
            ApiVersion::new(1, 2, 0),
        ];
        let client = [
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 2, 0),
        ];
        assert_eq!(
            VersionNegotiator::negotiate(&server, &client),
            Some(ApiVersion::new(1, 2, 0)),
        );
    }

    #[test]
    fn version_compatible_newer_patch() {
        let server = ApiVersion::new(1, 2, 5);
        let client = ApiVersion::new(1, 2, 3);
        assert!(server.is_compatible_with(&client));
    }

    #[test]
    fn version_incompatible_older_patch() {
        let server = ApiVersion::new(1, 2, 1);
        let client = ApiVersion::new(1, 2, 3);
        assert!(!server.is_compatible_with(&client));
    }

    #[test]
    fn router_with_and_constraint() {
        let mut router = ApiVersionRouter::new();
        router.add_route(
            VersionConstraint::And(
                Box::new(VersionConstraint::Gte(ApiVersion::new(1, 0, 0))),
                Box::new(VersionConstraint::Lt(ApiVersion::new(2, 0, 0))),
            ),
            "v1_range",
        );
        assert_eq!(
            router.resolve(&ApiVersion::new(1, 5, 0)),
            Some("v1_range"),
        );
        assert!(router.resolve(&ApiVersion::new(2, 0, 0)).is_none());
    }

    #[test]
    fn catalog_overwrite_endpoint() {
        let mut catalog = ApiCatalog::new();
        catalog.register(VersionedEndpoint::new(
            "/infer",
            "V1",
            ApiVersion::new(1, 0, 0),
            VersionConstraint::Exact(ApiVersion::new(1, 0, 0)),
        ));
        catalog.register(VersionedEndpoint::new(
            "/infer",
            "V2",
            ApiVersion::new(2, 0, 0),
            VersionConstraint::Exact(ApiVersion::new(2, 0, 0)),
        ));
        assert_eq!(catalog.endpoint_count(), 1);
        assert_eq!(catalog.get("/infer").unwrap().description, "V2");
    }

    #[test]
    fn version_zero_zero_zero() {
        let v = ApiVersion::new(0, 0, 0);
        assert_eq!(v.to_string(), "0.0.0");
        assert!(v.is_compatible_with(&v));
    }

    #[test]
    fn constraint_and_display() {
        let c = VersionConstraint::And(
            Box::new(VersionConstraint::Gte(ApiVersion::new(1, 0, 0))),
            Box::new(VersionConstraint::Lt(ApiVersion::new(2, 0, 0))),
        );
        assert_eq!(c.to_string(), ">=1.0.0, <2.0.0");
    }

    #[test]
    fn endpoint_since_version() {
        let ep = VersionedEndpoint::new(
            "/new",
            "New endpoint",
            ApiVersion::new(2, 1, 0),
            VersionConstraint::Gte(ApiVersion::new(2, 1, 0)),
        );
        assert_eq!(ep.since, ApiVersion::new(2, 1, 0));
    }

    #[test]
    fn migration_step_count() {
        let mut guide = MigrationGuide::new();
        guide.add_step(
            ApiVersion::new(1, 0, 0),
            ApiVersion::new(1, 1, 0),
            "s1",
        );
        guide.add_step(
            ApiVersion::new(1, 1, 0),
            ApiVersion::new(1, 2, 0),
            "s2",
        );
        assert_eq!(guide.step_count(), 2);
    }
}
