//! Reusable compile-time build metadata helpers.
//!
//! Build scripts set `VERGEN_*` values for the crate they compile. Call
//! [`BuildMetadata::from_env`] from that crate so metadata is captured in the
//! caller's compile-time environment, not this helper crate's environment.

const fn env_or_unknown(value: Option<&'static str>) -> &'static str {
    match value {
        Some(value) => value,
        None => "unknown",
    }
}

/// Shared compile-time build metadata values produced by `vergen`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BuildMetadata {
    pub git_sha: &'static str,
    pub git_branch: &'static str,
    pub build_timestamp: &'static str,
    pub rustc_semver: &'static str,
    pub cargo_target_triple: &'static str,
    pub cargo_opt_level: &'static str,
}

impl BuildMetadata {
    /// Resolve build metadata from compile-time `vergen` environment variables.
    #[must_use]
    pub const fn from_env(
        git_sha: Option<&'static str>,
        git_branch: Option<&'static str>,
        build_timestamp: Option<&'static str>,
        rustc_semver: Option<&'static str>,
        cargo_target_triple: Option<&'static str>,
        cargo_opt_level: Option<&'static str>,
    ) -> Self {
        Self {
            git_sha: env_or_unknown(git_sha),
            git_branch: env_or_unknown(git_branch),
            build_timestamp: env_or_unknown(build_timestamp),
            rustc_semver: env_or_unknown(rustc_semver),
            cargo_target_triple: env_or_unknown(cargo_target_triple),
            cargo_opt_level: env_or_unknown(cargo_opt_level),
        }
    }

    /// Build an all-unknown metadata value.
    #[must_use]
    pub const fn unknown() -> Self {
        Self::from_env(None, None, None, None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::BuildMetadata;

    #[test]
    fn fills_missing_fields_with_unknown() {
        assert_eq!(BuildMetadata::unknown().git_sha, "unknown");
        assert_eq!(BuildMetadata::unknown().git_branch, "unknown");
        assert_eq!(BuildMetadata::unknown().build_timestamp, "unknown");
        assert_eq!(BuildMetadata::unknown().rustc_semver, "unknown");
        assert_eq!(BuildMetadata::unknown().cargo_target_triple, "unknown");
        assert_eq!(BuildMetadata::unknown().cargo_opt_level, "unknown");
    }

    #[test]
    fn uses_supplied_env_values() {
        let metadata = BuildMetadata::from_env(
            Some("abc123"),
            Some("main"),
            Some("2026-05-09T00:00:00Z"),
            Some("rustc 1.93.0"),
            Some("x86_64-unknown-linux-gnu"),
            Some("3"),
        );

        assert_eq!(metadata.git_sha, "abc123");
        assert_eq!(metadata.git_branch, "main");
        assert_eq!(metadata.build_timestamp, "2026-05-09T00:00:00Z");
        assert_eq!(metadata.rustc_semver, "rustc 1.93.0");
        assert_eq!(metadata.cargo_target_triple, "x86_64-unknown-linux-gnu");
        assert_eq!(metadata.cargo_opt_level, "3");
    }

    #[test]
    fn from_env_substitutes_unknown_per_field() {
        // Mixed Some/None inputs should fill exactly the None slots with "unknown".
        let metadata =
            BuildMetadata::from_env(Some("sha"), None, Some("ts"), None, Some("triple"), None);

        assert_eq!(metadata.git_sha, "sha");
        assert_eq!(metadata.git_branch, "unknown");
        assert_eq!(metadata.build_timestamp, "ts");
        assert_eq!(metadata.rustc_semver, "unknown");
        assert_eq!(metadata.cargo_target_triple, "triple");
        assert_eq!(metadata.cargo_opt_level, "unknown");
    }

    #[test]
    fn unknown_equals_all_none_from_env() {
        assert_eq!(
            BuildMetadata::unknown(),
            BuildMetadata::from_env(None, None, None, None, None, None)
        );
    }

    #[test]
    fn metadata_is_copy_and_eq() {
        let a = BuildMetadata::from_env(Some("x"), None, None, None, None, None);
        let b = a;
        assert_eq!(a, b);
        // a is still usable because BuildMetadata is Copy.
        assert_eq!(a.git_sha, "x");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_serializes_all_fields() -> Result<(), Box<dyn std::error::Error>> {
        let metadata = BuildMetadata::from_env(
            Some("deadbeef"),
            Some("feature/x"),
            Some("2026-01-01T00:00:00Z"),
            Some("rustc 1.95"),
            Some("aarch64-apple-darwin"),
            Some("0"),
        );
        let json: serde_json::Value = serde_json::to_value(&metadata)?;
        let obj = json
            .as_object()
            .ok_or_else(|| std::io::Error::other("serialized metadata should be a JSON object"))?;
        assert_eq!(obj.get("git_sha").and_then(|v| v.as_str()), Some("deadbeef"));
        assert_eq!(obj.get("git_branch").and_then(|v| v.as_str()), Some("feature/x"));
        assert_eq!(
            obj.get("build_timestamp").and_then(|v| v.as_str()),
            Some("2026-01-01T00:00:00Z")
        );
        assert_eq!(obj.get("rustc_semver").and_then(|v| v.as_str()), Some("rustc 1.95"));
        assert_eq!(
            obj.get("cargo_target_triple").and_then(|v| v.as_str()),
            Some("aarch64-apple-darwin")
        );
        assert_eq!(obj.get("cargo_opt_level").and_then(|v| v.as_str()), Some("0"));
        Ok(())
    }
}
