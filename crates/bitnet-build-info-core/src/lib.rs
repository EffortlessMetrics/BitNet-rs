//! Reusable build metadata constants sourced from `vergen` environment variables.

/// Git commit hash at build time.
pub const GIT_SHA: &str = match option_env!("VERGEN_GIT_SHA") {
    Some(value) => value,
    None => "unknown",
};

/// Git branch name at build time.
pub const GIT_BRANCH: &str = match option_env!("VERGEN_GIT_BRANCH") {
    Some(value) => value,
    None => "unknown",
};

/// Build timestamp at build time.
pub const BUILD_TIMESTAMP: &str = match option_env!("VERGEN_BUILD_TIMESTAMP") {
    Some(value) => value,
    None => "unknown",
};

/// Rust version used for build.
pub const RUSTC_VERSION: &str = match option_env!("VERGEN_RUSTC_SEMVER") {
    Some(value) => value,
    None => "unknown",
};

/// Cargo target triple used for build.
pub const CARGO_TARGET_TRIPLE: &str = match option_env!("VERGEN_CARGO_TARGET_TRIPLE") {
    Some(value) => value,
    None => "unknown",
};

/// Cargo optimization level used for build.
pub const CARGO_OPT_LEVEL: &str = match option_env!("VERGEN_CARGO_OPT_LEVEL") {
    Some(value) => value,
    None => "unknown",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_non_empty() {
        for value in [
            GIT_SHA,
            GIT_BRANCH,
            BUILD_TIMESTAMP,
            RUSTC_VERSION,
            CARGO_TARGET_TRIPLE,
            CARGO_OPT_LEVEL,
        ] {
            assert!(!value.is_empty());
        }
    }
}
