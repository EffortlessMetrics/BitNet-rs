use bitnet_build_info_core::BuildMetadata;

const METADATA: BuildMetadata = BuildMetadata::from_env(
    option_env!("VERGEN_GIT_SHA"),
    None,
    option_env!("VERGEN_BUILD_TIMESTAMP"),
    option_env!("VERGEN_RUSTC_SEMVER"),
    option_env!("VERGEN_CARGO_TARGET_TRIPLE"),
    None,
);

/// Git commit hash at build time
pub const GIT_HASH: &str = METADATA.git_sha;

/// Build timestamp
pub const BUILD_TIMESTAMP: &str = METADATA.build_timestamp;

/// Target triple
pub const TARGET: &str = METADATA.cargo_target_triple;

/// Rust version used for build
pub const RUSTC_VERSION: &str = METADATA.rustc_semver;
