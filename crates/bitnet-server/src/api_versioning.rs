//! API versioning for the inference server.
//!
//! Re-exported from `bitnet-api-version-core` so server and other crates can share
//! a single canonical implementation.

pub use bitnet_api_version_core::{
    ApiVersion, NegotiationResult, VersionRange, extract_version_from_path, version_header,
};
