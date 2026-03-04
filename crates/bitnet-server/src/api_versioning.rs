//! API versioning for the inference server.
//!
//! This module re-exports the shared API versioning primitives from
//! `bitnet-api-versioning-core` to keep existing import paths stable.

pub use bitnet_api_versioning_core::{
    ApiVersion, NegotiationResult, VersionRange, extract_version_from_path, version_header,
};
