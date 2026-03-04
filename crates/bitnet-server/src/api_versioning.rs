//! API versioning for the inference server.
//!
//! This is a thin re-export shim; the implementation lives in
//! `bitnet-api-versioning-core`.

pub use bitnet_api_versioning_core::{
    ApiVersion, NegotiationResult, VersionRange, extract_version_from_path, version_header,
};
