//! Server endpoint registry façade.
//!
//! This module re-exports shared endpoint registry primitives from
//! `bitnet-endpoint-registry-core` to preserve the existing server API.

pub use bitnet_endpoint_registry_core::{Endpoint, EndpointRegistry, HttpMethod};
