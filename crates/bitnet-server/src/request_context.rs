//! Request context façade for inference API.
//!
//! This module re-exports shared request context primitives from
//! `bitnet-request-context-core` to preserve the existing server API.

pub use bitnet_request_context_core::{ClientInfo, RequestBatch, RequestContext, RequestId};
