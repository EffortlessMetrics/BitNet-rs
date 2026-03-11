//! API key authentication middleware.
//!
//! This module re-exports the shared API-key authentication primitives from
//! `bitnet-api-key-auth-core` to keep existing import paths stable.

pub use bitnet_api_key_auth_core::{ApiKey, AuthMode, AuthResult, KeyStore};
