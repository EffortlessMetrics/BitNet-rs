//! GGUF format implementation

mod loader;
mod reader;
mod types;

#[cfg(test)]
mod tests;

pub use loader::GgufLoader;
pub use reader::GgufReader;
pub use types::GgufTensors;
pub use types::*;

// Re-export lightweight GGUF types from the dedicated micro-crate.
// Note: `GgufValue`, `GgufHeader`, and `TensorInfo` already exist in this
// module (from types.rs) so only non-conflicting names are re-exported.
pub use bitnet_gguf::{
    GGUF_MAGIC, GgufFileInfo, GgufMetadataKv as GgufRawMetadataKv, GgufValue as GgufRawValue,
    GgufValueType, TensorInfo as RawTensorInfo, check_magic, parse_header as parse_gguf_header,
    read_version as read_gguf_version,
};
