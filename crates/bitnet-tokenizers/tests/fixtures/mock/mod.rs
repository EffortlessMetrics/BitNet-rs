//! Mock Infrastructure for Universal Tokenizer Discovery Testing
//!
//! Provides mock implementations for unit testing without file I/O or network access.

#![cfg(test)]
#![cfg(feature = "cpu")]

pub mod mock_download_manager;
pub mod mock_gguf_reader;
