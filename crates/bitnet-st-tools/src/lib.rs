//! SafeTensors utilities for BitNet: LN inspection and LN-preserving merge
//!
//! This crate provides two binary tools:
//! - `st-ln-inspect`: Inspect LayerNorm gamma dtype and RMS from SafeTensors
//! - `st-merge-ln-f16`: Merge SafeTensors shards with LN gamma forced to F16

pub mod common;
