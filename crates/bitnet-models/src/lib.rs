//! Model definitions and loading for BitNet inference

pub mod architecture;
pub mod bitnet;
pub mod capability_check;
pub mod catalog;
pub mod checkpoint;
pub mod comparison;
pub mod comparison_report;
pub mod config;
pub mod config_detection;
pub mod config_serde;
pub mod conversion;
pub mod conversion_pipeline;
pub mod correction_policy;
pub mod dense_crossval;
pub mod dense_gguf_descriptors;
pub mod dense_gguf_linear_fixture;
pub mod dense_gguf_norm_fixture;
pub mod dense_gguf_q8_dispatch;
pub mod dense_gguf_q8_equivalence;
pub mod dense_gguf_q8_sidecar;
pub mod download_manager;
pub mod fingerprint;
pub mod format_detect;
pub mod format_detector;
pub mod formats;
pub mod gguf_metadata;
pub mod gguf_min;
pub mod gguf_parity;
pub mod gguf_simple;
pub mod gguf_writer;
pub mod health_check;
pub mod hf_loader;
pub mod layer_inspector;
pub mod loader;
pub mod loading_progress;
pub mod memory_estimator;
pub mod metadata_extractor;
pub mod minimal;
pub mod model_catalog;
pub mod model_checkpoint;
pub mod model_compare;
pub mod model_config_builder;
pub mod model_contracts;
pub mod model_diff;
pub mod model_fingerprint;
pub mod model_kernel_compat;
pub mod model_metadata;
pub mod model_registry;
pub mod model_validation;
pub mod model_validator;
pub mod names;
pub mod production_loader;
pub mod pruning_analysis;
pub mod qk256_utils;
pub mod quant;
pub mod registry_persist;
pub mod registry_query;
pub mod safetensors_reader;
pub mod security;
pub mod shard_index;
pub mod transformer;
pub mod validation;
pub mod validation_suite;
pub mod validator;
pub mod weight_format;
pub mod weight_loader;
pub mod weight_loader_pipeline;
pub mod weight_mapper;
pub mod weight_stats;

#[cfg(test)]
mod transformer_tests;

pub use bitnet::*;
#[allow(deprecated)]
pub use gguf_simple::load_gguf;
pub use gguf_simple::load_gguf_full;
pub use gguf_simple::{GGUFLoaderConfig, GgufLoaderMode}; // AC1: Export loader config/mode
pub use loader::*;
pub use production_loader::*;

// Export GGUF reader for tokenizer loading
pub use formats::gguf::GgufReader;

// Export weight mapper utilities for crossval tests
pub use weight_mapper::WeightMapper;
pub use weight_mapper::dry_run_remap_names;

// AC2: Re-export QK256 tolerance constants from bitnet-quantization (Issue #469)
pub use bitnet_quantization::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};
