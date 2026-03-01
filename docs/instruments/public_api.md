# Public API Snapshot

Lightweight snapshot of top-level `pub` declarations in selected library crates.

Generated at: 2026-03-01T22:51:51.447709+00:00

## bitnet

Public items: 19

- `pub const BUILD_TIMESTAMP: &str = match option_env!("VERGEN_BUILD_TIMESTAMP") {`
- `pub const GIT_HASH: &str = match option_env!("VERGEN_GIT_SHA") {`
- `pub const MSRV: &str = "1.90.0";`
- `pub const RUSTC_VERSION: &str = match option_env!("VERGEN_RUSTC_SEMVER") {`
- `pub const TARGET: &str = match option_env!("VERGEN_CARGO_TARGET_TRIPLE") {`
- `pub const VERSION: &str = env!("CARGO_PKG_VERSION");`
- `pub mod build_info {`
- `pub mod prelude {`
- `pub use bitnet_common as common;`
- `pub use bitnet_inference as inference;`
- `pub use bitnet_kernels as kernels;`
- `pub use bitnet_models as models;`
- `pub use bitnet_quantization as quantization;`
- `pub use bitnet_tokenizers as tokenizers;`
- `pub use crate::common::{`
- `pub use crate::inference::InferenceEngine;`
- `pub use crate::models::{BitNetModel, ModelLoader};`
- `pub use crate::quantization::Quantize;`
- `pub use crate::tokenizers::Tokenizer;`

## bitnet-cli

Public items: 6

- `pub fn build_cli() -> clap::Command {`
- `pub mod commands;`
- `pub mod config;`
- `pub mod exit;`
- `pub mod ln_rules;`
- `pub mod tokenizer_discovery;`

## bitnet-inference

Public items: 60

- `pub mod backends;`
- `pub mod batch;`
- `pub mod cache;`
- `pub mod config;`
- `pub mod config_builder;`
- `pub mod cpu_opt;`
- `pub mod engine;`
- `pub mod ffi_session; // FFI session wrapper for validation-only parity checking`
- `pub mod generation;`
- `pub mod gguf;`
- `pub mod gpu_streaming;`
- `pub mod kernel_recorder;`
- `pub mod kv_cache_optimized;`
- `pub mod layers;`
- `pub mod memory_pool;`
- `pub mod metrics;`
- `pub mod npu;`
- `pub mod parity;`
- `pub mod prefix_cache;`
- `pub mod prelude {`
- `pub mod production_engine; // always available (sync parser)`
- `pub mod profiler;`
- `pub mod prompt_template; // Chat and instruct format templates`
- `pub mod receipts; // AC4: Inference receipt generation`
- `pub mod rt;`
- `pub mod runtime_utils;`
- `pub mod sampling;`
- `pub mod simple_forward;`
- `pub mod streaming;`
- `pub mod tensor_parallel;`
- `pub mod thread_pool;`
- `pub mod token_stream;`
- `pub use anyhow::Result;`
- `pub use backends::{Backend, CpuBackend, GpuBackend};`
- `pub use batch::{BatchConfig, BatchRequest, BatchResult, BatchScheduler, SingleResult};`
- `pub use bitnet_common::CorrectionRecord;`
- `pub use bitnet_engine_core::{BackendInfo, InferenceSession, SessionConfig, SessionMetrics};`
- `pub use bitnet_inference_metrics_core::{ThroughputMetrics, TimingMetrics};`
- `pub use cache::{CacheConfig, KVCache};`
- `pub use config::{GenerationConfig, InferenceConfig};`
- `pub use engine::{InferenceEngine, InferenceResult};`
- `pub use futures_util::{Stream, StreamExt};`
- `pub use generation::{`
- `pub use gguf::{GGUF_HEADER_LEN, GgufError, GgufHeader, GgufKv, GgufValue, read_kv_pairs};`
- `pub use gpu_streaming::{GpuGenerationStream, GpuStreamingConfig, GpuTokenEvent};`
- `pub use kernel_recorder::KernelRecorder;`
- `pub use kv_cache_optimized::{`
- `pub use layers::{BitNetAttention, LookupTable, QuantizedLinear};`
- `pub use metrics::{`
- `pub use npu::{BITNET_ENABLE_NPU, map_device_token, npu_requested};`
- `pub use parity::{`
- `pub use prefix_cache::{`
- `pub use production_engine::{`
- `pub use prompt_template::{ChatRole, ChatTurn, PromptTemplate, TemplateType};`
- `pub use receipts::{`
- `pub use sampling::{SamplingConfig, SamplingStrategy};`
- `pub use streaming::{GenerationStream, StreamingConfig};`
- `pub use super::{`
- `pub use thread_pool::{InferenceThreadPool, ThreadPoolConfig, ThreadPoolMetrics};`
- `pub use token_stream::{StreamConfig, StreamEvent, StreamStats, TokenBuffer, TokenStream};`

## bitnet-server

Public items: 78

- `pub active_requests: usize,`
- `pub async fn new(config: ServerConfig) -> Result<Self> {`
- `pub async fn shutdown(&self) -> Result<()> {`
- `pub async fn start(&self) -> Result<()> {`
- `pub base: InferenceRequest,`
- `pub base: InferenceResponse,`
- `pub batch_engine: Arc<BatchEngine>,`
- `pub batch_engine_stats: batch_engine::BatchEngineStats,`
- `pub batch_id: Option<String>,`
- `pub batch_size: Option<usize>,`
- `pub concurrency_manager: Arc<ConcurrencyManager>,`
- `pub concurrency_stats: concurrency::ConcurrencyStats,`
- `pub config: ServerConfig,`
- `pub details: Option<serde_json::Value>,`
- `pub device: Option<String>,`
- `pub device_preference: Option<String>,`
- `pub device_statuses: Vec<execution_router::DeviceStatus>,`
- `pub device_used: String,`
- `pub error: String,`
- `pub error_code: String,`
- `pub execution_router: Arc<ExecutionRouter>,`
- `pub fn create_app(&self) -> Router {`
- `pub inference_time_ms: u64,`
- `pub max_tokens: Option<usize>,`
- `pub message: String,`
- `pub metrics: Arc<MetricsCollector>,`
- `pub mod batch_engine;`
- `pub mod caching;`
- `pub mod canary;`
- `pub mod concurrency;`
- `pub mod config;`
- `pub mod execution_router;`
- `pub mod gpu_streaming;`
- `pub mod health;`
- `pub mod model_manager;`
- `pub mod model_registry;`
- `pub mod monitoring;`
- `pub mod security;`
- `pub mod sse;`
- `pub mod streaming;`
- `pub mod websocket;`
- `pub model: Option<String>,`
- `pub model_id: Option<String>,`
- `pub model_id: String,`
- `pub model_manager: Arc<ModelManager>,`
- `pub model_path: String,`
- `pub models_loaded: usize,`
- `pub priority: Option<String>,`
- `pub prompt: String,`
- `pub quantization_hint: Option<String>,`
- `pub quantization_type: String,`
- `pub queue_time_ms: u64,`
- `pub repetition_penalty: Option<f32>,`
- `pub request_id: Option<String>,`
- `pub security_validator: Arc<SecurityValidator>,`
- `pub start_time: Instant,`
- `pub status: String,`
- `pub struct BitNetServer {`
- `pub struct EnhancedInferenceRequest {`
- `pub struct EnhancedInferenceResponse {`
- `pub struct ErrorResponse {`
- `pub struct InferenceRequest {`
- `pub struct InferenceResponse {`
- `pub struct ModelLoadRequest {`
- `pub struct ModelLoadResponse {`
- `pub struct ProductionAppState {`
- `pub struct ServerStats {`
- `pub temperature: Option<f32>,`
- `pub text: String,`
- `pub timeout_ms: Option<u64>,`
- `pub tokenizer_path: Option<String>,`
- `pub tokens_generated: u64,`
- `pub tokens_per_second: f64,`
- `pub top_k: Option<usize>,`
- `pub top_p: Option<f32>,`
- `pub total_requests: u64,`
- `pub uptime_seconds: u64,`
- `pub use config::{DeviceConfig, ServerConfig};`

## bitnet-tokenizers

Public items: 57

- `pub add_bos: bool,`
- `pub add_eos: bool,`
- `pub add_space_prefix: bool,`
- `pub bos_token_id: Option<u32>,`
- `pub bpe_merges: Option<Vec<String>>,`
- `pub byte_fallback: bool,`
- `pub enum TokenizerFileKind {`
- `pub eos_token_id: Option<u32>,`
- `pub fn add_bos_hint(&self) -> Option<bool> {`
- `pub fn bos_eos_eot(&self) -> (Option<u32>, Option<u32>, Option<u32>) {`
- `pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Tokenizer>> {`
- `pub fn from_gguf(reader: &bitnet_models::GgufReader) -> anyhow::Result<Self> {`
- `pub fn from_gguf_reader(reader: &bitnet_models::GgufReader) -> Result<Arc<dyn Tokenizer>> {`
- `pub fn from_path(path: &Path) -> Result<(Arc<dyn Tokenizer>, TokenizerFileKind)> {`
- `pub fn from_pretrained(name: &str) -> Result<Arc<dyn Tokenizer>> {`
- `pub fn kind(&self) -> crate::gguf_loader::GgufTokKind {`
- `pub fn new() -> Self {`
- `pub fn try_from_gguf_metadata<F>(_build_from_arrays: F) -> Option<Arc<dyn Tokenizer>>`
- `pub fn with_config(`
- `pub mod auto;`
- `pub mod deterministic;`
- `pub mod discovery;`
- `pub mod download;`
- `pub mod error_handling;`
- `pub mod fallback;`
- `pub mod gguf_loader;`
- `pub mod gguf_tokenizer;`
- `pub mod hf_tokenizer;`
- `pub mod loader;`
- `pub mod sp_tokenizer;`
- `pub mod spm_tokenizer;`
- `pub mod strategy;`
- `pub mod universal;`
- `pub mod utils;`
- `pub mod vocabulary;`
- `pub model_type: String,`
- `pub pad_token_id: Option<u32>,`
- `pub pre_tokenizer: Option<String>,`
- `pub struct BasicTokenizer {`
- `pub struct RustGgufTokenizer {`
- `pub struct TokenizerBuilder;`
- `pub struct TokenizerConfig {`
- `pub trait Tokenizer: Send + Sync {`
- `pub unk_token_id: Option<u32>,`
- `pub use discovery::{TokenizerDiscovery, TokenizerDownloadInfo, TokenizerStrategy};`
- `pub use download::{DownloadProgress, SmartTokenizerDownload};`
- `pub use error_handling::{CacheManager, ModelTypeDetector, TokenizerErrorHandler};`
- `pub use fallback::TokenizerFallbackChain;`
- `pub use gguf_loader::{GgufTokKind, RustTokenizer};`
- `pub use hf_tokenizer::HfTokenizer;`
- `pub use loader::load_tokenizer;`
- `pub use mock::MockTokenizer;`
- `pub use spm_tokenizer::SpmTokenizer;`
- `pub use strategy::{`
- `pub use universal::{TokenizerBackend, UniversalTokenizer};`
- `pub vocab_size: usize,`
- `pub vocabulary: Option<Vec<(String, f32)>>,`
