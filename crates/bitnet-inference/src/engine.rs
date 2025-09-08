//! # Inference Engine Implementation
//!
//! Core inference engine with CPU and GPU backend support, streaming generation,
//! and comprehensive configuration options.

use anyhow::{Context, Result};
use bitnet_common::{BitNetConfig, BitNetTensor, ConcreteTensor, Device, Tensor};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use candle_core::{DType, IndexOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

use crate::{
    backends::{Backend, CpuBackend, GpuBackend},
    cache::{CacheConfig, KVCache},
    config::{GenerationConfig, InferenceConfig},
    gguf,
    sampling::{SamplingConfig, SamplingStrategy},
    streaming::{GenerationStream, StreamingConfig},
};

/// Summary information about a tensor in the GGUF header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSummary {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<u64>,
    /// Underlying GGML dtype identifier
    pub dtype: u32,
    /// Human-readable dtype name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype_name: Option<String>,
    /// Tensor category (weight, embedding, bias, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Parameter count for this tensor
    pub parameter_count: u64,
}

/// Categorized metadata for better organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorizedMetadata {
    /// Model configuration parameters (vocab_size, context_length, etc.)
    pub model_params: HashMap<String, String>,
    /// Architecture details (attention, layer info, etc.)
    pub architecture: HashMap<String, String>,
    /// Tokenizer configuration
    pub tokenizer: HashMap<String, String>,
    /// Training and generation metadata
    pub training: HashMap<String, String>,
    /// Quantization-specific metadata
    pub quantization: HashMap<String, String>,
    /// Other uncategorized metadata
    pub other: HashMap<String, String>,
}

/// Enhanced tensor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    /// Total number of parameters across all tensors
    pub total_parameters: u64,
    /// Parameter count by tensor category
    pub parameters_by_category: HashMap<String, u64>,
    /// Unique data types present
    pub unique_dtypes: Vec<u32>,
    /// Data type distribution
    pub dtype_distribution: HashMap<String, usize>,
    /// Largest tensor by parameter count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub largest_tensor: Option<String>,
    /// Memory footprint estimate in bytes
    pub estimated_memory_bytes: u64,
}

/// Lightweight model info from GGUF header with enhanced categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub header: gguf::GgufHeader,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    kv_specs: Vec<gguf::GgufKv>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    quantization_hints: Vec<gguf::GgufKv>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tensor_summaries: Vec<TensorSummary>,
    /// Categorized metadata for easy access
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categorized_metadata: Option<CategorizedMetadata>,
    /// Enhanced tensor statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_statistics: Option<TensorStatistics>,
}

impl ModelInfo {
    pub fn version(&self) -> u32 {
        self.header.version
    }
    pub fn n_tensors(&self) -> u64 {
        self.header.n_tensors
    }
    pub fn n_kv(&self) -> u64 {
        self.header.n_kv
    }

    /// Raw key/value metadata from the GGUF header
    pub fn kv_specs(&self) -> &[gguf::GgufKv] {
        &self.kv_specs
    }

    /// Metadata entries that hint at model quantization
    pub fn quantization_hints(&self) -> &[gguf::GgufKv] {
        &self.quantization_hints
    }

    /// Summary information for tensors described in the GGUF header
    pub fn tensor_summaries(&self) -> &[TensorSummary] {
        &self.tensor_summaries
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize ModelInfo to JSON")
    }

    /// Serialize to compact JSON string
    pub fn to_json_compact(&self) -> Result<String> {
        serde_json::to_string(self).context("Failed to serialize ModelInfo to compact JSON")
    }

    /// Get categorized metadata, computing it if not already available
    pub fn get_categorized_metadata(&mut self) -> &CategorizedMetadata {
        if self.categorized_metadata.is_none() {
            self.categorized_metadata = Some(self.compute_categorized_metadata());
        }
        self.categorized_metadata.as_ref().unwrap()
    }

    /// Get tensor statistics, computing them if not already available
    pub fn get_tensor_statistics(&mut self) -> &TensorStatistics {
        if self.tensor_statistics.is_none() {
            self.tensor_statistics = Some(self.compute_tensor_statistics());
        }
        self.tensor_statistics.as_ref().unwrap()
    }

    /// Compute categorized metadata from raw KV pairs
    fn compute_categorized_metadata(&self) -> CategorizedMetadata {
        let mut metadata = CategorizedMetadata {
            model_params: HashMap::new(),
            architecture: HashMap::new(),
            tokenizer: HashMap::new(),
            training: HashMap::new(),
            quantization: HashMap::new(),
            other: HashMap::new(),
        };

        for kv in &self.kv_specs {
            let value_str = format_gguf_value(&kv.value);
            let category = categorize_kv_key(&kv.key);

            match category {
                "model" => {
                    metadata.model_params.insert(kv.key.clone(), value_str);
                }
                "architecture" => {
                    metadata.architecture.insert(kv.key.clone(), value_str);
                }
                "tokenizer" => {
                    metadata.tokenizer.insert(kv.key.clone(), value_str);
                }
                "training" => {
                    metadata.training.insert(kv.key.clone(), value_str);
                }
                "quantization" => {
                    metadata.quantization.insert(kv.key.clone(), value_str);
                }
                _ => {
                    metadata.other.insert(kv.key.clone(), value_str);
                }
            }
        }

        metadata
    }

    /// Compute tensor statistics
    fn compute_tensor_statistics(&self) -> TensorStatistics {
        let total_parameters: u64 =
            self.tensor_summaries.iter().map(|t| t.shape.iter().product::<u64>()).sum();

        let mut parameters_by_category = HashMap::new();
        let mut dtype_distribution = HashMap::new();
        let mut largest_tensor = None;
        let mut max_params = 0;
        let mut estimated_memory_bytes = 0u64;

        for tensor in &self.tensor_summaries {
            let params = tensor.shape.iter().product::<u64>();
            let category = tensor.category.as_deref().unwrap_or("other");
            *parameters_by_category.entry(category.to_string()).or_insert(0) += params;

            let dtype_name = format_dtype(tensor.dtype);
            *dtype_distribution.entry(dtype_name).or_insert(0) += 1;

            if params > max_params {
                max_params = params;
                largest_tensor = Some(tensor.name.clone());
            }

            // Estimate memory usage based on dtype
            let bytes_per_param = estimate_bytes_per_param(tensor.dtype);
            estimated_memory_bytes += params * bytes_per_param as u64;
        }

        let unique_dtypes: Vec<u32> = self
            .tensor_summaries
            .iter()
            .map(|t| t.dtype)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        TensorStatistics {
            total_parameters,
            parameters_by_category,
            unique_dtypes,
            dtype_distribution,
            largest_tensor,
            estimated_memory_bytes,
        }
    }
}

/// Synchronous inspection used before full model initialization.
pub fn inspect_model(path: &Path) -> gguf::Result<ModelInfo> {
    const MAX_KEY_LEN: u64 = 1024 * 1024;
    const MAX_STR_LEN: u64 = 10 * 1024 * 1024;
    const ARRAY_SAMPLE_LIMIT: usize = 256;

    fn read_u32_le<R: Read>(r: &mut R) -> gguf::Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64_le<R: Read>(r: &mut R) -> gguf::Result<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_string<R: Read>(r: &mut R) -> gguf::Result<String> {
        let len = read_u64_le(r)?;
        if len > MAX_STR_LEN {
            return Err(gguf::GgufError::StringTooLarge(len));
        }
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| gguf::GgufError::Malformed)
    }

    fn scalar_size_bytes(ty: u32) -> Option<usize> {
        match ty {
            0 | 1 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            7 => Some(1),
            _ => None,
        }
    }

    fn value_from_type_scalar<R: Read>(r: &mut R, ty: u32) -> gguf::Result<gguf::GgufValue> {
        Ok(match ty {
            0 => {
                let mut b = [0];
                r.read_exact(&mut b)?;
                gguf::GgufValue::U8(b[0])
            }
            1 => {
                let mut b = [0];
                r.read_exact(&mut b)?;
                gguf::GgufValue::I8(b[0] as i8)
            }
            2 => {
                let mut b = [0; 2];
                r.read_exact(&mut b)?;
                gguf::GgufValue::U16(u16::from_le_bytes(b))
            }
            3 => {
                let mut b = [0; 2];
                r.read_exact(&mut b)?;
                gguf::GgufValue::I16(i16::from_le_bytes(b))
            }
            4 => {
                let mut b = [0; 4];
                r.read_exact(&mut b)?;
                gguf::GgufValue::U32(u32::from_le_bytes(b))
            }
            5 => {
                let mut b = [0; 4];
                r.read_exact(&mut b)?;
                gguf::GgufValue::I32(i32::from_le_bytes(b))
            }
            6 => {
                let mut b = [0; 4];
                r.read_exact(&mut b)?;
                gguf::GgufValue::F32(f32::from_le_bytes(b))
            }
            7 => {
                let mut b = [0];
                r.read_exact(&mut b)?;
                gguf::GgufValue::Bool(b[0] != 0)
            }
            10 => {
                let mut b = [0; 8];
                r.read_exact(&mut b)?;
                gguf::GgufValue::U64(u64::from_le_bytes(b))
            }
            11 => {
                let mut b = [0; 8];
                r.read_exact(&mut b)?;
                gguf::GgufValue::I64(i64::from_le_bytes(b))
            }
            12 => {
                let mut b = [0; 8];
                r.read_exact(&mut b)?;
                gguf::GgufValue::F64(f64::from_le_bytes(b))
            }
            _ => return Err(gguf::GgufError::InvalidKvType(ty)),
        })
    }

    fn read_array_value<R: Read + Seek>(r: &mut R) -> gguf::Result<Vec<gguf::GgufValue>> {
        let elem_ty = read_u32_le(r)?;
        let len = read_u64_le(r)?;
        let keep = len.min(ARRAY_SAMPLE_LIMIT as u64) as usize;
        let mut out = Vec::with_capacity(keep);

        if elem_ty == 8 {
            for i in 0..len {
                let slen = read_u64_le(r)?;
                if slen > MAX_STR_LEN {
                    return Err(gguf::GgufError::StringTooLarge(slen));
                }
                if (i as usize) < keep {
                    let mut sbuf = vec![0u8; slen as usize];
                    r.read_exact(&mut sbuf)?;
                    let s = String::from_utf8(sbuf).map_err(|_| gguf::GgufError::Malformed)?;
                    out.push(gguf::GgufValue::String(s));
                } else {
                    r.seek(SeekFrom::Current(slen as i64))?;
                }
            }
            return Ok(out);
        }

        if let Some(sz) = scalar_size_bytes(elem_ty) {
            for i in 0..len {
                if (i as usize) < keep {
                    out.push(value_from_type_scalar(r, elem_ty)?);
                } else {
                    let rem = len - i;
                    let skip = (rem as u128) * (sz as u128);
                    if skip > i64::MAX as u128 {
                        let chunk = 1_000_000_000; // 1GB chunks
                        let mut remaining = skip;
                        while remaining > 0 {
                            let to_skip = remaining.min(chunk as u128) as i64;
                            r.seek(SeekFrom::Current(to_skip))?;
                            remaining -= to_skip as u128;
                        }
                    } else {
                        r.seek(SeekFrom::Current(skip as i64))?;
                    }
                    break;
                }
            }
            return Ok(out);
        }

        Err(gguf::GgufError::InvalidKvType(elem_ty))
    }

    let f = std::fs::File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header_buf = [0u8; gguf::GGUF_HEADER_LEN];
    let n = r.read(&mut header_buf)?;
    if n < gguf::GGUF_HEADER_LEN {
        return Err(gguf::GgufError::ShortHeader(n));
    }
    let header = gguf::parse_header(&header_buf)?;

    // parse kv pairs
    let mut kv_specs = Vec::new();
    for _ in 0..header.n_kv {
        let key_len = read_u64_le(&mut r)?;
        if key_len > MAX_KEY_LEN {
            return Err(gguf::GgufError::StringTooLarge(key_len));
        }
        let mut key_buf = vec![0u8; key_len as usize];
        r.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf).map_err(|_| gguf::GgufError::Malformed)?;

        let value_type = read_u32_le(&mut r)?;
        let value = match value_type {
            8 => gguf::GgufValue::String(read_string(&mut r)?),
            9 => gguf::GgufValue::Array(read_array_value(&mut r)?),
            ty => value_from_type_scalar(&mut r, ty)?,
        };

        kv_specs.push(gguf::GgufKv { key, value });
    }

    // collect quantization hints with enhanced detection
    let quantization_hints = kv_specs
        .iter()
        .filter(|kv| {
            let k = kv.key.to_lowercase();
            // Enhanced quantization detection patterns
            k.contains("quant")
                || k.contains("file_type")
                || k.contains("bitnet")
                || k.contains("iq2_s")
                || k.contains("i2_s")
                || k.contains("tl1")
                || k.contains("tl2")
                || k.contains("q4_0")
                || k.contains("q4_1")
                || k.contains("q5_0")
                || k.contains("q5_1")
                || k.contains("q8_0")
                || k.contains("q8_1")
                || k.contains("q2_k")
                || k.contains("q3_k")
                || k.contains("q4_k")
                || k.contains("q5_k")
                || k.contains("q6_k")
                || k.contains("q8_k")
                || k.contains("precision")
                || k.contains("dtype")
                || k.contains("data_type")
                || k.ends_with("_bits")
        })
        .cloned()
        .collect();

    // tensor summaries
    let mut tensor_summaries = Vec::new();
    for _ in 0..header.n_tensors {
        let name = read_string(&mut r)?;
        let n_dims = read_u32_le(&mut r)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64_le(&mut r)?);
        }
        let dtype = read_u32_le(&mut r)?;
        // skip offset
        let _ = read_u64_le(&mut r)?;
        let parameter_count = shape.iter().product::<u64>();
        let dtype_name = Some(format_dtype(dtype));
        let category = Some(categorize_tensor_name(&name));
        tensor_summaries.push(TensorSummary {
            name,
            shape,
            dtype,
            dtype_name,
            category,
            parameter_count,
        });
    }

    Ok(ModelInfo {
        header,
        kv_specs,
        quantization_hints,
        tensor_summaries,
        categorized_metadata: None,
        tensor_statistics: None,
    })
}

/// Result type for inference operations
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub generated_text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
    pub tokens_per_second: f64,
}

/// Main inference engine for BitNet models
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    backend: Box<dyn Backend>,
    cache: Arc<RwLock<KVCache>>,
    config: InferenceConfig,
}

impl InferenceEngine {
    /// Get reference to the tokenizer
    pub fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.tokenizer.clone()
    }

    /// Create a new inference engine
    #[instrument(skip(model, tokenizer))]
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
    ) -> Result<Self> {
        info!("Creating inference engine with device: {:?}", device);

        let config = InferenceConfig::default();
        let cache_config = CacheConfig::default();
        let cache = Arc::new(RwLock::new(KVCache::new(cache_config)?));

        let backend: Box<dyn Backend> = match device {
            Device::Cpu => {
                debug!("Using CPU backend");
                Box::new(CpuBackend::new(model.clone())?)
            }
            Device::Cuda(_) => {
                debug!("Using GPU backend");
                Box::new(GpuBackend::new(model.clone(), device)?)
            }
            Device::Metal => {
                debug!("Using GPU backend (Metal)");
                Box::new(GpuBackend::new(model.clone(), device)?)
            }
        };

        Ok(Self { model, tokenizer, backend, cache, config })
    }

    /// Create inference engine with custom configuration
    pub fn with_config(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
        config: InferenceConfig,
    ) -> Result<Self> {
        let mut engine = Self::new(model, tokenizer, device)?;
        engine.config = config;
        Ok(engine)
    }

    /// Evaluate token IDs and return logits for deterministic comparison
    /// This is used for cross-validation with C++ implementation
    pub async fn eval_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        // Start timing
        let start = std::time::Instant::now();

        // Convert token IDs to ConcreteTensor
        let device = candle_core::Device::Cpu;
        let input_tensor = candle_core::Tensor::from_slice(ids, &[1, ids.len()], &device)?;
        let input = ConcreteTensor::BitNet(BitNetTensor::new(input_tensor));

        // Get cache for forward pass
        let mut cache = self.cache.write().await;

        // Run forward pass through model to get logits
        let logits_tensor = self.backend.forward(&input, &mut cache).await?;

        // Extract logits as f32 vector
        let flat_logits = match logits_tensor {
            ConcreteTensor::BitNet(ref tensor) => tensor.to_vec()?,
            ConcreteTensor::Mock(_) => vec![0.0; 100], // Mock implementation for testing
        };

        debug!("eval_ids: processed {} tokens in {:?}", ids.len(), start.elapsed());

        Ok(flat_logits)
    }

    /// Generate text from a prompt
    #[instrument(skip(self))]
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let config = GenerationConfig::default();
        self.generate_with_config(prompt, &config).await
    }

    /// Generate text with custom configuration
    #[instrument(skip(self, config))]
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        let start_time = std::time::Instant::now();

        debug!("Generating text for prompt: {:?}", &prompt[..50.min(prompt.len())]);

        // Tokenize input
        let input_tokens =
            self.tokenizer.encode(prompt, true, true).context("Failed to tokenize input prompt")?;

        debug!("Input tokens: {} tokens", input_tokens.len());

        // Generate tokens
        let generated_tokens = self
            .generate_tokens(&input_tokens, config)
            .await
            .context("Failed to generate tokens")?;

        // Decode output
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens)
            .context("Failed to decode generated tokens")?;

        let duration = start_time.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / duration.as_secs_f64();

        info!(
            "Generated {} tokens in {:?} ({:.2} tokens/sec)",
            generated_tokens.len(),
            duration,
            tokens_per_second
        );

        Ok(generated_text)
    }

    /// Generate streaming tokens
    pub fn generate_stream(&self, prompt: &str) -> Result<GenerationStream> {
        let config = GenerationConfig::default();
        self.generate_stream_with_config(prompt, &config)
    }

    /// Generate streaming tokens with configuration
    pub fn generate_stream_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationStream> {
        let streaming_config = StreamingConfig::default();

        GenerationStream::new(
            self.model.clone(),
            self.tokenizer.clone(),
            self.backend.clone_backend(),
            self.cache.clone(),
            prompt.to_string(),
            config.clone(),
            streaming_config,
        )
    }

    /// Prefill the model's cache with the provided prompt tokens.
    ///
    /// This runs a forward pass over the entire prompt to populate any
    /// internal state required for subsequent token generation. The logits
    /// from this pass are discarded since prefill is only used for warming
    /// the model and measuring latency.
    ///
    /// # Arguments
    /// * `tokens` - The prompt tokens to prefill with. Can be empty.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Token sequence is too long for the model's context
    /// - Forward pass fails due to model or backend issues
    /// - Cache operations fail
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        debug!("Starting prefill with {} tokens", tokens.len());

        // Handle empty token sequence gracefully
        if tokens.is_empty() {
            debug!("Prefill with empty tokens - skipping forward pass");
            return Ok(());
        }

        // Check if sequence length exceeds model limits
        if tokens.len() > self.config.max_context_length {
            return Err(anyhow::anyhow!(
                "Token sequence length {} exceeds maximum context length {}",
                tokens.len(),
                self.config.max_context_length
            ));
        }

        // Validate tokens are within vocabulary range
        let vocab_size = self.tokenizer.vocab_size() as u32;
        for (i, &token) in tokens.iter().enumerate() {
            if token >= vocab_size {
                return Err(anyhow::anyhow!(
                    "Invalid token {} at position {}: exceeds vocabulary size {}",
                    token,
                    i,
                    vocab_size
                ));
            }
        }

        // Perform a forward pass to populate the cache. Ignore the returned
        // logits since we're only interested in the side effects and timing.
        let start_time = std::time::Instant::now();
        let result = self.forward_pass(tokens).await;
        let prefill_time = start_time.elapsed();

        match result {
            Ok(_) => {
                debug!(
                    "Prefill completed successfully in {:?} ({:.2} tokens/sec)",
                    prefill_time,
                    tokens.len() as f64 / prefill_time.as_secs_f64()
                );
                Ok(())
            }
            Err(e) => {
                warn!("Prefill failed after {:?}: {}", prefill_time, e);
                Err(e.context("Prefill forward pass failed"))
            }
        }
    }

    /// Generate tokens using the configured backend
    pub async fn generate_tokens(
        &self,
        input_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut current_tokens = input_tokens.to_vec();

        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        };

        let mut sampling_strategy = SamplingStrategy::new(sampling_config);

        for step in 0..config.max_new_tokens {
            // Forward pass through model
            let logits = self.forward_pass(&current_tokens).await?;

            // Sample next token first
            let next_token = sampling_strategy.sample(&logits, &current_tokens)?;

            // Capture logits if requested (after sampling to know chosen_id)
            if let Some(cb) = &config.logits_cb
                && (step as usize) < config.logits_tap_steps
            {
                let k = config.logits_topk.min(logits.len());

                // Use partial selection for efficiency on large vocabs
                let mut indices: Vec<usize> = (0..logits.len()).collect();
                if k < logits.len() {
                    indices.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
                        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indices.truncate(k);
                }

                // Sort the top-k for consistent ordering
                indices.sort_by(|&a, &b| {
                    logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
                });

                let topk: Vec<(u32, f32)> =
                    indices.into_iter().map(|idx| (idx as u32, logits[idx])).collect();

                // Pass topk and the chosen token
                (cb)(step as usize, topk, next_token);
            }

            // Check for stop conditions
            if self.should_stop(next_token, &generated_tokens, config) {
                break;
            }

            generated_tokens.push(next_token);
            current_tokens.push(next_token);

            // Limit context length
            if current_tokens.len() > self.config.max_context_length {
                let keep_length = self.config.max_context_length / 2;
                current_tokens = current_tokens[current_tokens.len() - keep_length..].to_vec();
            }
        }

        Ok(generated_tokens)
    }

    /// Perform forward pass through the model
    async fn forward_pass(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Convert tokens to tensor
        let input_tensor = self.tokens_to_tensor(tokens)?;

        // Get cache for this sequence
        let mut cache = self.cache.write().await;

        // Forward pass through backend
        let output_tensor = self.backend.forward(&input_tensor, &mut cache).await?;

        // Extract logits from output tensor
        self.tensor_to_logits(&output_tensor)
    }

    /// Convert tokens to input tensor
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        // Use the model's embed method to convert tokens to embeddings
        Ok(self.model.embed(tokens)?)
    }

    /// Extract logits from output tensor
    fn tensor_to_logits(&self, tensor: &ConcreteTensor) -> Result<Vec<f32>> {
        use bitnet_common::BitNetError;

        // Use the model's logits method to get vocabulary predictions
        let logits_tensor = self.model.logits(tensor)?; // [B,T,V]

        // Extract shape
        let shape = logits_tensor.shape();
        if shape.len() != 3 {
            return Err(BitNetError::Validation("Expected 3D logits tensor [B,T,V]".into()).into());
        }
        let (batch, seq_len, _vocab) = (shape[0], shape[1], shape[2]);

        if batch != 1 {
            return Err(BitNetError::Validation("Only batch=1 supported".into()).into());
        }

        // Get the underlying Candle tensor and extract last timestep
        match &logits_tensor {
            ConcreteTensor::BitNet(t) => {
                let candle = t.to_candle()?;
                // Get last timestep: narrow dim=1 at (seq_len-1), then squeeze
                let last = candle
                    .narrow(1, seq_len - 1, 1)?  // [B, 1, V]
                    .squeeze(1)?                  // [B, V]
                    .i(0)?; // [V]
                let last =
                    if last.dtype() != DType::F32 { last.to_dtype(DType::F32)? } else { last };
                Ok(last.to_vec1::<f32>()?)
            }
            ConcreteTensor::Mock(_) => {
                // Fallback for tests
                let vocab_size = self.tokenizer.vocab_size();
                Ok(vec![0.1; vocab_size])
            }
        }
    }

    /// Check if generation should stop
    fn should_stop(&self, token: u32, generated_tokens: &[u32], config: &GenerationConfig) -> bool {
        // Check for EOS token from config, fallback to tokenizer default
        let eos_token = config.eos_token_id.or_else(|| self.tokenizer.eos_token_id());
        if let Some(eos) = eos_token
            && token == eos
        {
            return true;
        }

        // Check for stop sequences
        if !config.stop_sequences.is_empty() {
            let current_text = self.tokenizer.decode(generated_tokens).unwrap_or_default();
            for stop_seq in &config.stop_sequences {
                if current_text.ends_with(stop_seq) {
                    return true;
                }
            }
        }

        false
    }

    /// Get model configuration
    pub fn model_config(&self) -> &BitNetConfig {
        self.model.config()
    }

    /// Get inference statistics
    pub async fn get_stats(&self) -> InferenceStats {
        let cache = self.cache.read().await;
        InferenceStats {
            cache_size: cache.size(),
            cache_usage: cache.usage_percent(),
            backend_type: self.backend.backend_type(),
        }
    }

    /// Clear the KV cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Statistics about the inference engine
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub cache_size: usize,
    pub cache_usage: f64,
    pub backend_type: String,
}

/// Enhanced categorization function for KV metadata keys
pub fn categorize_kv_key(key: &str) -> &'static str {
    let key_lower = key.to_lowercase();

    // Model parameters
    if key_lower.contains("vocab_size")
        || key_lower.contains("context_length")
        || key_lower.contains("embedding_length")
        || key_lower.contains("block_count")
        || key_lower.contains("feed_forward_length")
        || key_lower.contains("attention.head_count")
        || key_lower.contains("n_vocab")
        || key_lower.contains("n_ctx")
        || key_lower.contains("n_embd")
        || key_lower.contains("n_layer")
        || key_lower.contains("n_ff")
        || key_lower.contains("n_head")
    {
        "model"
    }
    // Architecture details
    else if key_lower.contains("architecture")
        || key_lower.contains("attention")
        || key_lower.contains("rope")
        || key_lower.contains("layer")
        || key_lower.contains("norm")
        || key_lower.contains("activation")
        || key_lower.contains("gelu")
        || key_lower.contains("silu")
    {
        "architecture"
    }
    // Tokenizer configuration
    else if key_lower.contains("tokenizer")
        || key_lower.contains("bos_token")
        || key_lower.contains("eos_token")
        || key_lower.contains("pad_token")
        || key_lower.contains("unk_token")
        || key_lower.contains("vocab")
        || key_lower.contains("token")
    {
        "tokenizer"
    }
    // Training and generation metadata
    else if key_lower.contains("training")
        || key_lower.contains("finetuning")
        || key_lower.contains("dataset")
        || key_lower.contains("epoch")
        || key_lower.contains("learning_rate")
        || key_lower.contains("batch_size")
        || key_lower.contains("temperature")
        || key_lower.contains("top_p")
        || key_lower.contains("top_k")
    {
        "training"
    }
    // Quantization metadata
    else if key_lower.contains("quant")
        || key_lower.contains("file_type")
        || key_lower.contains("bitnet")
        || key_lower.contains("iq2_s")
        || key_lower.contains("i2_s")
        || key_lower.contains("tl1")
        || key_lower.contains("tl2")
    {
        "quantization"
    }
    // Everything else
    else {
        "other"
    }
}

/// Categorize tensor names for better organization
pub fn categorize_tensor_name(name: &str) -> String {
    let name_lower = name.to_lowercase();

    // Check specific component types first (highest priority)
    if name_lower.contains("bias") || name_lower.contains(".b") || name_lower.ends_with("b") {
        "bias".to_string()
    } else if name_lower.contains("embed") || name_lower.contains("token") {
        "embedding".to_string()
    } else if name_lower.contains("norm") || name_lower.contains("ln") {
        "normalization".to_string()
    } else if name_lower.contains("head") || name_lower.contains("lm_head") {
        "output_head".to_string()
    }
    // Check for generic weight patterns (medium priority)
    else if name_lower.contains("weight") {
        "weight".to_string()
    }
    // Check for layer/module location indicators (lowest priority)
    else if name_lower.contains("attention") || name_lower.contains("attn") {
        "attention".to_string()
    } else if name_lower.contains("mlp")
        || name_lower.contains("feed_forward")
        || name_lower.contains("ffn")
    {
        "feed_forward".to_string()
    } else {
        "other".to_string()
    }
}

/// Enhanced dtype formatting with comprehensive coverage
pub fn format_dtype(dtype: u32) -> String {
    match dtype {
        0 => "F32".to_string(),
        1 => "F16".to_string(),
        2 => "Q4_0".to_string(),
        3 => "Q4_1".to_string(),
        4 => "Q5_0".to_string(),
        5 => "Q5_1".to_string(),
        6 => "Q8_0".to_string(),
        7 => "Q8_1".to_string(),
        8 => "Q2_K".to_string(),
        9 => "Q3_K".to_string(),
        10 => "Q4_K".to_string(),
        11 => "Q5_K".to_string(),
        12 => "Q6_K".to_string(),
        13 => "Q8_K".to_string(),
        14 => "IQ2_XXS".to_string(),
        15 => "IQ2_XS".to_string(),
        16 => "IQ3_XXS".to_string(),
        17 => "I2_S".to_string(),  // BitNet native format
        18 => "IQ2_S".to_string(), // BitNet extended format
        19 => "TL1".to_string(),   // Table lookup 1
        20 => "TL2".to_string(),   // Table lookup 2
        21 => "IQ1_S".to_string(),
        22 => "IQ4_NL".to_string(),
        23 => "IQ3_S".to_string(),
        24 => "IQ2_S_NEW".to_string(),
        25 => "IQ4_XS".to_string(),
        _ => format!("Unknown({})", dtype),
    }
}

/// Estimate bytes per parameter for memory calculation
pub fn estimate_bytes_per_param(dtype: u32) -> usize {
    match dtype {
        0 => 4,     // F32
        1 => 2,     // F16
        2 | 3 => 3, // Q4_0, Q4_1 (roughly 4 bits per param + overhead)
        4 | 5 => 3, // Q5_0, Q5_1
        6 | 7 => 1, // Q8_0, Q8_1
        8 => 3,     // Q2_K (roughly 2.25 bits)
        9 => 3,     // Q3_K
        10 => 3,    // Q4_K
        11 => 4,    // Q5_K
        12 => 4,    // Q6_K
        13 => 1,    // Q8_K
        17 => 1,    // I2_S (BitNet 2-bit)
        18 => 3,    // IQ2_S
        19 => 1,    // TL1 (table lookup)
        20 => 1,    // TL2
        _ => 2,     // Unknown, assume 2 bytes
    }
}

/// Format GGUF values for display
pub fn format_gguf_value(value: &gguf::GgufValue) -> String {
    match value {
        gguf::GgufValue::U8(v) => v.to_string(),
        gguf::GgufValue::I8(v) => v.to_string(),
        gguf::GgufValue::U16(v) => v.to_string(),
        gguf::GgufValue::I16(v) => v.to_string(),
        gguf::GgufValue::U32(v) => v.to_string(),
        gguf::GgufValue::I32(v) => v.to_string(),
        gguf::GgufValue::F32(v) => {
            if v.fract() == 0.0 {
                format!("{:.0}", v)
            } else {
                format!("{:.6}", v)
            }
        }
        gguf::GgufValue::Bool(v) => v.to_string(),
        gguf::GgufValue::String(v) => v.clone(),
        gguf::GgufValue::Array(arr) => {
            if arr.len() <= 3 {
                format!("[{}]", arr.iter().map(format_gguf_value).collect::<Vec<_>>().join(", "))
            } else {
                format!(
                    "[{}, ... +{} more]",
                    arr.iter().take(2).map(format_gguf_value).collect::<Vec<_>>().join(", "),
                    arr.len() - 2
                )
            }
        }
        gguf::GgufValue::U64(v) => v.to_string(),
        gguf::GgufValue::I64(v) => v.to_string(),
        gguf::GgufValue::F64(v) => {
            if v.fract() == 0.0 {
                format!("{:.0}", v)
            } else {
                format!("{:.6}", v)
            }
        }
    }
}

// MockTensor is now defined in bitnet_common

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockModel {
        config: BitNetConfig,
    }

    impl MockModel {
        fn new() -> Self {
            Self { config: BitNetConfig::default() }
        }
    }

    impl Model for MockModel {
        fn config(&self) -> &BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn std::any::Any,
        ) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }

        fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 10, 768]))
        }

        fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 10, 50257]))
        }
    }

    struct MockTokenizer;

    impl Tokenizer for MockTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_bos: bool,
            _add_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }

        fn decode(&self, _tokens: &[u32]) -> bitnet_common::Result<String> {
            Ok("mock generated text".to_string())
        }

        fn vocab_size(&self) -> usize {
            50257
        }

        fn token_to_piece(&self, _token: u32) -> Option<String> {
            Some("piece".to_string())
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(50256)
        }

        fn pad_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[tokio::test]
    async fn test_inference_engine_creation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_prefill_functionality() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let tokens = vec![1, 2, 3, 4, 5];

        // Test that prefill executes without error
        let result = engine.prefill(&tokens).await;
        assert!(result.is_ok(), "Prefill should execute successfully");
    }

    #[tokio::test]
    async fn test_prefill_with_empty_tokens() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let empty_tokens = vec![];

        // Test that prefill handles empty input gracefully
        let result = engine.prefill(&empty_tokens).await;
        assert!(result.is_ok(), "Prefill should handle empty input gracefully");
    }

    #[tokio::test]
    async fn test_prefill_with_large_sequence() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let large_tokens: Vec<u32> = (0..1000).collect();

        // Test that prefill handles large sequences
        let result = engine.prefill(&large_tokens).await;
        assert!(result.is_ok(), "Prefill should handle large sequences");
    }

    #[tokio::test]
    async fn test_prefill_multiple_calls() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![4, 5, 6];

        // Test multiple prefill calls
        let result1 = engine.prefill(&tokens1).await;
        let result2 = engine.prefill(&tokens2).await;

        assert!(result1.is_ok(), "First prefill should succeed");
        assert!(result2.is_ok(), "Second prefill should succeed");
    }

    #[tokio::test]
    async fn test_prefill_timing_measurable() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let tokens = vec![1, 2, 3, 4, 5];

        // Measure prefill timing
        let start = std::time::Instant::now();
        let result = engine.prefill(&tokens).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Prefill should succeed");
        // Since MockModel doesn't have timing delay, we just ensure it's measurable
        assert!(elapsed.as_nanos() > 0, "Prefill should have measurable timing");
    }

    #[tokio::test]
    async fn test_prefill_invalid_tokens() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let vocab_size = tokenizer.vocab_size() as u32;
        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let invalid_tokens = vec![1, 2, vocab_size + 10]; // Include token beyond vocab

        // Should fail with invalid token error
        let result = engine.prefill(&invalid_tokens).await;
        assert!(result.is_err(), "Prefill should fail with invalid tokens");

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid token"), "Error should mention invalid token");
        assert!(
            error_msg.contains("exceeds vocabulary size"),
            "Error should mention vocabulary limit"
        );
    }

    #[tokio::test]
    async fn test_prefill_context_length_exceeded() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // Create tokens that exceed the default context length
        let context_limit = engine.config.max_context_length;
        let too_many_tokens: Vec<u32> = (0..context_limit + 100).map(|i| i as u32 % 1000).collect();

        // Should fail with context length error
        let result = engine.prefill(&too_many_tokens).await;
        assert!(result.is_err(), "Prefill should fail when context length exceeded");

        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("exceeds maximum context length"),
            "Error should mention context length limit"
        );
    }

    #[tokio::test]
    async fn test_prefill_edge_case_single_token() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let single_token = vec![1];

        // Should handle single token gracefully
        let result = engine.prefill(&single_token).await;
        assert!(result.is_ok(), "Prefill should handle single token");
    }

    // Test requires full engine implementation
    #[cfg_attr(not(feature = "full-engine"), ignore)]
    #[tokio::test]
    async fn test_text_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Hello, world!").await;

        assert!(result.is_ok());
        let generated_text = result.unwrap();
        assert!(!generated_text.is_empty());
    }
}
