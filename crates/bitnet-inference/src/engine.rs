//! # Inference Engine Implementation
//!
//! Core inference engine with CPU and GPU backend support, streaming generation,
//! and comprehensive configuration options.

use anyhow::{Context, Result};
use bitnet_common::{BitNetConfig, BitNetTensor, ConcreteTensor, Device, Tensor};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use candle_core::{DType, IndexOp};
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
#[derive(Debug, Clone)]
pub struct TensorSummary {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<u64>,
    /// Underlying GGML dtype identifier
    pub dtype: u32,
}

/// Lightweight model info from GGUF header
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub header: gguf::GgufHeader,
    kv_specs: Vec<gguf::GgufKv>,
    quantization_hints: Vec<gguf::GgufKv>,
    tensor_summaries: Vec<TensorSummary>,
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

    // collect quantization hints
    let quantization_hints = kv_specs
        .iter()
        .filter(|kv| {
            let k = kv.key.to_lowercase();
            k.contains("quant") || k.contains("file_type")
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
        tensor_summaries.push(TensorSummary { name, shape, dtype });
    }

    Ok(ModelInfo { header, kv_specs, quantization_hints, tensor_summaries })
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

    /// Generate tokens using the configured backend
    async fn generate_tokens(
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
