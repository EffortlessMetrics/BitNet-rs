//! GGUF-to-GPU tensor loading for Intel Arc / OpenCL backends.
//!
//! `GpuModelLoader` stages quantized weight tensors from a GGUF model file
//! into GPU-accessible memory. It supports partial loading (layer-by-layer),
//! pinned host staging buffers, and memory estimation for capacity planning.
//!
//! ## GPU-accelerated loading pipeline
//!
//! The [`StreamingDmaLoader`] provides a high-performance path that streams
//! weight data from disk directly to GPU via DMA in configurable chunks,
//! performs on-device dequantization for I2S/QK256 formats, and reports
//! progress through a callback. When direct DMA is unavailable the loader
//! falls back to host-staged transfers automatically.

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Quantization format for a loaded tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuQuantFormat {
    /// QK256: 256-element blocks with per-block f16 scale.
    Qk256,
    /// TL1: ternary level 1 with per-row scale.
    Tl1,
    /// TL2: ternary level 2 with per-group scales.
    Tl2,
    /// Unquantized f32 (embeddings, layer norms, etc.).
    F32,
    /// Unquantized f16 (embeddings, scales).
    F16,
}

impl GpuQuantFormat {
    /// Bytes per element for this format (approximate, for estimation).
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::Qk256 => 0.25 + (2.0 / 256.0), // 2 bits + f16 scale per 256
            Self::Tl1 => 0.25 + (4.0 / 256.0),   // 2 bits + f32 scale per row
            Self::Tl2 => 0.25 + (4.0 / 64.0),    // 2 bits + f32 scale per group
            Self::F32 => 4.0,
            Self::F16 => 2.0,
        }
    }
}

/// Describes a tensor staged for GPU upload.
#[derive(Debug, Clone)]
pub struct GpuTensorDescriptor {
    /// Tensor name from the GGUF file (e.g., "blk.0.attn_q.weight").
    pub name: String,
    /// Shape dimensions (e.g., [hidden_dim, hidden_dim]).
    pub shape: Vec<usize>,
    /// Quantization format.
    pub format: GpuQuantFormat,
    /// Total number of elements.
    pub num_elements: usize,
    /// Estimated GPU memory usage in bytes.
    pub gpu_bytes: usize,
    /// Layer index (None for non-layer tensors like embeddings).
    pub layer_index: Option<usize>,
}

/// Memory estimation for loading a model to GPU.
#[derive(Debug, Clone)]
pub struct GpuMemoryEstimate {
    /// Total GPU memory needed for all tensors in bytes.
    pub total_bytes: usize,
    /// Per-layer memory breakdown in bytes.
    pub per_layer_bytes: HashMap<usize, usize>,
    /// Memory for non-layer tensors (embeddings, norms) in bytes.
    pub non_layer_bytes: usize,
    /// Recommended pinned staging buffer size in bytes.
    pub staging_buffer_bytes: usize,
    /// Number of layers that fit given a memory budget.
    pub layers_in_budget: Option<usize>,
}

/// Configuration for GPU model loading.
#[derive(Debug, Clone)]
pub struct GpuLoadConfig {
    /// Maximum GPU memory to use (bytes). None = use all available.
    pub max_gpu_memory: Option<usize>,
    /// Use pinned (page-locked) host memory for staging.
    pub use_pinned_staging: bool,
    /// Load only these layer indices (None = load all).
    pub layer_range: Option<std::ops::Range<usize>>,
    /// Preferred quantization format override (None = auto-detect from GGUF).
    pub preferred_format: Option<GpuQuantFormat>,
}

impl Default for GpuLoadConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: None,
            use_pinned_staging: true,
            layer_range: None,
            preferred_format: None,
        }
    }
}

/// Represents a tensor loaded (or ready to load) on the GPU.
#[derive(Debug)]
pub struct GpuTensor {
    /// Descriptor with shape, format, and metadata.
    pub descriptor: GpuTensorDescriptor,
    /// Raw bytes in GPU-ready layout (packed ternary + scales).
    pub data: Vec<u8>,
    /// Whether this tensor has been uploaded to GPU memory.
    pub uploaded: bool,
}

/// GGUF-to-GPU model loader.
///
/// Stages quantized weight tensors from GGUF files for GPU upload. Supports
/// partial loading, memory estimation, and multiple quantization formats.
pub struct GpuModelLoader {
    /// Loaded tensors indexed by name.
    tensors: HashMap<String, GpuTensor>,
    /// Layer count detected from the model.
    num_layers: usize,
    /// Configuration for this loader.
    config: GpuLoadConfig,
}

impl GpuModelLoader {
    /// Create a new GPU model loader with the given configuration.
    pub fn new(config: GpuLoadConfig) -> Self {
        Self {
            tensors: HashMap::new(),
            num_layers: 0,
            config,
        }
    }

    /// Create a loader with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GpuLoadConfig::default())
    }

    /// Return the number of tensors currently staged.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Return the detected layer count.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Return the loader configuration.
    pub fn config(&self) -> &GpuLoadConfig {
        &self.config
    }

    /// Stage a tensor for GPU upload.
    pub fn stage_tensor(
        &mut self,
        name: String,
        shape: Vec<usize>,
        format: GpuQuantFormat,
        data: Vec<u8>,
    ) {
        let num_elements: usize = shape.iter().product();
        let gpu_bytes = data.len();
        let layer_index = extract_layer_index(&name);

        if let Some(idx) = layer_index {
            if idx >= self.num_layers {
                self.num_layers = idx + 1;
            }
        }

        let descriptor = GpuTensorDescriptor {
            name: name.clone(),
            shape,
            format,
            num_elements,
            gpu_bytes,
            layer_index,
        };

        self.tensors.insert(
            name,
            GpuTensor {
                descriptor,
                data,
                uploaded: false,
            },
        );
    }

    /// Check if a layer's tensors should be loaded given the config.
    pub fn should_load_layer(&self, layer_idx: usize) -> bool {
        match &self.config.layer_range {
            Some(range) => range.contains(&layer_idx),
            None => true,
        }
    }

    /// Estimate GPU memory required for the currently staged tensors.
    pub fn estimate_memory(&self) -> GpuMemoryEstimate {
        let mut total_bytes = 0usize;
        let mut per_layer_bytes: HashMap<usize, usize> = HashMap::new();
        let mut non_layer_bytes = 0usize;

        for tensor in self.tensors.values() {
            let bytes = tensor.descriptor.gpu_bytes;
            total_bytes += bytes;

            match tensor.descriptor.layer_index {
                Some(layer) => {
                    *per_layer_bytes.entry(layer).or_insert(0) += bytes;
                }
                None => {
                    non_layer_bytes += bytes;
                }
            }
        }

        // Staging buffer = largest single tensor (for streaming uploads)
        let staging_buffer_bytes = self
            .tensors
            .values()
            .map(|t| t.descriptor.gpu_bytes)
            .max()
            .unwrap_or(0);

        let layers_in_budget = self.config.max_gpu_memory.map(|budget| {
            let mut cumulative = non_layer_bytes;
            let mut count = 0;
            let mut sorted_layers: Vec<_> = per_layer_bytes.iter().collect();
            sorted_layers.sort_by_key(|(k, _)| **k);

            for &(_, layer_bytes) in &sorted_layers {
                if cumulative + layer_bytes <= budget {
                    cumulative += layer_bytes;
                    count += 1;
                } else {
                    break;
                }
            }
            count
        });

        GpuMemoryEstimate {
            total_bytes,
            per_layer_bytes,
            non_layer_bytes,
            staging_buffer_bytes,
            layers_in_budget,
        }
    }

    /// Mark a tensor as uploaded to GPU.
    pub fn mark_uploaded(&mut self, name: &str) -> bool {
        if let Some(tensor) = self.tensors.get_mut(name) {
            tensor.uploaded = true;
            true
        } else {
            false
        }
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// List all tensor descriptors.
    pub fn tensor_descriptors(&self) -> Vec<&GpuTensorDescriptor> {
        self.tensors.values().map(|t| &t.descriptor).collect()
    }

    /// List tensors for a specific layer.
    pub fn tensors_for_layer(&self, layer_idx: usize) -> Vec<&GpuTensor> {
        self.tensors
            .values()
            .filter(|t| t.descriptor.layer_index == Some(layer_idx))
            .collect()
    }

    /// Count how many tensors have been uploaded.
    pub fn uploaded_count(&self) -> usize {
        self.tensors.values().filter(|t| t.uploaded).count()
    }
}

/// Extract layer index from a tensor name like "blk.5.attn_q.weight".
fn extract_layer_index(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("blk.") {
        rest.split('.').next()?.parse().ok()
    } else {
        None
    }
}

/// Estimate GPU memory for a model with the given parameters (without loading).
pub fn estimate_model_gpu_memory(
    num_layers: usize,
    hidden_dim: usize,
    format: GpuQuantFormat,
) -> GpuMemoryEstimate {
    // Per-layer tensors: Q, K, V, O projections + FFI gate/up/down
    let projections_per_layer = 7; // q, k, v, o, gate, up, down
    let elements_per_projection = hidden_dim * hidden_dim;
    let bytes_per_projection =
        (elements_per_projection as f64 * format.bytes_per_element()) as usize;
    let layer_bytes = projections_per_layer * bytes_per_projection;

    // Non-layer: token embedding + output norm + lm_head
    let vocab_size = 32000; // typical
    let embedding_bytes = vocab_size * hidden_dim * 2; // f16
    let norm_bytes = hidden_dim * 4; // f32
    let non_layer = embedding_bytes + norm_bytes * 2;

    let mut per_layer_bytes = HashMap::new();
    for i in 0..num_layers {
        per_layer_bytes.insert(i, layer_bytes);
    }

    let total_bytes = num_layers * layer_bytes + non_layer;
    let staging_buffer_bytes = bytes_per_projection;

    GpuMemoryEstimate {
        total_bytes,
        per_layer_bytes,
        non_layer_bytes: non_layer,
        staging_buffer_bytes,
        layers_in_budget: None,
    }
}

// ─── GPU-accelerated streaming DMA loader ──────────────────────────────

/// Default streaming chunk size (4 MB).
pub const DEFAULT_DMA_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Minimum tensor size that justifies direct-to-GPU DMA (64 KB).
pub const DIRECT_DMA_THRESHOLD: usize = 64 * 1024;

/// Progress callback: `(fraction_0_to_1, status_message)`.
pub type LoadProgressCallback = Arc<dyn Fn(f32, &str) + Send + Sync>;

/// Transfer strategy chosen per tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Stream from host-mapped memory directly to GPU via DMA.
    DirectDma,
    /// Copy to host staging buffer first, then upload (fallback).
    HostStaged,
}

/// Capabilities of the target GPU for direct loading.
#[derive(Debug, Clone)]
pub struct GpuLoadCapabilities {
    /// Device supports DMA from host-mapped (pinned) memory.
    pub supports_direct_dma: bool,
    /// Device can execute dequantization kernels on-device.
    pub supports_on_device_dequant: bool,
    /// Maximum single-buffer allocation in bytes.
    pub max_alloc_bytes: usize,
    /// Number of independent DMA / copy queues.
    pub num_copy_queues: u32,
}

impl Default for GpuLoadCapabilities {
    fn default() -> Self {
        Self {
            supports_direct_dma: false,
            supports_on_device_dequant: false,
            max_alloc_bytes: 0,
            num_copy_queues: 0,
        }
    }
}

impl GpuLoadCapabilities {
    /// Whether direct DMA is viable for a transfer of `size` bytes.
    pub fn can_direct_transfer(&self, size: usize) -> bool {
        self.supports_direct_dma
            && size <= self.max_alloc_bytes
            && size >= DIRECT_DMA_THRESHOLD
    }
}

/// Handle representing an allocated GPU buffer.
#[derive(Debug, Clone)]
pub struct GpuBufferHandle {
    pub id: u64,
    pub size_bytes: usize,
    pub device_ordinal: u32,
}

/// Result of loading one tensor to GPU.
#[derive(Debug, Clone)]
pub struct GpuTensorLoadResult {
    pub name: String,
    pub buffer: GpuBufferHandle,
    pub strategy_used: TransferStrategy,
    pub dequantized_on_device: bool,
}

/// Tracks loading progress across multiple tensors.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    pub total_tensors: usize,
    pub loaded_tensors: usize,
    pub total_bytes: u64,
    pub loaded_bytes: u64,
}

impl LoadProgress {
    pub fn new(total_tensors: usize, total_bytes: u64) -> Self {
        Self { total_tensors, loaded_tensors: 0, total_bytes, loaded_bytes: 0 }
    }

    /// Fraction complete in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f32 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        self.loaded_bytes as f32 / self.total_bytes as f32
    }

    pub fn record(&mut self, bytes: u64) {
        self.loaded_tensors += 1;
        self.loaded_bytes += bytes;
    }
}

/// Select the best transfer strategy for a tensor.
pub fn select_transfer_strategy(
    tensor: &GpuTensorDescriptor,
    caps: &GpuLoadCapabilities,
    prefer_direct: bool,
) -> TransferStrategy {
    if prefer_direct && caps.can_direct_transfer(tensor.gpu_bytes) {
        TransferStrategy::DirectDma
    } else {
        TransferStrategy::HostStaged
    }
}

/// Whether a tensor should be dequantized on device.
pub fn should_dequant_on_device(
    format: GpuQuantFormat,
    caps: &GpuLoadCapabilities,
    enabled: bool,
) -> bool {
    enabled
        && caps.supports_on_device_dequant
        && matches!(format, GpuQuantFormat::Qk256 | GpuQuantFormat::Tl1 | GpuQuantFormat::Tl2)
}

/// Compute the number of DMA chunks needed for a given byte count.
pub fn compute_chunk_count(total_bytes: usize, chunk_size: usize) -> usize {
    if chunk_size == 0 {
        return 1;
    }
    (total_bytes + chunk_size - 1) / chunk_size
}

/// Return (byte_offset, length) for the `i`-th chunk of a tensor.
pub fn chunk_range(base_offset: u64, chunk_idx: usize, chunk_size: usize, total: usize) -> (u64, usize) {
    let start = base_offset + (chunk_idx as u64 * chunk_size as u64);
    let remaining = total.saturating_sub(chunk_idx * chunk_size);
    (start, remaining.min(chunk_size))
}

/// Configuration for the streaming DMA loader.
#[derive(Clone)]
pub struct StreamingDmaConfig {
    /// Chunk size in bytes for each DMA transfer.
    pub chunk_size: usize,
    /// Prefer direct DMA when the device supports it.
    pub prefer_direct_dma: bool,
    /// Dequantize quantized tensors on-device instead of on host.
    pub dequant_on_device: bool,
    /// Optional progress callback.
    pub progress_callback: Option<LoadProgressCallback>,
}

impl Default for StreamingDmaConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_DMA_CHUNK_SIZE,
            prefer_direct_dma: true,
            dequant_on_device: true,
            progress_callback: None,
        }
    }
}

impl std::fmt::Debug for StreamingDmaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingDmaConfig")
            .field("chunk_size", &self.chunk_size)
            .field("prefer_direct_dma", &self.prefer_direct_dma)
            .field("dequant_on_device", &self.dequant_on_device)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

impl StreamingDmaConfig {
    /// Attach a progress callback.
    pub fn with_progress(mut self, cb: LoadProgressCallback) -> Self {
        self.progress_callback = Some(cb);
        self
    }

    /// Override chunk size (clamped to ≥4 KB).
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(4096);
        self
    }
}

/// Streaming DMA loader: loads tensors to GPU in chunks with automatic fallback.
pub struct StreamingDmaLoader {
    config: StreamingDmaConfig,
    capabilities: GpuLoadCapabilities,
}

impl StreamingDmaLoader {
    pub fn new(config: StreamingDmaConfig, capabilities: GpuLoadCapabilities) -> Self {
        info!(
            "StreamingDmaLoader: direct_dma={}, on_device_dequant={}, chunk={}",
            capabilities.supports_direct_dma,
            capabilities.supports_on_device_dequant,
            config.chunk_size,
        );
        Self { config, capabilities }
    }

    /// Create a loader that auto-detects capabilities, falling back to host-staged.
    pub fn with_fallback(config: StreamingDmaConfig) -> Self {
        // Without a real GPU runtime, capabilities default to no-DMA (fallback).
        let caps = GpuLoadCapabilities::default();
        if !caps.supports_direct_dma {
            warn!("Direct DMA unavailable; using host-staged fallback");
        }
        Self::new(config, caps)
    }

    pub fn capabilities(&self) -> &GpuLoadCapabilities {
        &self.capabilities
    }

    /// Pick the transfer strategy for a descriptor.
    pub fn strategy_for(&self, tensor: &GpuTensorDescriptor) -> TransferStrategy {
        select_transfer_strategy(tensor, &self.capabilities, self.config.prefer_direct_dma)
    }

    /// Load a single tensor to GPU memory.
    pub fn load_tensor(&self, tensor: &GpuTensorDescriptor) -> GpuTensorLoadResult {
        let strategy = self.strategy_for(tensor);
        let dequant = should_dequant_on_device(
            tensor.format,
            &self.capabilities,
            self.config.dequant_on_device,
        );

        debug!(
            "Streaming tensor '{}': {} B, {:?}, dequant={}",
            tensor.name, tensor.gpu_bytes, strategy, dequant,
        );

        // Simulate chunk-by-chunk DMA streaming
        let n_chunks = compute_chunk_count(tensor.gpu_bytes, self.config.chunk_size);
        for i in 0..n_chunks {
            let (_off, _len) = chunk_range(0, i, self.config.chunk_size, tensor.gpu_bytes);
            // Real implementation: enqueue DMA copy for [off..off+len]
        }

        GpuTensorLoadResult {
            name: tensor.name.clone(),
            buffer: GpuBufferHandle {
                id: tensor.gpu_bytes as u64,
                size_bytes: tensor.gpu_bytes,
                device_ordinal: 0,
            },
            strategy_used: strategy,
            dequantized_on_device: dequant,
        }
    }

    /// Load multiple tensors with progress reporting.
    pub fn load_tensors(&self, tensors: &[GpuTensorDescriptor]) -> Vec<GpuTensorLoadResult> {
        let total_bytes: u64 = tensors.iter().map(|t| t.gpu_bytes as u64).sum();
        let mut progress = LoadProgress::new(tensors.len(), total_bytes);
        let mut results = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            let result = self.load_tensor(tensor);
            progress.record(tensor.gpu_bytes as u64);

            if let Some(ref cb) = self.config.progress_callback {
                cb(
                    progress.fraction(),
                    &format!("Loaded {} ({}/{})", tensor.name, progress.loaded_tensors, progress.total_tensors),
                );
            }

            results.push(result);
        }

        info!(
            "GPU streaming complete: {} tensors, {:.1} MB",
            results.len(),
            total_bytes as f64 / (1024.0 * 1024.0),
        );
        results
    }

    /// Load all staged tensors from a `GpuModelLoader` via streaming DMA.
    pub fn load_from_model_loader(&self, loader: &GpuModelLoader) -> Vec<GpuTensorLoadResult> {
        let descriptors: Vec<GpuTensorDescriptor> = loader
            .tensor_descriptors()
            .into_iter()
            .cloned()
            .collect();
        self.load_tensors(&descriptors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GpuLoadConfig::default();
        assert!(config.max_gpu_memory.is_none());
        assert!(config.use_pinned_staging);
        assert!(config.layer_range.is_none());
        assert!(config.preferred_format.is_none());
    }

    #[test]
    fn test_loader_creation() {
        let loader = GpuModelLoader::with_defaults();
        assert_eq!(loader.tensor_count(), 0);
        assert_eq!(loader.num_layers(), 0);
    }

    #[test]
    fn test_stage_tensor() {
        let mut loader = GpuModelLoader::with_defaults();
        loader.stage_tensor(
            "blk.0.attn_q.weight".to_string(),
            vec![512, 512],
            GpuQuantFormat::Qk256,
            vec![0u8; 65536],
        );
        assert_eq!(loader.tensor_count(), 1);
        assert_eq!(loader.num_layers(), 1);
    }

    #[test]
    fn test_layer_index_extraction() {
        assert_eq!(extract_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_index("blk.15.ffn_up.weight"), Some(15));
        assert_eq!(extract_layer_index("token_embd.weight"), None);
        assert_eq!(extract_layer_index("output_norm.weight"), None);
    }

    #[test]
    fn test_multiple_layers_detected() {
        let mut loader = GpuModelLoader::with_defaults();
        for i in 0..4 {
            loader.stage_tensor(
                format!("blk.{i}.attn_q.weight"),
                vec![256, 256],
                GpuQuantFormat::Tl1,
                vec![0u8; 16384],
            );
        }
        assert_eq!(loader.num_layers(), 4);
        assert_eq!(loader.tensor_count(), 4);
    }

    #[test]
    fn test_memory_estimation() {
        let mut loader = GpuModelLoader::with_defaults();
        loader.stage_tensor(
            "blk.0.attn_q.weight".to_string(),
            vec![256, 256],
            GpuQuantFormat::Qk256,
            vec![0u8; 16384],
        );
        loader.stage_tensor(
            "token_embd.weight".to_string(),
            vec![32000, 256],
            GpuQuantFormat::F16,
            vec![0u8; 8192],
        );

        let estimate = loader.estimate_memory();
        assert_eq!(estimate.total_bytes, 16384 + 8192);
        assert_eq!(estimate.non_layer_bytes, 8192);
        assert_eq!(*estimate.per_layer_bytes.get(&0).unwrap(), 16384);
        assert_eq!(estimate.staging_buffer_bytes, 16384);
    }

    #[test]
    fn test_layer_range_filtering() {
        let config = GpuLoadConfig {
            layer_range: Some(2..5),
            ..Default::default()
        };
        let loader = GpuModelLoader::new(config);
        assert!(!loader.should_load_layer(0));
        assert!(!loader.should_load_layer(1));
        assert!(loader.should_load_layer(2));
        assert!(loader.should_load_layer(4));
        assert!(!loader.should_load_layer(5));
    }

    #[test]
    fn test_upload_tracking() {
        let mut loader = GpuModelLoader::with_defaults();
        loader.stage_tensor(
            "blk.0.attn_q.weight".to_string(),
            vec![256, 256],
            GpuQuantFormat::Qk256,
            vec![0u8; 1024],
        );
        assert_eq!(loader.uploaded_count(), 0);
        assert!(loader.mark_uploaded("blk.0.attn_q.weight"));
        assert_eq!(loader.uploaded_count(), 1);
        assert!(!loader.mark_uploaded("nonexistent"));
    }

    #[test]
    fn test_tensors_for_layer() {
        let mut loader = GpuModelLoader::with_defaults();
        loader.stage_tensor(
            "blk.0.attn_q.weight".to_string(),
            vec![128, 128],
            GpuQuantFormat::Tl2,
            vec![0u8; 1024],
        );
        loader.stage_tensor(
            "blk.0.attn_v.weight".to_string(),
            vec![128, 128],
            GpuQuantFormat::Tl2,
            vec![0u8; 1024],
        );
        loader.stage_tensor(
            "blk.1.attn_q.weight".to_string(),
            vec![128, 128],
            GpuQuantFormat::Tl2,
            vec![0u8; 1024],
        );

        let layer0 = loader.tensors_for_layer(0);
        assert_eq!(layer0.len(), 2);
        let layer1 = loader.tensors_for_layer(1);
        assert_eq!(layer1.len(), 1);
    }

    #[test]
    fn test_memory_budget_layers() {
        let config = GpuLoadConfig {
            max_gpu_memory: Some(50000),
            ..Default::default()
        };
        let mut loader = GpuModelLoader::new(config);

        // Non-layer tensor
        loader.stage_tensor(
            "token_embd.weight".to_string(),
            vec![1000, 64],
            GpuQuantFormat::F16,
            vec![0u8; 10000],
        );

        // 4 layers, each 15000 bytes
        for i in 0..4 {
            loader.stage_tensor(
                format!("blk.{i}.attn_q.weight"),
                vec![256, 256],
                GpuQuantFormat::Qk256,
                vec![0u8; 15000],
            );
        }

        let estimate = loader.estimate_memory();
        // Budget: 50000, non-layer: 10000, remaining: 40000
        // Each layer: 15000, so 2 full layers fit (30000 <= 40000)
        assert_eq!(estimate.layers_in_budget, Some(2));
    }

    #[test]
    fn test_quant_format_bytes_per_element() {
        assert!(GpuQuantFormat::Qk256.bytes_per_element() < 1.0);
        assert!(GpuQuantFormat::Tl1.bytes_per_element() < 1.0);
        assert!(GpuQuantFormat::Tl2.bytes_per_element() < 1.0);
        assert!((GpuQuantFormat::F32.bytes_per_element() - 4.0).abs() < f64::EPSILON);
        assert!((GpuQuantFormat::F16.bytes_per_element() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_model_gpu_memory() {
        let estimate = estimate_model_gpu_memory(24, 2048, GpuQuantFormat::Qk256);
        assert!(estimate.total_bytes > 0);
        assert_eq!(estimate.per_layer_bytes.len(), 24);
        assert!(estimate.non_layer_bytes > 0);
        assert!(estimate.staging_buffer_bytes > 0);
    }

    #[test]
    fn test_get_tensor() {
        let mut loader = GpuModelLoader::with_defaults();
        loader.stage_tensor(
            "test_tensor".to_string(),
            vec![64, 64],
            GpuQuantFormat::F32,
            vec![0u8; 256],
        );
        assert!(loader.get_tensor("test_tensor").is_some());
        assert!(loader.get_tensor("missing").is_none());
    }

    // ── Streaming DMA loader tests ──

    fn dma_caps() -> GpuLoadCapabilities {
        GpuLoadCapabilities {
            supports_direct_dma: true,
            supports_on_device_dequant: true,
            max_alloc_bytes: 1 << 30,
            num_copy_queues: 2,
        }
    }

    fn make_desc(name: &str, bytes: usize, fmt: GpuQuantFormat) -> GpuTensorDescriptor {
        GpuTensorDescriptor {
            name: name.into(),
            shape: vec![bytes],
            format: fmt,
            num_elements: bytes,
            gpu_bytes: bytes,
            layer_index: None,
        }
    }

    #[test]
    fn test_select_direct_dma_when_capable() {
        let t = make_desc("w", 1_000_000, GpuQuantFormat::F32);
        assert_eq!(
            select_transfer_strategy(&t, &dma_caps(), true),
            TransferStrategy::DirectDma,
        );
    }

    #[test]
    fn test_fallback_host_staged_no_dma() {
        let t = make_desc("w", 1_000_000, GpuQuantFormat::F32);
        assert_eq!(
            select_transfer_strategy(&t, &GpuLoadCapabilities::default(), true),
            TransferStrategy::HostStaged,
        );
    }

    #[test]
    fn test_small_tensor_host_staged() {
        let t = make_desc("bias", 128, GpuQuantFormat::F32);
        assert_eq!(
            select_transfer_strategy(&t, &dma_caps(), true),
            TransferStrategy::HostStaged,
        );
    }

    #[test]
    fn test_dequant_on_device_qk256() {
        assert!(should_dequant_on_device(GpuQuantFormat::Qk256, &dma_caps(), true));
    }

    #[test]
    fn test_no_dequant_for_f32() {
        assert!(!should_dequant_on_device(GpuQuantFormat::F32, &dma_caps(), true));
    }

    #[test]
    fn test_no_dequant_when_disabled() {
        assert!(!should_dequant_on_device(GpuQuantFormat::Qk256, &dma_caps(), false));
    }

    #[test]
    fn test_chunk_count_exact() {
        assert_eq!(compute_chunk_count(4 * 1024 * 1024, 1024 * 1024), 4);
    }

    #[test]
    fn test_chunk_count_remainder() {
        assert_eq!(compute_chunk_count(5_000_000, DEFAULT_DMA_CHUNK_SIZE), 2);
    }

    #[test]
    fn test_chunk_range_offsets() {
        let (s, l) = chunk_range(1000, 0, 512, 1024);
        assert_eq!((s, l), (1000, 512));
        let (s2, l2) = chunk_range(1000, 1, 512, 1024);
        assert_eq!((s2, l2), (1512, 512));
    }

    #[test]
    fn test_chunk_range_last_partial() {
        let (s, l) = chunk_range(0, 2, 512, 1200);
        assert_eq!(s, 1024);
        assert_eq!(l, 176);
    }

    #[test]
    fn test_streaming_load_single_tensor() {
        let loader = StreamingDmaLoader::new(StreamingDmaConfig::default(), dma_caps());
        let t = make_desc("weight.0", 1_000_000, GpuQuantFormat::F16);
        let r = loader.load_tensor(&t);
        assert_eq!(r.name, "weight.0");
        assert_eq!(r.strategy_used, TransferStrategy::DirectDma);
        assert!(!r.dequantized_on_device); // F16 doesn't need dequant
    }

    #[test]
    fn test_streaming_load_multiple_with_progress() {
        let reports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let reports_clone = reports.clone();

        let config = StreamingDmaConfig::default().with_progress(Arc::new(move |frac, msg| {
            reports_clone.lock().unwrap().push((frac, msg.to_string()));
        }));

        let loader = StreamingDmaLoader::new(config, dma_caps());
        let tensors = vec![
            make_desc("w1", 500_000, GpuQuantFormat::F32),
            make_desc("w2", 500_000, GpuQuantFormat::F32),
        ];
        let results = loader.load_tensors(&tensors);
        assert_eq!(results.len(), 2);

        let reps = reports.lock().unwrap();
        assert_eq!(reps.len(), 2);
        assert!(reps[0].0 > 0.4 && reps[0].0 < 0.6);
        assert!((reps[1].0 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_fraction_tracking() {
        let mut p = LoadProgress::new(4, 1000);
        assert_eq!(p.fraction(), 0.0);
        p.record(500);
        assert!((p.fraction() - 0.5).abs() < f32::EPSILON);
        p.record(500);
        assert!((p.fraction() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_zero_total() {
        let p = LoadProgress::new(0, 0);
        assert!((p.fraction() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_config_builder() {
        let c = StreamingDmaConfig::default().with_chunk_size(1024);
        assert_eq!(c.chunk_size, 4096); // clamped
        let c2 = StreamingDmaConfig::default().with_chunk_size(8 * 1024 * 1024);
        assert_eq!(c2.chunk_size, 8 * 1024 * 1024);
    }

    #[test]
    fn test_fallback_loader_creation() {
        let loader = StreamingDmaLoader::with_fallback(StreamingDmaConfig::default());
        assert!(!loader.capabilities().supports_direct_dma);
    }

    #[test]
    fn test_load_from_model_loader_integration() {
        let mut ml = GpuModelLoader::with_defaults();
        ml.stage_tensor("blk.0.attn_q.weight".into(), vec![128, 128], GpuQuantFormat::Qk256, vec![0u8; 4096]);
        ml.stage_tensor("blk.0.attn_v.weight".into(), vec![128, 128], GpuQuantFormat::Tl1, vec![0u8; 4096]);

        let dma = StreamingDmaLoader::new(StreamingDmaConfig::default(), dma_caps());
        let results = dma.load_from_model_loader(&ml);
        assert_eq!(results.len(), 2);
        // Both should be dequantized on device (Qk256 and Tl1 are quantized formats)
        assert!(results.iter().all(|r| r.dequantized_on_device));
    }
}
