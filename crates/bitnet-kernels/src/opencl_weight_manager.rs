//! OpenCL model weight management for GPU inference.
//!
//! Manages model weight loading, GPU upload, memory mapping, and weight
//! sharing across inference instances. All implementations are CPU reference
//! code — no OpenCL runtime required.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// WeightDtype — supported data types
// ---------------------------------------------------------------------------

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightDtype {
    /// 32-bit float.
    F32,
    /// 16-bit float (IEEE 754).
    F16,
    /// 16-bit brain float.
    BF16,
    /// 8-bit integer (signed).
    I8,
    /// 2-bit ternary packed (I2_S).
    I2S,
    /// QK256 block-quantised.
    QK256,
}

impl WeightDtype {
    /// Bytes per element (for packed types, bytes per logical element).
    pub fn element_bytes(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::I8 => 1.0,
            Self::I2S => 0.25,   // 4 values per byte
            Self::QK256 => 0.25, // approximate for block format
        }
    }
}

impl fmt::Display for WeightDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2S => write!(f, "i2s"),
            Self::QK256 => write!(f, "qk256"),
        }
    }
}

// ---------------------------------------------------------------------------
// WeightDescriptor
// ---------------------------------------------------------------------------

/// Describes a single model weight tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightDescriptor {
    /// Tensor name (e.g. "layers.0.attention.wq.weight").
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: WeightDtype,
    /// Byte offset within the model file.
    pub byte_offset: usize,
    /// Total size in bytes.
    pub byte_size: usize,
}

impl WeightDescriptor {
    /// Create a new weight descriptor.
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: WeightDtype,
        byte_offset: usize,
        byte_size: usize,
    ) -> Self {
        Self { name: name.into(), shape, dtype, byte_offset, byte_size }
    }

    /// Number of logical elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Validate the descriptor for internal consistency.
    pub fn validate(&self) -> Result<(), WeightError> {
        if self.name.is_empty() {
            return Err(WeightError::InvalidDescriptor("weight name must not be empty".into()));
        }
        if self.shape.is_empty() {
            return Err(WeightError::InvalidDescriptor(
                "shape must have at least one dimension".into(),
            ));
        }
        if self.shape.contains(&0) {
            return Err(WeightError::InvalidDescriptor("shape dimensions must be non-zero".into()));
        }
        if self.byte_size == 0 {
            return Err(WeightError::InvalidDescriptor("byte_size must be non-zero".into()));
        }
        Ok(())
    }

    /// Extract the layer index from the name, if present.
    pub fn layer_index(&self) -> Option<usize> {
        // Match "layers.N." or "blk.N."
        for part in self.name.split('.') {
            if let Ok(idx) = part.parse::<usize>() {
                return Some(idx);
            }
        }
        None
    }
}

impl fmt::Display for WeightDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}, {}, {} bytes @ offset {})",
            self.name, self.shape, self.dtype, self.byte_size, self.byte_offset
        )
    }
}

// ---------------------------------------------------------------------------
// WeightError
// ---------------------------------------------------------------------------

/// Errors from the weight management subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    /// Descriptor failed validation.
    InvalidDescriptor(String),
    /// Requested weight not found.
    NotFound(String),
    /// Weight data size mismatch.
    SizeMismatch { expected: usize, got: usize },
    /// GPU memory exhausted.
    GpuMemoryExhausted { requested: usize, available: usize },
    /// Weight already uploaded.
    AlreadyUploaded(String),
    /// Memory mapping failure.
    MmapFailed(String),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(msg) => write!(f, "invalid descriptor: {msg}"),
            Self::NotFound(name) => write!(f, "weight not found: {name}"),
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected}, got {got}")
            }
            Self::GpuMemoryExhausted { requested, available } => {
                write!(
                    f,
                    "GPU memory exhausted: need {requested} bytes, \
                     {available} available"
                )
            }
            Self::AlreadyUploaded(name) => {
                write!(f, "weight already uploaded: {name}")
            }
            Self::MmapFailed(msg) => write!(f, "mmap failed: {msg}"),
        }
    }
}

impl std::error::Error for WeightError {}

// ---------------------------------------------------------------------------
// WeightStore — host-side weight storage
// ---------------------------------------------------------------------------

/// Holds all model weights in host memory.
///
/// Supports lazy loading: descriptors can be registered before actual data
/// is provided. Data is loaded on first access.
#[derive(Debug)]
pub struct WeightStore {
    descriptors: Vec<WeightDescriptor>,
    /// Name → index in `descriptors`.
    name_index: HashMap<String, usize>,
    /// Actual weight bytes, keyed by descriptor index.
    data: HashMap<usize, Vec<u8>>,
}

impl WeightStore {
    pub fn new() -> Self {
        Self { descriptors: Vec::new(), name_index: HashMap::new(), data: HashMap::new() }
    }

    /// Register a weight descriptor without loading data yet.
    pub fn register(&mut self, desc: WeightDescriptor) -> Result<(), WeightError> {
        desc.validate()?;
        let idx = self.descriptors.len();
        self.name_index.insert(desc.name.clone(), idx);
        self.descriptors.push(desc);
        Ok(())
    }

    /// Load raw bytes for a previously registered weight.
    pub fn load_data(&mut self, name: &str, data: Vec<u8>) -> Result<(), WeightError> {
        let &idx = self.name_index.get(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        let expected = self.descriptors[idx].byte_size;
        if data.len() != expected {
            return Err(WeightError::SizeMismatch { expected, got: data.len() });
        }
        self.data.insert(idx, data);
        Ok(())
    }

    /// Register a weight and load its data in one step.
    pub fn register_with_data(
        &mut self,
        desc: WeightDescriptor,
        data: Vec<u8>,
    ) -> Result<(), WeightError> {
        desc.validate()?;
        if data.len() != desc.byte_size {
            return Err(WeightError::SizeMismatch { expected: desc.byte_size, got: data.len() });
        }
        let idx = self.descriptors.len();
        self.name_index.insert(desc.name.clone(), idx);
        self.descriptors.push(desc);
        self.data.insert(idx, data);
        Ok(())
    }

    /// Get a reference to weight data by name.
    pub fn get_data(&self, name: &str) -> Result<&[u8], WeightError> {
        let &idx = self.name_index.get(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        self.data.get(&idx).map(|v| v.as_slice()).ok_or_else(|| WeightError::NotFound(name.into()))
    }

    /// Get a weight descriptor by name.
    pub fn get_descriptor(&self, name: &str) -> Result<&WeightDescriptor, WeightError> {
        let &idx = self.name_index.get(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        Ok(&self.descriptors[idx])
    }

    /// Number of registered weights.
    pub fn weight_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Number of weights with loaded data.
    pub fn loaded_count(&self) -> usize {
        self.data.len()
    }

    /// Total bytes of loaded data.
    pub fn total_loaded_bytes(&self) -> usize {
        self.data.values().map(|v| v.len()).sum()
    }

    /// Whether data is loaded for the given name.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.name_index.get(name).is_some_and(|idx| self.data.contains_key(idx))
    }

    /// Iterate over all descriptors.
    pub fn descriptors(&self) -> &[WeightDescriptor] {
        &self.descriptors
    }

    /// All weight names.
    pub fn names(&self) -> Vec<&str> {
        self.descriptors.iter().map(|d| d.name.as_str()).collect()
    }

    /// Remove loaded data for a weight (keeps descriptor).
    pub fn unload(&mut self, name: &str) -> Result<(), WeightError> {
        let &idx = self.name_index.get(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        self.data.remove(&idx);
        Ok(())
    }
}

impl Default for WeightStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GpuWeightBuffer — simulated GPU-resident buffer
// ---------------------------------------------------------------------------

/// Upload status for a single weight buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadStatus {
    /// Not yet uploaded.
    Pending,
    /// Upload in progress.
    InProgress,
    /// Successfully uploaded to GPU.
    Complete,
    /// Upload failed.
    Failed,
}

impl fmt::Display for UploadStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Complete => write!(f, "complete"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// A GPU-resident weight buffer (CPU reference simulation).
#[derive(Debug, Clone)]
pub struct GpuWeightBuffer {
    /// Name of the weight this buffer holds.
    pub name: String,
    /// Size in bytes on the GPU.
    pub size: usize,
    /// Current upload status.
    pub status: UploadStatus,
    /// Simulated GPU data (in CPU reference mode).
    data: Option<Vec<u8>>,
}

impl GpuWeightBuffer {
    /// Create a new pending GPU buffer.
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self { name: name.into(), size, status: UploadStatus::Pending, data: None }
    }

    /// Simulate uploading data to GPU.
    pub fn upload(&mut self, data: &[u8]) -> Result<(), WeightError> {
        if data.len() != self.size {
            return Err(WeightError::SizeMismatch { expected: self.size, got: data.len() });
        }
        self.status = UploadStatus::InProgress;
        self.data = Some(data.to_vec());
        self.status = UploadStatus::Complete;
        Ok(())
    }

    /// Read back the GPU data (for verification).
    pub fn readback(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    /// Whether the upload is complete.
    pub fn is_ready(&self) -> bool {
        self.status == UploadStatus::Complete
    }

    /// Release GPU memory.
    pub fn release(&mut self) {
        self.data = None;
        self.status = UploadStatus::Pending;
    }

    /// Resident bytes (0 if not uploaded).
    pub fn resident_bytes(&self) -> usize {
        if self.is_ready() { self.size } else { 0 }
    }
}

// ---------------------------------------------------------------------------
// WeightUploader — async upload with progress
// ---------------------------------------------------------------------------

/// Progress of a batch weight upload operation.
#[derive(Debug, Clone)]
pub struct UploadProgress {
    /// Total weights to upload.
    pub total: usize,
    /// Weights completed so far.
    pub completed: usize,
    /// Total bytes to upload.
    pub total_bytes: usize,
    /// Bytes uploaded so far.
    pub uploaded_bytes: usize,
    /// Simulated bandwidth in bytes/ns (GB/s equivalent).
    pub bandwidth_gbps: f64,
}

impl UploadProgress {
    /// Fraction complete in [0.0, 1.0].
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.completed as f64 / self.total as f64
    }

    /// Whether upload is done.
    pub fn is_done(&self) -> bool {
        self.completed >= self.total
    }
}

/// Manages uploading weights from host to GPU.
#[derive(Debug)]
pub struct WeightUploader {
    /// GPU buffers keyed by weight name.
    buffers: HashMap<String, GpuWeightBuffer>,
    /// Simulated GPU memory budget in bytes.
    gpu_budget: usize,
    /// Bytes currently resident on GPU.
    gpu_used: usize,
    /// Simulated upload bandwidth in GB/s.
    bandwidth_gbps: f64,
    /// Total upload duration simulated (ns).
    total_upload_ns: u64,
}

impl WeightUploader {
    /// Create an uploader with a GPU memory budget and simulated bandwidth.
    pub fn new(gpu_budget: usize, bandwidth_gbps: f64) -> Self {
        Self {
            buffers: HashMap::new(),
            gpu_budget,
            gpu_used: 0,
            bandwidth_gbps,
            total_upload_ns: 0,
        }
    }

    /// Upload a single weight from host data.
    pub fn upload_weight(&mut self, name: &str, data: &[u8]) -> Result<(), WeightError> {
        if self.buffers.contains_key(name) && self.buffers[name].status == UploadStatus::Complete {
            return Err(WeightError::AlreadyUploaded(name.into()));
        }
        let needed = data.len();
        if self.gpu_used + needed > self.gpu_budget {
            return Err(WeightError::GpuMemoryExhausted {
                requested: needed,
                available: self.gpu_budget.saturating_sub(self.gpu_used),
            });
        }
        let mut buf = GpuWeightBuffer::new(name, needed);
        buf.upload(data)?;
        self.gpu_used += needed;
        // Simulate transfer time
        if self.bandwidth_gbps > 0.0 {
            let ns = (needed as f64 / self.bandwidth_gbps).round() as u64;
            self.total_upload_ns += ns;
        }
        self.buffers.insert(name.to_string(), buf);
        Ok(())
    }

    /// Batch upload from a `WeightStore`.
    pub fn upload_all(&mut self, store: &WeightStore) -> Result<UploadProgress, WeightError> {
        let names: Vec<String> = store.names().into_iter().map(Into::into).collect();
        let total = names.len();
        let total_bytes: usize = store.descriptors().iter().map(|d| d.byte_size).sum();
        let mut completed = 0;
        let mut uploaded_bytes = 0;

        for name in &names {
            if let Ok(data) = store.get_data(name) {
                self.upload_weight(name, data)?;
                completed += 1;
                uploaded_bytes += data.len();
            }
        }

        Ok(UploadProgress {
            total,
            completed,
            total_bytes,
            uploaded_bytes,
            bandwidth_gbps: self.bandwidth_gbps,
        })
    }

    /// Get a reference to a GPU buffer.
    pub fn get_buffer(&self, name: &str) -> Option<&GpuWeightBuffer> {
        self.buffers.get(name)
    }

    /// Release a single weight from GPU.
    pub fn release_weight(&mut self, name: &str) -> Result<(), WeightError> {
        let buf = self.buffers.get_mut(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        self.gpu_used = self.gpu_used.saturating_sub(buf.size);
        buf.release();
        Ok(())
    }

    /// Release all GPU weight buffers.
    pub fn release_all(&mut self) {
        for buf in self.buffers.values_mut() {
            buf.release();
        }
        self.gpu_used = 0;
    }

    /// Current GPU memory usage.
    pub fn gpu_used_bytes(&self) -> usize {
        self.gpu_used
    }

    /// Available GPU memory.
    pub fn gpu_available_bytes(&self) -> usize {
        self.gpu_budget.saturating_sub(self.gpu_used)
    }

    /// Number of uploaded weights.
    pub fn uploaded_count(&self) -> usize {
        self.buffers.values().filter(|b| b.is_ready()).count()
    }

    /// Total simulated upload time in nanoseconds.
    pub fn total_upload_time_ns(&self) -> u64 {
        self.total_upload_ns
    }

    /// Current progress snapshot.
    pub fn progress(&self) -> UploadProgress {
        let completed = self.uploaded_count();
        let total = self.buffers.len();
        let uploaded_bytes = self.buffers.values().map(|b| b.resident_bytes()).sum();
        let total_bytes: usize = self.buffers.values().map(|b| b.size).sum();
        UploadProgress {
            total,
            completed,
            total_bytes,
            uploaded_bytes,
            bandwidth_gbps: self.bandwidth_gbps,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightSharing — reference-counted weight sharing
// ---------------------------------------------------------------------------

/// Reference-counted handle to a shared weight store + GPU uploader.
///
/// Multiple inference instances can share the same weights on the GPU
/// without duplicating memory.
#[derive(Debug)]
pub struct WeightSharing {
    /// The shared weight store.
    store: Arc<RwLock<WeightStore>>,
    /// Global reference count.
    ref_count: Arc<AtomicU64>,
    /// Unique id for this sharing group.
    group_id: u64,
}

/// Next group-id counter.
static NEXT_GROUP_ID: AtomicU64 = AtomicU64::new(1);

impl WeightSharing {
    /// Create a new sharing group wrapping an existing store.
    pub fn new(store: WeightStore) -> Self {
        Self {
            store: Arc::new(RwLock::new(store)),
            ref_count: Arc::new(AtomicU64::new(1)),
            group_id: NEXT_GROUP_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Acquire an additional reference.
    pub fn acquire(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
        Self {
            store: Arc::clone(&self.store),
            ref_count: Arc::clone(&self.ref_count),
            group_id: self.group_id,
        }
    }

    /// Release this reference. Returns the remaining count.
    pub fn release(&self) -> u64 {
        let prev = self.ref_count.fetch_sub(1, Ordering::AcqRel);
        prev.saturating_sub(1)
    }

    /// Current reference count.
    pub fn ref_count(&self) -> u64 {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Group identifier.
    pub fn group_id(&self) -> u64 {
        self.group_id
    }

    /// Read access to the underlying store.
    pub fn read_store(&self) -> std::sync::RwLockReadGuard<'_, WeightStore> {
        self.store.read().expect("WeightStore lock poisoned")
    }

    /// Write access to the underlying store.
    pub fn write_store(&self) -> std::sync::RwLockWriteGuard<'_, WeightStore> {
        self.store.write().expect("WeightStore lock poisoned")
    }
}

impl Clone for WeightSharing {
    fn clone(&self) -> Self {
        self.acquire()
    }
}

// ---------------------------------------------------------------------------
// MemoryMappedWeights — mmap-based access simulation
// ---------------------------------------------------------------------------

/// Simulated memory-mapped weight access for large models.
///
/// In production this would use `mmap(2)` / `CreateFileMapping`; the CPU
/// reference stores a contiguous byte buffer and serves slices from it.
#[derive(Debug)]
pub struct MemoryMappedWeights {
    /// Backing buffer (simulates the mapped region).
    backing: Vec<u8>,
    /// Descriptor registry for region lookup.
    descriptors: Vec<WeightDescriptor>,
    /// Name → descriptor index.
    name_index: HashMap<String, usize>,
    /// Whether the mapping is currently active.
    active: bool,
}

impl MemoryMappedWeights {
    /// Create a new memory-mapped region from raw bytes and descriptors.
    pub fn new(backing: Vec<u8>, descriptors: Vec<WeightDescriptor>) -> Result<Self, WeightError> {
        let mut name_index = HashMap::new();
        for (i, desc) in descriptors.iter().enumerate() {
            desc.validate()?;
            let end = desc.byte_offset + desc.byte_size;
            if end > backing.len() {
                return Err(WeightError::MmapFailed(format!(
                    "weight '{}' extends beyond mapped region \
                     (end={end}, region={})",
                    desc.name,
                    backing.len()
                )));
            }
            name_index.insert(desc.name.clone(), i);
        }
        Ok(Self { backing, descriptors, name_index, active: true })
    }

    /// Access weight data by name.
    pub fn get(&self, name: &str) -> Result<&[u8], WeightError> {
        if !self.active {
            return Err(WeightError::MmapFailed("mapping is not active".into()));
        }
        let &idx = self.name_index.get(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        let desc = &self.descriptors[idx];
        let start = desc.byte_offset;
        let end = start + desc.byte_size;
        Ok(&self.backing[start..end])
    }

    /// Total mapped region size.
    pub fn mapped_size(&self) -> usize {
        self.backing.len()
    }

    /// Number of weights in the mapped region.
    pub fn weight_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Deactivate the mapping (simulate munmap).
    pub fn unmap(&mut self) {
        self.active = false;
    }

    /// Whether the mapping is active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Reactivate the mapping.
    pub fn remap(&mut self) {
        self.active = true;
    }

    /// Get descriptor by name.
    pub fn get_descriptor(&self, name: &str) -> Result<&WeightDescriptor, WeightError> {
        let &idx = self.name_index.get(name).ok_or_else(|| WeightError::NotFound(name.into()))?;
        Ok(&self.descriptors[idx])
    }
}

// ---------------------------------------------------------------------------
// WeightPrefetcher — layer-ahead prefetch scheduling
// ---------------------------------------------------------------------------

/// Prefetch request issued by the prefetcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchRequest {
    /// Weight names to prefetch.
    pub names: Vec<String>,
    /// Target layer being prefetched for.
    pub layer_index: usize,
}

/// Schedules prefetch of upcoming layer weights during inference.
///
/// Given the current layer, the prefetcher returns which weights should be
/// uploaded next (lookahead of `depth` layers).
#[derive(Debug)]
pub struct WeightPrefetcher {
    /// All weight descriptors grouped by layer index.
    layer_weights: HashMap<usize, Vec<String>>,
    /// Number of layers to look ahead.
    depth: usize,
    /// Total layer count.
    total_layers: usize,
    /// Layers already prefetched.
    prefetched: Vec<bool>,
}

impl WeightPrefetcher {
    /// Build a prefetcher from descriptors.
    pub fn new(descriptors: &[WeightDescriptor], depth: usize) -> Self {
        let mut layer_weights: HashMap<usize, Vec<String>> = HashMap::new();
        let mut max_layer = 0usize;
        for desc in descriptors {
            if let Some(idx) = desc.layer_index() {
                layer_weights.entry(idx).or_default().push(desc.name.clone());
                if idx > max_layer {
                    max_layer = idx;
                }
            }
        }
        let total_layers = if layer_weights.is_empty() { 0 } else { max_layer + 1 };
        let prefetched = vec![false; total_layers];
        Self { layer_weights, depth, total_layers, prefetched }
    }

    /// Get prefetch requests for layers ahead of `current_layer`.
    pub fn prefetch_for(&mut self, current_layer: usize) -> Vec<PrefetchRequest> {
        let mut requests = Vec::new();
        let start = current_layer + 1;
        let end = (current_layer + 1 + self.depth).min(self.total_layers);
        for layer in start..end {
            if layer < self.prefetched.len()
                && !self.prefetched[layer]
                && let Some(names) = self.layer_weights.get(&layer)
            {
                requests.push(PrefetchRequest { names: names.clone(), layer_index: layer });
                self.prefetched[layer] = true;
            }
        }
        requests
    }

    /// Reset prefetch state (e.g. for a new generation).
    pub fn reset(&mut self) {
        self.prefetched.fill(false);
    }

    /// Total number of layers tracked.
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Lookahead depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Names of weights for a specific layer.
    pub fn weights_for_layer(&self, layer: usize) -> Option<&[String]> {
        self.layer_weights.get(&layer).map(|v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// WeightStats — aggregate statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics about model weights.
#[derive(Debug, Clone, Default)]
pub struct WeightStats {
    /// Total size of all weights in bytes.
    pub total_size: usize,
    /// Bytes currently resident on GPU.
    pub gpu_resident: usize,
    /// Simulated upload bandwidth in GB/s.
    pub upload_bandwidth_gbps: f64,
    /// Number of active sharing references.
    pub share_count: u64,
    /// Number of weight tensors.
    pub weight_count: usize,
    /// Number of weights uploaded to GPU.
    pub gpu_weight_count: usize,
    /// Bytes loaded in host memory.
    pub host_loaded: usize,
}

impl WeightStats {
    /// Gather stats from a store and uploader.
    pub fn gather(store: &WeightStore, uploader: &WeightUploader, share_count: u64) -> Self {
        let total_size: usize = store.descriptors().iter().map(|d| d.byte_size).sum();
        Self {
            total_size,
            gpu_resident: uploader.gpu_used_bytes(),
            upload_bandwidth_gbps: uploader.progress().bandwidth_gbps,
            share_count,
            weight_count: store.weight_count(),
            gpu_weight_count: uploader.uploaded_count(),
            host_loaded: store.total_loaded_bytes(),
        }
    }

    /// Fraction of model resident on GPU.
    pub fn gpu_fraction(&self) -> f64 {
        if self.total_size == 0 {
            return 0.0;
        }
        self.gpu_resident as f64 / self.total_size as f64
    }
}

impl fmt::Display for WeightStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "weights={}, total={} B, gpu={} B ({:.1}%), \
             bw={:.1} GB/s, shares={}",
            self.weight_count,
            self.total_size,
            self.gpu_resident,
            self.gpu_fraction() * 100.0,
            self.upload_bandwidth_gbps,
            self.share_count,
        )
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn desc(name: &str, shape: Vec<usize>, byte_size: usize) -> WeightDescriptor {
        WeightDescriptor::new(name, shape, WeightDtype::F32, 0, byte_size)
    }

    fn desc_at(name: &str, shape: Vec<usize>, offset: usize, byte_size: usize) -> WeightDescriptor {
        WeightDescriptor::new(name, shape, WeightDtype::F32, offset, byte_size)
    }

    fn sample_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i & 0xFF) as u8).collect()
    }

    fn layer_desc(layer: usize, suffix: &str, size: usize) -> WeightDescriptor {
        WeightDescriptor::new(
            format!("layers.{layer}.{suffix}"),
            vec![size],
            WeightDtype::F32,
            0,
            size,
        )
    }

    // =====================================================================
    // WeightDescriptor tests
    // =====================================================================

    #[test]
    fn descriptor_basic_creation() {
        let d = desc("w", vec![4, 4], 64);
        assert_eq!(d.name, "w");
        assert_eq!(d.shape, vec![4, 4]);
        assert_eq!(d.dtype, WeightDtype::F32);
        assert_eq!(d.byte_size, 64);
    }

    #[test]
    fn descriptor_num_elements() {
        let d = desc("w", vec![3, 4, 5], 240);
        assert_eq!(d.num_elements(), 60);
    }

    #[test]
    fn descriptor_validate_ok() {
        let d = desc("w", vec![4], 16);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn descriptor_validate_empty_name() {
        let d = desc("", vec![4], 16);
        assert!(matches!(d.validate(), Err(WeightError::InvalidDescriptor(_))));
    }

    #[test]
    fn descriptor_validate_empty_shape() {
        let d = desc("w", vec![], 16);
        assert!(matches!(d.validate(), Err(WeightError::InvalidDescriptor(_))));
    }

    #[test]
    fn descriptor_validate_zero_dim() {
        let d = desc("w", vec![4, 0], 16);
        assert!(matches!(d.validate(), Err(WeightError::InvalidDescriptor(_))));
    }

    #[test]
    fn descriptor_validate_zero_byte_size() {
        let d = desc("w", vec![4], 0);
        assert!(matches!(d.validate(), Err(WeightError::InvalidDescriptor(_))));
    }

    #[test]
    fn descriptor_layer_index_present() {
        let d = desc("layers.5.wq", vec![4], 16);
        assert_eq!(d.layer_index(), Some(5));
    }

    #[test]
    fn descriptor_layer_index_absent() {
        let d = desc("embedding.weight", vec![4], 16);
        assert_eq!(d.layer_index(), None);
    }

    #[test]
    fn descriptor_display() {
        let d = desc("w", vec![4], 16);
        let s = format!("{d}");
        assert!(s.contains("w"));
        assert!(s.contains("16 bytes"));
    }

    #[test]
    fn descriptor_layer_index_blk_format() {
        let d = desc("blk.12.attn_q", vec![4], 16);
        assert_eq!(d.layer_index(), Some(12));
    }

    // =====================================================================
    // WeightDtype tests
    // =====================================================================

    #[test]
    fn dtype_element_bytes() {
        assert_eq!(WeightDtype::F32.element_bytes(), 4.0);
        assert_eq!(WeightDtype::F16.element_bytes(), 2.0);
        assert_eq!(WeightDtype::BF16.element_bytes(), 2.0);
        assert_eq!(WeightDtype::I8.element_bytes(), 1.0);
        assert_eq!(WeightDtype::I2S.element_bytes(), 0.25);
        assert_eq!(WeightDtype::QK256.element_bytes(), 0.25);
    }

    #[test]
    fn dtype_display() {
        assert_eq!(format!("{}", WeightDtype::F32), "f32");
        assert_eq!(format!("{}", WeightDtype::I2S), "i2s");
        assert_eq!(format!("{}", WeightDtype::QK256), "qk256");
    }

    #[test]
    fn dtype_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightDtype::F32);
        set.insert(WeightDtype::F16);
        set.insert(WeightDtype::F32); // duplicate
        assert_eq!(set.len(), 2);
    }

    // =====================================================================
    // WeightStore tests
    // =====================================================================

    #[test]
    fn store_register_and_count() {
        let mut store = WeightStore::new();
        store.register(desc("w1", vec![4], 16)).unwrap();
        store.register(desc("w2", vec![8], 32)).unwrap();
        assert_eq!(store.weight_count(), 2);
        assert_eq!(store.loaded_count(), 0);
    }

    #[test]
    fn store_load_and_get() {
        let mut store = WeightStore::new();
        store.register(desc("w", vec![4], 4)).unwrap();
        store.load_data("w", vec![1, 2, 3, 4]).unwrap();
        assert_eq!(store.get_data("w").unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn store_load_size_mismatch() {
        let mut store = WeightStore::new();
        store.register(desc("w", vec![4], 4)).unwrap();
        let err = store.load_data("w", vec![1, 2]).unwrap_err();
        assert!(matches!(err, WeightError::SizeMismatch { .. }));
    }

    #[test]
    fn store_load_not_found() {
        let mut store = WeightStore::new();
        let err = store.load_data("missing", vec![]).unwrap_err();
        assert!(matches!(err, WeightError::NotFound(_)));
    }

    #[test]
    fn store_register_with_data() {
        let mut store = WeightStore::new();
        let data = vec![0xAA; 8];
        store.register_with_data(desc("w", vec![2], 8), data.clone()).unwrap();
        assert_eq!(store.weight_count(), 1);
        assert_eq!(store.loaded_count(), 1);
        assert_eq!(store.get_data("w").unwrap(), &data[..]);
    }

    #[test]
    fn store_register_with_data_size_mismatch() {
        let mut store = WeightStore::new();
        let err = store.register_with_data(desc("w", vec![2], 8), vec![0; 4]).unwrap_err();
        assert!(matches!(err, WeightError::SizeMismatch { .. }));
    }

    #[test]
    fn store_get_descriptor() {
        let mut store = WeightStore::new();
        store.register(desc("w", vec![4], 16)).unwrap();
        let d = store.get_descriptor("w").unwrap();
        assert_eq!(d.name, "w");
    }

    #[test]
    fn store_is_loaded() {
        let mut store = WeightStore::new();
        store.register(desc("w", vec![4], 4)).unwrap();
        assert!(!store.is_loaded("w"));
        store.load_data("w", vec![0; 4]).unwrap();
        assert!(store.is_loaded("w"));
    }

    #[test]
    fn store_names() {
        let mut store = WeightStore::new();
        store.register(desc("b", vec![1], 4)).unwrap();
        store.register(desc("a", vec![1], 4)).unwrap();
        let names = store.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn store_unload() {
        let mut store = WeightStore::new();
        store.register_with_data(desc("w", vec![1], 4), vec![0; 4]).unwrap();
        assert!(store.is_loaded("w"));
        store.unload("w").unwrap();
        assert!(!store.is_loaded("w"));
        // descriptor still present
        assert_eq!(store.weight_count(), 1);
    }

    #[test]
    fn store_unload_not_found() {
        let mut store = WeightStore::new();
        assert!(matches!(store.unload("x"), Err(WeightError::NotFound(_))));
    }

    #[test]
    fn store_total_loaded_bytes() {
        let mut store = WeightStore::new();
        store.register_with_data(desc("a", vec![1], 4), vec![0; 4]).unwrap();
        store.register_with_data(desc("b", vec![1], 8), vec![0; 8]).unwrap();
        assert_eq!(store.total_loaded_bytes(), 12);
    }

    #[test]
    fn store_empty() {
        let store = WeightStore::new();
        assert_eq!(store.weight_count(), 0);
        assert_eq!(store.loaded_count(), 0);
        assert_eq!(store.total_loaded_bytes(), 0);
    }

    #[test]
    fn store_default() {
        let store = WeightStore::default();
        assert_eq!(store.weight_count(), 0);
    }

    // =====================================================================
    // GpuWeightBuffer tests
    // =====================================================================

    #[test]
    fn gpu_buffer_new_pending() {
        let buf = GpuWeightBuffer::new("w", 64);
        assert_eq!(buf.status, UploadStatus::Pending);
        assert!(!buf.is_ready());
        assert_eq!(buf.resident_bytes(), 0);
    }

    #[test]
    fn gpu_buffer_upload_ok() {
        let mut buf = GpuWeightBuffer::new("w", 4);
        buf.upload(&[1, 2, 3, 4]).unwrap();
        assert!(buf.is_ready());
        assert_eq!(buf.readback().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(buf.resident_bytes(), 4);
    }

    #[test]
    fn gpu_buffer_upload_size_mismatch() {
        let mut buf = GpuWeightBuffer::new("w", 4);
        let err = buf.upload(&[1, 2]).unwrap_err();
        assert!(matches!(err, WeightError::SizeMismatch { .. }));
    }

    #[test]
    fn gpu_buffer_release() {
        let mut buf = GpuWeightBuffer::new("w", 4);
        buf.upload(&[1, 2, 3, 4]).unwrap();
        buf.release();
        assert!(!buf.is_ready());
        assert!(buf.readback().is_none());
        assert_eq!(buf.resident_bytes(), 0);
    }

    #[test]
    fn upload_status_display() {
        assert_eq!(format!("{}", UploadStatus::Pending), "pending");
        assert_eq!(format!("{}", UploadStatus::Complete), "complete");
    }

    // =====================================================================
    // WeightUploader tests
    // =====================================================================

    #[test]
    fn uploader_upload_single() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[0; 64]).unwrap();
        assert_eq!(up.uploaded_count(), 1);
        assert_eq!(up.gpu_used_bytes(), 64);
    }

    #[test]
    fn uploader_memory_exhausted() {
        let mut up = WeightUploader::new(32, 10.0);
        let err = up.upload_weight("w", &[0; 64]).unwrap_err();
        assert!(matches!(err, WeightError::GpuMemoryExhausted { .. }));
    }

    #[test]
    fn uploader_already_uploaded() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[0; 8]).unwrap();
        let err = up.upload_weight("w", &[0; 8]).unwrap_err();
        assert!(matches!(err, WeightError::AlreadyUploaded(_)));
    }

    #[test]
    fn uploader_release_single() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[0; 64]).unwrap();
        up.release_weight("w").unwrap();
        assert_eq!(up.gpu_used_bytes(), 0);
    }

    #[test]
    fn uploader_release_all() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("a", &[0; 32]).unwrap();
        up.upload_weight("b", &[0; 32]).unwrap();
        up.release_all();
        assert_eq!(up.gpu_used_bytes(), 0);
        assert_eq!(up.uploaded_count(), 0);
    }

    #[test]
    fn uploader_release_not_found() {
        let mut up = WeightUploader::new(1024, 10.0);
        assert!(matches!(up.release_weight("missing"), Err(WeightError::NotFound(_))));
    }

    #[test]
    fn uploader_available_bytes() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[0; 100]).unwrap();
        assert_eq!(up.gpu_available_bytes(), 924);
    }

    #[test]
    fn uploader_upload_all() {
        let mut store = WeightStore::new();
        store.register_with_data(desc("a", vec![1], 4), vec![1; 4]).unwrap();
        store.register_with_data(desc("b", vec![1], 8), vec![2; 8]).unwrap();
        let mut up = WeightUploader::new(1024, 10.0);
        let progress = up.upload_all(&store).unwrap();
        assert_eq!(progress.completed, 2);
        assert_eq!(progress.uploaded_bytes, 12);
        assert!(progress.is_done());
    }

    #[test]
    fn uploader_upload_all_partial_load() {
        let mut store = WeightStore::new();
        store.register(desc("a", vec![1], 4)).unwrap(); // no data
        store.register_with_data(desc("b", vec![1], 8), vec![0; 8]).unwrap();
        let mut up = WeightUploader::new(1024, 10.0);
        let progress = up.upload_all(&store).unwrap();
        assert_eq!(progress.completed, 1);
        // total still counts all descriptors
        assert_eq!(progress.total, 2);
    }

    #[test]
    fn uploader_total_upload_time() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[0; 100]).unwrap();
        assert!(up.total_upload_time_ns() > 0);
    }

    #[test]
    fn uploader_progress_snapshot() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[0; 64]).unwrap();
        let p = up.progress();
        assert_eq!(p.total, 1);
        assert_eq!(p.completed, 1);
        assert!((p.fraction() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn uploader_get_buffer() {
        let mut up = WeightUploader::new(1024, 10.0);
        up.upload_weight("w", &[9; 4]).unwrap();
        let buf = up.get_buffer("w").unwrap();
        assert!(buf.is_ready());
        assert_eq!(buf.readback().unwrap(), &[9; 4]);
    }

    #[test]
    fn upload_progress_fraction_empty() {
        let p = UploadProgress {
            total: 0,
            completed: 0,
            total_bytes: 0,
            uploaded_bytes: 0,
            bandwidth_gbps: 0.0,
        };
        assert!((p.fraction() - 1.0).abs() < f64::EPSILON);
        assert!(p.is_done());
    }

    // =====================================================================
    // WeightSharing tests
    // =====================================================================

    #[test]
    fn sharing_initial_ref_count() {
        let ws = WeightSharing::new(WeightStore::new());
        assert_eq!(ws.ref_count(), 1);
    }

    #[test]
    fn sharing_acquire_increments() {
        let ws = WeightSharing::new(WeightStore::new());
        let ws2 = ws.acquire();
        assert_eq!(ws.ref_count(), 2);
        assert_eq!(ws2.ref_count(), 2);
    }

    #[test]
    fn sharing_release_decrements() {
        let ws = WeightSharing::new(WeightStore::new());
        let ws2 = ws.acquire();
        let remaining = ws2.release();
        assert_eq!(remaining, 1);
        assert_eq!(ws.ref_count(), 1);
    }

    #[test]
    fn sharing_group_id_unique() {
        let ws1 = WeightSharing::new(WeightStore::new());
        let ws2 = WeightSharing::new(WeightStore::new());
        assert_ne!(ws1.group_id(), ws2.group_id());
    }

    #[test]
    fn sharing_clone_increments() {
        let ws = WeightSharing::new(WeightStore::new());
        let _ws2 = ws.clone();
        assert_eq!(ws.ref_count(), 2);
    }

    #[test]
    fn sharing_same_group_id_after_acquire() {
        let ws = WeightSharing::new(WeightStore::new());
        let ws2 = ws.acquire();
        assert_eq!(ws.group_id(), ws2.group_id());
    }

    #[test]
    fn sharing_read_store() {
        let mut store = WeightStore::new();
        store.register(desc("w", vec![1], 4)).unwrap();
        let ws = WeightSharing::new(store);
        let guard = ws.read_store();
        assert_eq!(guard.weight_count(), 1);
    }

    #[test]
    fn sharing_write_store() {
        let ws = WeightSharing::new(WeightStore::new());
        {
            let mut guard = ws.write_store();
            guard.register(desc("w", vec![1], 4)).unwrap();
        }
        assert_eq!(ws.read_store().weight_count(), 1);
    }

    #[test]
    fn sharing_multiple_instances_see_same_data() {
        let mut store = WeightStore::new();
        store.register_with_data(desc("w", vec![1], 4), vec![42; 4]).unwrap();
        let ws1 = WeightSharing::new(store);
        let ws2 = ws1.acquire();
        assert_eq!(
            ws1.read_store().get_data("w").unwrap(),
            ws2.read_store().get_data("w").unwrap()
        );
    }

    // =====================================================================
    // MemoryMappedWeights tests
    // =====================================================================

    #[test]
    fn mmap_basic_access() {
        let backing = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let descs = vec![desc_at("a", vec![2], 0, 4), desc_at("b", vec![2], 4, 4)];
        let mmap = MemoryMappedWeights::new(backing, descs).unwrap();
        assert_eq!(mmap.get("a").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(mmap.get("b").unwrap(), &[5, 6, 7, 8]);
    }

    #[test]
    fn mmap_not_found() {
        let mmap = MemoryMappedWeights::new(vec![0; 4], vec![desc_at("a", vec![1], 0, 4)]).unwrap();
        assert!(matches!(mmap.get("missing"), Err(WeightError::NotFound(_))));
    }

    #[test]
    fn mmap_out_of_bounds() {
        let err =
            MemoryMappedWeights::new(vec![0; 4], vec![desc_at("a", vec![1], 2, 8)]).unwrap_err();
        assert!(matches!(err, WeightError::MmapFailed(_)));
    }

    #[test]
    fn mmap_unmap_prevents_access() {
        let mut mmap =
            MemoryMappedWeights::new(vec![0; 4], vec![desc_at("a", vec![1], 0, 4)]).unwrap();
        assert!(mmap.is_active());
        mmap.unmap();
        assert!(!mmap.is_active());
        assert!(matches!(mmap.get("a"), Err(WeightError::MmapFailed(_))));
    }

    #[test]
    fn mmap_remap_restores_access() {
        let mut mmap =
            MemoryMappedWeights::new(vec![0; 4], vec![desc_at("a", vec![1], 0, 4)]).unwrap();
        mmap.unmap();
        mmap.remap();
        assert!(mmap.get("a").is_ok());
    }

    #[test]
    fn mmap_mapped_size() {
        let mmap = MemoryMappedWeights::new(vec![0; 1024], vec![]).unwrap();
        assert_eq!(mmap.mapped_size(), 1024);
    }

    #[test]
    fn mmap_weight_count() {
        let descs = vec![desc_at("a", vec![1], 0, 2), desc_at("b", vec![1], 2, 2)];
        let mmap = MemoryMappedWeights::new(vec![0; 4], descs).unwrap();
        assert_eq!(mmap.weight_count(), 2);
    }

    #[test]
    fn mmap_get_descriptor() {
        let descs = vec![desc_at("a", vec![3], 0, 12)];
        let mmap = MemoryMappedWeights::new(vec![0; 12], descs).unwrap();
        let d = mmap.get_descriptor("a").unwrap();
        assert_eq!(d.shape, vec![3]);
    }

    #[test]
    fn mmap_empty_region() {
        let mmap = MemoryMappedWeights::new(vec![], vec![]).unwrap();
        assert_eq!(mmap.mapped_size(), 0);
        assert_eq!(mmap.weight_count(), 0);
    }

    // =====================================================================
    // WeightPrefetcher tests
    // =====================================================================

    #[test]
    fn prefetcher_requests_next_layers() {
        let descs = vec![layer_desc(0, "wq", 16), layer_desc(1, "wq", 16), layer_desc(2, "wq", 16)];
        let mut pf = WeightPrefetcher::new(&descs, 2);
        let reqs = pf.prefetch_for(0);
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].layer_index, 1);
        assert_eq!(reqs[1].layer_index, 2);
    }

    #[test]
    fn prefetcher_no_duplicate_requests() {
        let descs = vec![layer_desc(0, "wq", 16), layer_desc(1, "wq", 16)];
        let mut pf = WeightPrefetcher::new(&descs, 2);
        let reqs1 = pf.prefetch_for(0);
        assert_eq!(reqs1.len(), 1); // only layer 1
        let reqs2 = pf.prefetch_for(0);
        assert!(reqs2.is_empty()); // already prefetched
    }

    #[test]
    fn prefetcher_reset() {
        let descs = vec![layer_desc(0, "wq", 16), layer_desc(1, "wq", 16)];
        let mut pf = WeightPrefetcher::new(&descs, 2);
        pf.prefetch_for(0);
        pf.reset();
        let reqs = pf.prefetch_for(0);
        assert_eq!(reqs.len(), 1); // re-prefetchable after reset
    }

    #[test]
    fn prefetcher_beyond_last_layer() {
        let descs = vec![layer_desc(0, "wq", 16)];
        let mut pf = WeightPrefetcher::new(&descs, 5);
        let reqs = pf.prefetch_for(0);
        assert!(reqs.is_empty()); // no layer 1+
    }

    #[test]
    fn prefetcher_total_layers() {
        let descs = vec![layer_desc(0, "wq", 16), layer_desc(3, "wq", 16)];
        let pf = WeightPrefetcher::new(&descs, 1);
        assert_eq!(pf.total_layers(), 4); // 0..=3
    }

    #[test]
    fn prefetcher_depth() {
        let pf = WeightPrefetcher::new(&[], 3);
        assert_eq!(pf.depth(), 3);
    }

    #[test]
    fn prefetcher_weights_for_layer() {
        let descs = vec![layer_desc(0, "wq", 16), layer_desc(0, "wk", 16)];
        let pf = WeightPrefetcher::new(&descs, 1);
        let names = pf.weights_for_layer(0).unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn prefetcher_empty_model() {
        let pf = WeightPrefetcher::new(&[], 2);
        assert_eq!(pf.total_layers(), 0);
    }

    #[test]
    fn prefetcher_non_layer_descriptors_ignored() {
        let descs = vec![desc("embedding.weight", vec![1], 16), layer_desc(0, "wq", 16)];
        let pf = WeightPrefetcher::new(&descs, 2);
        assert_eq!(pf.total_layers(), 1);
        assert!(pf.weights_for_layer(0).is_some());
    }

    // =====================================================================
    // WeightStats tests
    // =====================================================================

    #[test]
    fn stats_gather() {
        let mut store = WeightStore::new();
        store.register_with_data(desc("a", vec![1], 64), sample_data(64)).unwrap();
        store.register_with_data(desc("b", vec![1], 128), sample_data(128)).unwrap();
        let mut up = WeightUploader::new(4096, 10.0);
        up.upload_weight("a", &sample_data(64)).unwrap();
        let stats = WeightStats::gather(&store, &up, 3);

        assert_eq!(stats.total_size, 192);
        assert_eq!(stats.gpu_resident, 64);
        assert_eq!(stats.share_count, 3);
        assert_eq!(stats.weight_count, 2);
        assert_eq!(stats.gpu_weight_count, 1);
        assert_eq!(stats.host_loaded, 192);
    }

    #[test]
    fn stats_gpu_fraction_empty() {
        let stats = WeightStats::default();
        assert!((stats.gpu_fraction()).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_gpu_fraction_half() {
        let stats = WeightStats { total_size: 200, gpu_resident: 100, ..Default::default() };
        assert!((stats.gpu_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_display() {
        let stats = WeightStats {
            weight_count: 2,
            total_size: 1024,
            gpu_resident: 512,
            upload_bandwidth_gbps: 10.0,
            share_count: 1,
            ..Default::default()
        };
        let s = format!("{stats}");
        assert!(s.contains("weights=2"));
        assert!(s.contains("1024"));
    }

    // =====================================================================
    // WeightError tests
    // =====================================================================

    #[test]
    fn error_display_variants() {
        let e1 = WeightError::InvalidDescriptor("bad".into());
        assert!(format!("{e1}").contains("bad"));

        let e2 = WeightError::NotFound("w".into());
        assert!(format!("{e2}").contains("w"));

        let e3 = WeightError::SizeMismatch { expected: 8, got: 4 };
        assert!(format!("{e3}").contains("8"));

        let e4 = WeightError::GpuMemoryExhausted { requested: 100, available: 50 };
        assert!(format!("{e4}").contains("100"));

        let e5 = WeightError::AlreadyUploaded("w".into());
        assert!(format!("{e5}").contains("w"));

        let e6 = WeightError::MmapFailed("oops".into());
        assert!(format!("{e6}").contains("oops"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(WeightError::NotFound("w".into()));
        assert!(format!("{e}").contains("w"));
    }

    // =====================================================================
    // Property-style: loaded data preserves exact bytes
    // =====================================================================

    #[test]
    fn roundtrip_store_preserves_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let mut store = WeightStore::new();
        store.register_with_data(desc("w", vec![256], 256), data.clone()).unwrap();
        assert_eq!(store.get_data("w").unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_gpu_preserves_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let mut buf = GpuWeightBuffer::new("w", 256);
        buf.upload(&data).unwrap();
        assert_eq!(buf.readback().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_mmap_preserves_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let descs = vec![desc_at("w", vec![256], 0, 256)];
        let mmap = MemoryMappedWeights::new(data.clone(), descs).unwrap();
        assert_eq!(mmap.get("w").unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_upload_preserves_bytes() {
        let data: Vec<u8> = (0..128).collect();
        let mut up = WeightUploader::new(4096, 10.0);
        up.upload_weight("w", &data).unwrap();
        let buf = up.get_buffer("w").unwrap();
        assert_eq!(buf.readback().unwrap(), &data[..]);
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn single_weight_model() {
        let mut store = WeightStore::new();
        store.register_with_data(desc("only", vec![1], 1), vec![0xFF]).unwrap();
        let mut up = WeightUploader::new(1, 1.0);
        up.upload_weight("only", &[0xFF]).unwrap();
        assert_eq!(up.uploaded_count(), 1);
        assert_eq!(up.gpu_used_bytes(), 1);
    }

    #[test]
    fn large_model_budget_tracking() {
        let budget = 16 * 1024 * 1024 * 1024_usize; // 16 GiB
        let mut up = WeightUploader::new(budget, 50.0);
        let big = vec![0u8; 1024 * 1024]; // 1 MiB
        for i in 0..100 {
            up.upload_weight(&format!("w{i}"), &big).unwrap();
        }
        assert_eq!(up.gpu_used_bytes(), 100 * 1024 * 1024);
        assert_eq!(up.gpu_available_bytes(), budget - 100 * 1024 * 1024);
    }

    #[test]
    fn empty_model_stats() {
        let store = WeightStore::new();
        let up = WeightUploader::new(1024, 10.0);
        let stats = WeightStats::gather(&store, &up, 0);
        assert_eq!(stats.total_size, 0);
        assert_eq!(stats.weight_count, 0);
    }

    #[test]
    fn multiple_dtype_weights() {
        let mut store = WeightStore::new();
        store.register(WeightDescriptor::new("f32_w", vec![4], WeightDtype::F32, 0, 16)).unwrap();
        store.register(WeightDescriptor::new("f16_w", vec![4], WeightDtype::F16, 0, 8)).unwrap();
        store.register(WeightDescriptor::new("i2s_w", vec![32], WeightDtype::I2S, 0, 8)).unwrap();
        assert_eq!(store.weight_count(), 3);
    }

    #[test]
    fn descriptor_various_dtypes() {
        for dtype in [
            WeightDtype::F32,
            WeightDtype::F16,
            WeightDtype::BF16,
            WeightDtype::I8,
            WeightDtype::I2S,
            WeightDtype::QK256,
        ] {
            let d = WeightDescriptor::new("w", vec![1], dtype, 0, 4);
            assert_eq!(d.dtype, dtype);
        }
    }

    #[test]
    fn uploader_zero_bandwidth() {
        let mut up = WeightUploader::new(1024, 0.0);
        up.upload_weight("w", &[0; 8]).unwrap();
        assert_eq!(up.total_upload_time_ns(), 0);
    }

    #[test]
    fn mmap_adjacent_weights_no_overlap() {
        let backing = vec![0xAA; 16];
        let descs = vec![desc_at("a", vec![4], 0, 8), desc_at("b", vec![4], 8, 8)];
        let mmap = MemoryMappedWeights::new(backing, descs).unwrap();
        assert_eq!(mmap.get("a").unwrap().len(), 8);
        assert_eq!(mmap.get("b").unwrap().len(), 8);
    }
}
