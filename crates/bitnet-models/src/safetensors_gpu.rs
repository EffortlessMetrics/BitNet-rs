//! SafeTensors direct-to-GPU loading.
//!
//! Parses SafeTensors headers to locate tensor data offsets, then uses
//! memory-mapped I/O + DMA to transfer weights directly to GPU memory,
//! skipping intermediate host buffers when possible.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Tensor metadata types
// ---------------------------------------------------------------------------

/// Data type of a tensor stored in a SafeTensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDtype {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl TensorDtype {
    /// Size in bytes of a single element.
    pub fn element_size(self) -> usize {
        match self {
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I8 | Self::U8 => 1,
        }
    }

    fn from_str_repr(s: &str) -> Option<Self> {
        match s {
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "I8" => Some(Self::I8),
            "I16" => Some(Self::I16),
            "I32" => Some(Self::I32),
            "I64" => Some(Self::I64),
            "U8" => Some(Self::U8),
            "U16" => Some(Self::U16),
            "U32" => Some(Self::U32),
            "U64" => Some(Self::U64),
            _ => None,
        }
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
        };
        f.write_str(s)
    }
}

/// Metadata for a single tensor inside a SafeTensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Element data type.
    pub dtype: TensorDtype,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Byte offset from the start of the data section.
    pub data_offset_start: usize,
    /// Byte offset of the end of this tensor's data (exclusive).
    pub data_offset_end: usize,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn byte_size(&self) -> usize {
        self.data_offset_end - self.data_offset_start
    }
}

// ---------------------------------------------------------------------------
// SafeTensors header parsing
// ---------------------------------------------------------------------------

/// Parsed header of a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    /// Size (in bytes) of the JSON header.
    pub header_size: u64,
    /// Tensor metadata keyed by name.
    pub tensors: HashMap<String, TensorInfo>,
    /// Offset where the data section begins in the file.
    pub data_offset: u64,
}

/// Errors that can occur during SafeTensors GPU loading.
#[derive(Debug)]
pub enum SafeTensorsGpuError {
    /// I/O error reading the file.
    Io(io::Error),
    /// The header JSON is malformed.
    InvalidHeader(String),
    /// A referenced tensor was not found.
    TensorNotFound(String),
    /// Memory-map is unavailable; caller should use fallback.
    MmapUnavailable(String),
    /// GPU transfer failed.
    GpuTransferFailed(String),
    /// Unsupported dtype conversion requested.
    UnsupportedConversion {
        from: TensorDtype,
        to: TensorDtype,
    },
}

impl fmt::Display for SafeTensorsGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidHeader(msg) => write!(f, "invalid SafeTensors header: {msg}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::MmapUnavailable(msg) => write!(f, "mmap unavailable: {msg}"),
            Self::GpuTransferFailed(msg) => write!(f, "GPU transfer failed: {msg}"),
            Self::UnsupportedConversion { from, to } => {
                write!(f, "unsupported conversion: {from} -> {to}")
            }
        }
    }
}

impl std::error::Error for SafeTensorsGpuError {}

impl From<io::Error> for SafeTensorsGpuError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Parse the SafeTensors header from a reader.
///
/// The SafeTensors format starts with an 8-byte little-endian integer giving
/// the header length, followed by a JSON object whose keys are tensor names.
pub fn parse_header<R: Read + Seek>(reader: &mut R) -> Result<SafeTensorsHeader, SafeTensorsGpuError> {
    // Read header length (first 8 bytes, LE u64).
    let mut len_buf = [0u8; 8];
    reader
        .read_exact(&mut len_buf)
        .map_err(|e| SafeTensorsGpuError::InvalidHeader(format!("cannot read header length: {e}")))?;
    let header_size = u64::from_le_bytes(len_buf);

    if header_size == 0 || header_size > 100_000_000 {
        return Err(SafeTensorsGpuError::InvalidHeader(format!(
            "header size {header_size} out of reasonable range"
        )));
    }

    // Read header JSON.
    let mut header_buf = vec![0u8; header_size as usize];
    reader.read_exact(&mut header_buf)?;

    let header_json: serde_json::Value = serde_json::from_slice(&header_buf).map_err(|e| {
        SafeTensorsGpuError::InvalidHeader(format!("JSON parse error: {e}"))
    })?;

    let obj = header_json.as_object().ok_or_else(|| {
        SafeTensorsGpuError::InvalidHeader("header is not a JSON object".into())
    })?;

    let data_offset = 8 + header_size;
    let mut tensors = HashMap::new();

    for (key, value) in obj {
        // Skip the `__metadata__` key.
        if key == "__metadata__" {
            continue;
        }

        let tensor_obj = value.as_object().ok_or_else(|| {
            SafeTensorsGpuError::InvalidHeader(format!("tensor entry '{key}' is not an object"))
        })?;

        let dtype_str = tensor_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                SafeTensorsGpuError::InvalidHeader(format!("missing dtype for '{key}'"))
            })?;
        let dtype = TensorDtype::from_str_repr(dtype_str).ok_or_else(|| {
            SafeTensorsGpuError::InvalidHeader(format!("unknown dtype '{dtype_str}' for '{key}'"))
        })?;

        let shape: Vec<usize> = tensor_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SafeTensorsGpuError::InvalidHeader(format!("missing shape for '{key}'"))
            })?
            .iter()
            .map(|v| {
                v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    SafeTensorsGpuError::InvalidHeader(format!("invalid shape element for '{key}'"))
                })
            })
            .collect::<Result<_, _>>()?;

        let offsets = tensor_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                SafeTensorsGpuError::InvalidHeader(format!("missing data_offsets for '{key}'"))
            })?;
        if offsets.len() != 2 {
            return Err(SafeTensorsGpuError::InvalidHeader(format!(
                "data_offsets for '{key}' must have exactly 2 elements"
            )));
        }
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;

        tensors.insert(
            key.clone(),
            TensorInfo {
                name: key.clone(),
                dtype,
                shape,
                data_offset_start: start,
                data_offset_end: end,
            },
        );
    }

    Ok(SafeTensorsHeader {
        header_size,
        tensors,
        data_offset,
    })
}

// ---------------------------------------------------------------------------
// GPU buffer abstraction
// ---------------------------------------------------------------------------

/// Opaque handle representing a GPU buffer allocation.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Unique id for this allocation.
    pub id: u64,
    /// Size in bytes.
    pub size: usize,
    /// Target dtype stored in this buffer.
    pub dtype: TensorDtype,
}

/// Trait abstracting GPU memory operations so that the loader is testable
/// without real hardware.
pub trait GpuTransfer: Send + Sync {
    /// Allocate a buffer of `size` bytes on the GPU.
    fn allocate(&self, size: usize, dtype: TensorDtype) -> Result<GpuBuffer, SafeTensorsGpuError>;

    /// Copy `data` from host into the given GPU buffer.
    fn copy_host_to_device(
        &self,
        buffer: &GpuBuffer,
        data: &[u8],
    ) -> Result<(), SafeTensorsGpuError>;

    /// Perform on-GPU dtype conversion (e.g. BF16→F32).
    fn convert_dtype(
        &self,
        buffer: &GpuBuffer,
        from: TensorDtype,
        to: TensorDtype,
        num_elements: usize,
    ) -> Result<GpuBuffer, SafeTensorsGpuError>;

    /// Return the amount of free GPU memory in bytes.
    fn free_memory(&self) -> usize;

    /// Whether memory-mapped DMA transfer is supported.
    fn supports_mmap_dma(&self) -> bool;

    /// DMA transfer directly from memory-mapped region to GPU.
    fn dma_transfer(
        &self,
        buffer: &GpuBuffer,
        mmap_ptr: *const u8,
        len: usize,
    ) -> Result<(), SafeTensorsGpuError>;
}

// ---------------------------------------------------------------------------
// Progress callback
// ---------------------------------------------------------------------------

/// Progress report sent during model loading.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    /// Name of the tensor currently being loaded.
    pub tensor_name: String,
    /// 0-based index of the current tensor.
    pub current: usize,
    /// Total number of tensors to load.
    pub total: usize,
    /// Bytes loaded so far (cumulative).
    pub bytes_loaded: u64,
    /// Total bytes to load.
    pub bytes_total: u64,
}

/// Type-erased progress callback.
pub type ProgressCallback = Arc<dyn Fn(&LoadProgress) + Send + Sync>;

// ---------------------------------------------------------------------------
// SafeTensorsGpuLoader
// ---------------------------------------------------------------------------

/// Loads SafeTensors weights directly to GPU memory.
pub struct SafeTensorsGpuLoader<G: GpuTransfer> {
    gpu: Arc<G>,
    target_dtype: Option<TensorDtype>,
    progress_cb: Option<ProgressCallback>,
}

impl<G: GpuTransfer> SafeTensorsGpuLoader<G> {
    /// Create a new loader with the given GPU transfer backend.
    pub fn new(gpu: Arc<G>) -> Self {
        Self {
            gpu,
            target_dtype: None,
            progress_cb: None,
        }
    }

    /// If set, tensors will be converted to this dtype on the GPU after upload.
    pub fn with_target_dtype(mut self, dtype: TensorDtype) -> Self {
        self.target_dtype = Some(dtype);
        self
    }

    /// Register a progress callback.
    pub fn with_progress(mut self, cb: ProgressCallback) -> Self {
        self.progress_cb = Some(cb);
        self
    }

    /// Load all tensors from a SafeTensors file, returning a map of
    /// tensor-name → GPU buffer.
    pub fn load_file(
        &self,
        path: &Path,
    ) -> Result<HashMap<String, GpuBuffer>, SafeTensorsGpuError> {
        let mut file = std::fs::File::open(path)?;
        let header = parse_header(&mut file)?;

        let total = header.tensors.len();
        let bytes_total: u64 = header.tensors.values().map(|t| t.byte_size() as u64).sum();
        let mut bytes_loaded: u64 = 0;
        let mut result = HashMap::with_capacity(total);

        // Try mmap path first.
        let mmap_result = self.try_mmap_load(path, &header);

        match mmap_result {
            Ok(buffers) => return Ok(buffers),
            Err(SafeTensorsGpuError::MmapUnavailable(_)) => {
                // Fall through to host-staged path.
            }
            Err(e) => return Err(e),
        }

        // Host-staged fallback: read each tensor into a host buffer, then upload.
        for (idx, (name, info)) in header.tensors.iter().enumerate() {
            let buf = self.gpu.allocate(info.byte_size(), info.dtype)?;

            // Seek to tensor data.
            file.seek(SeekFrom::Start(header.data_offset + info.data_offset_start as u64))?;
            let mut host_buf = vec![0u8; info.byte_size()];
            file.read_exact(&mut host_buf)?;

            self.gpu.copy_host_to_device(&buf, &host_buf)?;

            let final_buf = self.maybe_convert(&buf, info)?;

            bytes_loaded += info.byte_size() as u64;
            if let Some(ref cb) = self.progress_cb {
                cb(&LoadProgress {
                    tensor_name: name.clone(),
                    current: idx,
                    total,
                    bytes_loaded,
                    bytes_total,
                });
            }

            result.insert(name.clone(), final_buf);
        }

        Ok(result)
    }

    /// Load a single tensor by name using the host-staged path.
    pub fn load_tensor(
        &self,
        path: &Path,
        tensor_name: &str,
    ) -> Result<GpuBuffer, SafeTensorsGpuError> {
        let mut file = std::fs::File::open(path)?;
        let header = parse_header(&mut file)?;

        let info = header
            .tensors
            .get(tensor_name)
            .ok_or_else(|| SafeTensorsGpuError::TensorNotFound(tensor_name.to_string()))?;

        let buf = self.gpu.allocate(info.byte_size(), info.dtype)?;

        file.seek(SeekFrom::Start(header.data_offset + info.data_offset_start as u64))?;
        let mut host_buf = vec![0u8; info.byte_size()];
        file.read_exact(&mut host_buf)?;

        self.gpu.copy_host_to_device(&buf, &host_buf)?;
        self.maybe_convert(&buf, info)
    }

    /// Attempt to load all tensors via memory-mapped DMA.
    fn try_mmap_load(
        &self,
        path: &Path,
        header: &SafeTensorsHeader,
    ) -> Result<HashMap<String, GpuBuffer>, SafeTensorsGpuError> {
        if !self.gpu.supports_mmap_dma() {
            return Err(SafeTensorsGpuError::MmapUnavailable(
                "GPU backend does not support DMA".into(),
            ));
        }

        let file = std::fs::File::open(path)?;
        let mmap = unsafe {
            memmap2::MmapOptions::new().map(&file).map_err(|e| {
                SafeTensorsGpuError::MmapUnavailable(format!("mmap failed: {e}"))
            })?
        };

        let total = header.tensors.len();
        let bytes_total: u64 = header.tensors.values().map(|t| t.byte_size() as u64).sum();
        let mut bytes_loaded: u64 = 0;
        let mut result = HashMap::with_capacity(total);

        for (idx, (name, info)) in header.tensors.iter().enumerate() {
            let buf = self.gpu.allocate(info.byte_size(), info.dtype)?;

            let abs_start = header.data_offset as usize + info.data_offset_start;
            let abs_end = header.data_offset as usize + info.data_offset_end;

            if abs_end > mmap.len() {
                return Err(SafeTensorsGpuError::GpuTransferFailed(format!(
                    "tensor '{name}' data extends beyond file ({abs_end} > {})",
                    mmap.len()
                )));
            }

            let ptr = mmap[abs_start..abs_end].as_ptr();
            self.gpu.dma_transfer(&buf, ptr, info.byte_size())?;

            let final_buf = self.maybe_convert(&buf, info)?;

            bytes_loaded += info.byte_size() as u64;
            if let Some(ref cb) = self.progress_cb {
                cb(&LoadProgress {
                    tensor_name: name.clone(),
                    current: idx,
                    total,
                    bytes_loaded,
                    bytes_total,
                });
            }

            result.insert(name.clone(), final_buf);
        }

        Ok(result)
    }

    /// Convert tensor dtype on GPU if a target dtype is configured.
    fn maybe_convert(
        &self,
        buf: &GpuBuffer,
        info: &TensorInfo,
    ) -> Result<GpuBuffer, SafeTensorsGpuError> {
        if let Some(target) = self.target_dtype {
            if target != info.dtype {
                return self
                    .gpu
                    .convert_dtype(buf, info.dtype, target, info.num_elements());
            }
        }
        Ok(buf.clone())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    /// Mock GPU backend for testing.
    struct MockGpu {
        next_id: AtomicU64,
        supports_dma: bool,
        free_mem: usize,
        /// Track all allocations for assertions.
        allocations: Mutex<Vec<(u64, usize, TensorDtype)>>,
        /// Track DMA calls.
        dma_calls: Mutex<Vec<(u64, usize)>>,
    }

    impl MockGpu {
        fn new(supports_dma: bool, free_mem: usize) -> Self {
            Self {
                next_id: AtomicU64::new(1),
                supports_dma,
                free_mem,
                allocations: Mutex::new(Vec::new()),
                dma_calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl GpuTransfer for MockGpu {
        fn allocate(&self, size: usize, dtype: TensorDtype) -> Result<GpuBuffer, SafeTensorsGpuError> {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.allocations.lock().unwrap().push((id, size, dtype));
            Ok(GpuBuffer { id, size, dtype })
        }

        fn copy_host_to_device(
            &self,
            _buffer: &GpuBuffer,
            _data: &[u8],
        ) -> Result<(), SafeTensorsGpuError> {
            Ok(())
        }

        fn convert_dtype(
            &self,
            _buffer: &GpuBuffer,
            _from: TensorDtype,
            to: TensorDtype,
            num_elements: usize,
        ) -> Result<GpuBuffer, SafeTensorsGpuError> {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let size = num_elements * to.element_size();
            Ok(GpuBuffer { id, size, dtype: to })
        }

        fn free_memory(&self) -> usize {
            self.free_mem
        }

        fn supports_mmap_dma(&self) -> bool {
            self.supports_dma
        }

        fn dma_transfer(
            &self,
            buffer: &GpuBuffer,
            _mmap_ptr: *const u8,
            len: usize,
        ) -> Result<(), SafeTensorsGpuError> {
            self.dma_calls.lock().unwrap().push((buffer.id, len));
            Ok(())
        }
    }

    /// Build a minimal SafeTensors byte buffer for testing.
    fn build_safetensors(tensors: &[(&str, TensorDtype, &[usize])]) -> Vec<u8> {
        use serde_json::json;
        let mut header_obj = serde_json::Map::new();
        let mut offset: usize = 0;
        let mut data_blobs: Vec<Vec<u8>> = Vec::new();

        for &(name, dtype, shape) in tensors {
            let num_elements: usize = shape.iter().product();
            let byte_size = num_elements * dtype.element_size();
            let start = offset;
            let end = offset + byte_size;
            offset = end;
            data_blobs.push(vec![0xAB; byte_size]);

            header_obj.insert(
                name.to_string(),
                json!({
                    "dtype": format!("{dtype}"),
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json = serde_json::to_vec(&serde_json::Value::Object(header_obj)).unwrap();
        let header_len = header_json.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_json);
        for blob in &data_blobs {
            buf.extend_from_slice(blob);
        }
        buf
    }

    // ---- Tests ----

    #[test]
    fn test_parse_header_single_tensor() {
        let data = build_safetensors(&[("weight", TensorDtype::F32, &[4, 4])]);
        let mut cursor = Cursor::new(&data);
        let header = parse_header(&mut cursor).unwrap();
        assert_eq!(header.tensors.len(), 1);
        let t = &header.tensors["weight"];
        assert_eq!(t.dtype, TensorDtype::F32);
        assert_eq!(t.shape, vec![4, 4]);
        assert_eq!(t.byte_size(), 64); // 16 elements × 4 bytes
    }

    #[test]
    fn test_parse_header_multiple_tensors() {
        let data = build_safetensors(&[
            ("a", TensorDtype::F16, &[2, 3]),
            ("b", TensorDtype::BF16, &[10]),
            ("c", TensorDtype::I8, &[100]),
        ]);
        let mut cursor = Cursor::new(&data);
        let header = parse_header(&mut cursor).unwrap();
        assert_eq!(header.tensors.len(), 3);
        assert_eq!(header.tensors["a"].dtype, TensorDtype::F16);
        assert_eq!(header.tensors["b"].dtype, TensorDtype::BF16);
        assert_eq!(header.tensors["c"].num_elements(), 100);
    }

    #[test]
    fn test_parse_header_invalid_json() {
        let bad_header = b"not json at all";
        let header_len = (bad_header.len() as u64).to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&header_len);
        data.extend_from_slice(bad_header);

        let mut cursor = Cursor::new(&data);
        let result = parse_header(&mut cursor);
        assert!(matches!(result, Err(SafeTensorsGpuError::InvalidHeader(_))));
    }

    #[test]
    fn test_parse_header_zero_length() {
        let data = 0u64.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        let result = parse_header(&mut cursor);
        assert!(matches!(result, Err(SafeTensorsGpuError::InvalidHeader(_))));
    }

    #[test]
    fn test_load_file_host_staged_fallback() {
        let st_data = build_safetensors(&[
            ("w1", TensorDtype::F32, &[2, 2]),
            ("w2", TensorDtype::F16, &[4]),
        ]);

        let dir = std::env::temp_dir().join("bitnet_st_gpu_test_fallback");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &st_data).unwrap();

        let gpu = Arc::new(MockGpu::new(false, 1 << 30));
        let loader = SafeTensorsGpuLoader::new(gpu.clone());
        let buffers = loader.load_file(&path).unwrap();

        assert_eq!(buffers.len(), 2);
        assert!(buffers.contains_key("w1"));
        assert!(buffers.contains_key("w2"));

        // Verify allocations happened.
        let allocs = gpu.allocations.lock().unwrap();
        assert_eq!(allocs.len(), 2);

        // No DMA calls because mmap not supported.
        let dma = gpu.dma_calls.lock().unwrap();
        assert!(dma.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_file_mmap_dma_path() {
        let st_data = build_safetensors(&[("w", TensorDtype::F32, &[8])]);

        let dir = std::env::temp_dir().join("bitnet_st_gpu_test_dma");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &st_data).unwrap();

        let gpu = Arc::new(MockGpu::new(true, 1 << 30));
        let loader = SafeTensorsGpuLoader::new(gpu.clone());
        let buffers = loader.load_file(&path).unwrap();

        assert_eq!(buffers.len(), 1);

        // DMA should have been used.
        let dma = gpu.dma_calls.lock().unwrap();
        assert_eq!(dma.len(), 1);
        assert_eq!(dma[0].1, 32); // 8 × 4 bytes

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_file_with_dtype_conversion() {
        let st_data = build_safetensors(&[("w", TensorDtype::BF16, &[4])]);

        let dir = std::env::temp_dir().join("bitnet_st_gpu_test_convert");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &st_data).unwrap();

        let gpu = Arc::new(MockGpu::new(false, 1 << 30));
        let loader = SafeTensorsGpuLoader::new(gpu.clone()).with_target_dtype(TensorDtype::F32);
        let buffers = loader.load_file(&path).unwrap();

        // The buffer should be F32 after conversion.
        let buf = &buffers["w"];
        assert_eq!(buf.dtype, TensorDtype::F32);
        assert_eq!(buf.size, 16); // 4 elements × 4 bytes

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_progress_callback_invoked() {
        let st_data = build_safetensors(&[
            ("a", TensorDtype::F32, &[2]),
            ("b", TensorDtype::F32, &[3]),
        ]);

        let dir = std::env::temp_dir().join("bitnet_st_gpu_test_progress");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &st_data).unwrap();

        let progress_reports: Arc<Mutex<Vec<LoadProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let reports_clone = progress_reports.clone();

        let gpu = Arc::new(MockGpu::new(false, 1 << 30));
        let loader = SafeTensorsGpuLoader::new(gpu)
            .with_progress(Arc::new(move |p| {
                reports_clone.lock().unwrap().push(p.clone());
            }));

        loader.load_file(&path).unwrap();

        let reports = progress_reports.lock().unwrap();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].total, 2);
        assert_eq!(reports[1].total, 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_tensor_not_found() {
        let st_data = build_safetensors(&[("w", TensorDtype::F32, &[4])]);

        let dir = std::env::temp_dir().join("bitnet_st_gpu_test_notfound");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &st_data).unwrap();

        let gpu = Arc::new(MockGpu::new(false, 1 << 30));
        let loader = SafeTensorsGpuLoader::new(gpu);
        let result = loader.load_tensor(&path, "nonexistent");

        assert!(matches!(result, Err(SafeTensorsGpuError::TensorNotFound(_))));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_tensor_dtype_element_sizes() {
        assert_eq!(TensorDtype::F16.element_size(), 2);
        assert_eq!(TensorDtype::BF16.element_size(), 2);
        assert_eq!(TensorDtype::F32.element_size(), 4);
        assert_eq!(TensorDtype::F64.element_size(), 8);
        assert_eq!(TensorDtype::I8.element_size(), 1);
        assert_eq!(TensorDtype::U8.element_size(), 1);
        assert_eq!(TensorDtype::I32.element_size(), 4);
        assert_eq!(TensorDtype::U64.element_size(), 8);
    }
}
