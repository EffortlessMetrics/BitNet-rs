//! Persistent CUDA BitNet context and weight-handle scaffolding.
//!
//! This module owns CUDA lifetime state for the BitNet-specific CUDA path:
//! device identity, one long-lived CUDA context/stream pair, upload-once weight
//! handles, reusable activation workspace metadata, and receipt-friendly stats.
//! It does not route transformer inference or launch BitNet kernels.

use bitnet_common::{KernelError, Result};
#[cfg(feature = "cuda")]
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use super::qk256_gemv::{CUDA_QK256_GEMV_KERNEL_ID, CUDA_QK256_GEMV_KERNEL_SRC, Qk256GemvConfig};
use super::qk256_gemv::{QK256_BLOCK_COLS, QK256_PACKED_BYTES_PER_BLOCK};
#[cfg(feature = "cuda")]
use super::quantized_matmul::{I2S_MATMUL_KERNEL_SRC, I2sMatmulConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    result::device as cu_device,
    sys::{self, CUdevice_attribute},
};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{Ptx, compile_ptx};

static NEXT_WEIGHT_ID: AtomicU64 = AtomicU64::new(1);
#[cfg(feature = "cuda")]
static NVRTC_COMPILE_LOCK: Mutex<()> = Mutex::new(());

/// Kernel ID recorded by the reusable CUDA I2_S GEMV primitive.
pub const CUDA_BITNET_I2S_GEMV_KERNEL_ID: &str = "i2s_gemv_cuda";
#[cfg(feature = "cuda")]
const I2S_MATMUL_FUNCTION_NAME: &str = "i2s_matmul_f32";

/// CUDA device identity recorded by the persistent BitNet context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBitnetDeviceInfo {
    /// Zero-based CUDA device index.
    pub device_index: usize,
    /// CUDA-reported device name.
    pub device_name: String,
    /// CUDA compute capability as `(major, minor)`.
    pub compute_capability: (u32, u32),
    /// Total VRAM reported by CUDA, when available.
    pub vram_bytes: Option<u64>,
}

impl CudaBitnetDeviceInfo {
    /// Return compute capability in receipt form, for example `12.0`.
    pub fn compute_capability_string(&self) -> String {
        format!("{}.{}", self.compute_capability.0, self.compute_capability.1)
    }
}

/// BitNet CUDA kernel family used by an uploaded weight handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaBitnetKernelFamily {
    /// I2_S packed BitNet weights.
    I2s,
    /// QK256 packed BitNet weights.
    Qk256,
}

impl CudaBitnetKernelFamily {
    /// Stable receipt label for this kernel family.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::I2s => "i2_s",
            Self::Qk256 => "qk256",
        }
    }
}

/// Logical tensor shape for a CUDA-resident BitNet weight.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaTensorShape {
    /// Tensor dimensions in canonical row-major order.
    pub dims: Vec<usize>,
}

impl CudaTensorShape {
    /// Construct a shape from explicit dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(invalid_arguments("CUDA tensor shape must have at least one dimension"));
        }
        if dims.contains(&0) {
            return Err(invalid_arguments("CUDA tensor shape dimensions must be non-zero"));
        }
        Ok(Self { dims })
    }

    /// Construct a two-dimensional matrix shape.
    pub fn matrix(rows: usize, cols: usize) -> Result<Self> {
        Self::new(vec![rows, cols])
    }

    /// Number of logical tensor elements.
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Stable identifier for a CUDA weight handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaWeightId(u64);

impl CudaWeightId {
    /// Return the numeric handle id for receipts and debug output.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Cloneable metadata handle for a CUDA-resident BitNet weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaWeightHandle {
    /// Stable handle id.
    pub id: CudaWeightId,
    /// Original tensor name.
    pub tensor_name: String,
    /// Logical tensor shape.
    pub shape: CudaTensorShape,
    /// CUDA BitNet kernel family this handle is prepared for.
    pub kernel_family: CudaBitnetKernelFamily,
    /// Packed weight payload size.
    pub packed_bytes: usize,
    /// Scale or side-table payload size.
    pub scale_bytes: usize,
    /// Quantization block size for packed linear kernels, when applicable.
    pub block_size: Option<usize>,
    /// True when the weight was packed during strict model load before upload.
    pub packed_at_load: bool,
    /// True when this handle was uploaded or registered once at context lifetime.
    pub uploaded_once: bool,
}

impl CudaWeightHandle {
    /// Total device-resident bytes represented by this handle.
    pub const fn total_bytes(&self) -> usize {
        self.packed_bytes + self.scale_bytes
    }
}

/// Reusable activation/output/scratch workspace for CUDA decode buffers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaActivationWorkspace {
    /// Current activation buffer capacity in bytes.
    pub activation_bytes: usize,
    /// Current output buffer capacity in bytes.
    pub output_bytes: usize,
    /// Current scratch buffer capacity in bytes.
    pub scratch_bytes: usize,
    /// Number of times the workspace grew.
    pub growth_count: u64,
    /// Number of times an existing allocation satisfied a request.
    pub reuse_count: u64,
}

impl CudaActivationWorkspace {
    /// Total workspace capacity in bytes.
    pub const fn total_bytes(&self) -> usize {
        self.activation_bytes + self.output_bytes + self.scratch_bytes
    }
}

/// Runtime stats used by future CUDA BitNet receipts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaBitnetRuntimeStats {
    /// Number of distinct upload-once weight handles.
    pub weight_uploads: u64,
    /// Bytes uploaded for CUDA weight handles.
    pub weight_upload_bytes: u64,
    /// Number of decode-time/per-token weight upload attempts.
    pub per_token_weight_uploads: u64,
    /// Number of activation workspace growth events.
    pub workspace_growths: u64,
    /// Number of activation workspace reuse events.
    pub workspace_reuses: u64,
}

/// Packed I2_S weights for one BitNet linear projection.
///
/// The logical weight shape is `[output_features, input_features]`. Packed
/// bytes are grouped per output feature with `ceil(input_features / 4)` bytes
/// per row, matching the existing I2_S CUDA kernel source.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedI2sWeights {
    /// Output feature count.
    pub output_features: usize,
    /// Input feature count.
    pub input_features: usize,
    /// Scale block size along the input dimension.
    pub block_size: usize,
    /// Packed I2_S payload, 4 ternary weights per byte.
    pub packed_weights: Vec<u8>,
    /// Per-output, per-block scales.
    pub scales: Vec<f32>,
}

impl PackedI2sWeights {
    /// Construct and validate packed I2_S weight metadata.
    pub fn new(
        output_features: usize,
        input_features: usize,
        block_size: usize,
        packed_weights: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        let weights = Self { output_features, input_features, block_size, packed_weights, scales };
        weights.validate()?;
        Ok(weights)
    }

    /// Logical shape used by CUDA weight handles.
    pub fn shape(&self) -> Result<CudaTensorShape> {
        CudaTensorShape::matrix(self.output_features, self.input_features)
    }

    /// Minimum packed byte count required for the logical shape.
    pub fn expected_packed_bytes(&self) -> usize {
        self.output_features * self.packed_row_bytes()
    }

    /// Minimum scale count required for the logical shape.
    pub fn expected_scale_count(&self) -> usize {
        self.output_features * self.blocks_per_output()
    }

    /// Number of packed bytes consumed per output feature.
    pub fn packed_row_bytes(&self) -> usize {
        self.input_features.div_ceil(4)
    }

    /// Number of scale blocks consumed per output feature.
    pub fn blocks_per_output(&self) -> usize {
        self.input_features.div_ceil(self.block_size)
    }

    /// Scale payload byte count.
    pub fn scale_bytes_len(&self) -> usize {
        self.scales.len() * std::mem::size_of::<f32>()
    }

    fn validate(&self) -> Result<()> {
        if self.output_features == 0 || self.input_features == 0 {
            return Err(invalid_arguments(format!(
                "I2S weights require non-zero shape: output_features={} input_features={}",
                self.output_features, self.input_features
            )));
        }
        if self.block_size != 32 && self.block_size != 256 {
            return Err(invalid_arguments(format!(
                "I2S weights require block_size 32 or 256, got {}",
                self.block_size
            )));
        }
        let expected_packed = self.expected_packed_bytes();
        if self.packed_weights.len() < expected_packed {
            return Err(invalid_arguments(format!(
                "I2S packed weights too small: expected at least {expected_packed}, got {}",
                self.packed_weights.len()
            )));
        }
        let expected_scales = self.expected_scale_count();
        if self.scales.len() < expected_scales {
            return Err(invalid_arguments(format!(
                "I2S scales too small: expected at least {expected_scales}, got {}",
                self.scales.len()
            )));
        }
        Ok(())
    }
}

/// Strict GGUF no-scale QK256 weights for one BitNet linear projection.
///
/// The logical weight shape is `[output_features, input_features]`. The GGUF
/// payload is stored row-major as whole QK256 blocks: 64 packed bytes encode
/// each 256-column block, and the last block may include ignored tail columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedQk256Weights {
    /// Output feature count.
    pub output_features: usize,
    /// Input feature count.
    pub input_features: usize,
    /// Packed bytes per output row.
    pub row_stride_bytes: usize,
    /// QK256 no-scale packed payload.
    pub packed_weights: Vec<u8>,
    /// True when the payload was validated and packed during strict GGUF load.
    pub packed_at_load: bool,
}

impl PackedQk256Weights {
    /// Construct and validate strict GGUF no-scale QK256 weight metadata.
    ///
    /// This rejects non-canonical row strides and payload sizes before any CUDA
    /// upload can happen, matching the MS BitNet GGUF no-scale QK256 layout.
    pub fn from_strict_gguf_no_scale(
        output_features: usize,
        input_features: usize,
        row_stride_bytes: usize,
        packed_weights: Vec<u8>,
    ) -> Result<Self> {
        let weights = Self {
            output_features,
            input_features,
            row_stride_bytes,
            packed_weights,
            packed_at_load: true,
        };
        weights.validate_strict_gguf_no_scale()?;
        Ok(weights)
    }

    /// Logical shape used by CUDA weight handles.
    pub fn shape(&self) -> Result<CudaTensorShape> {
        CudaTensorShape::matrix(self.output_features, self.input_features)
    }

    /// Number of packed QK256 blocks consumed per output feature.
    pub fn blocks_per_output(&self) -> usize {
        self.input_features.div_ceil(QK256_BLOCK_COLS)
    }

    /// Canonical packed row stride for the logical input feature count.
    pub fn expected_row_stride_bytes(&self) -> Result<usize> {
        checked_mul(self.blocks_per_output(), QK256_PACKED_BYTES_PER_BLOCK, "QK256 row stride")
    }

    /// Exact packed byte count required for the logical shape.
    pub fn expected_packed_bytes(&self) -> Result<usize> {
        checked_mul(self.output_features, self.row_stride_bytes, "QK256 packed payload")
    }

    /// No-scale QK256 has no scale or side-table payload.
    pub const fn scale_bytes_len(&self) -> usize {
        0
    }

    fn validate_strict_gguf_no_scale(&self) -> Result<()> {
        if self.output_features == 0 || self.input_features == 0 {
            return Err(invalid_arguments(format!(
                "strict GGUF QK256 weights require non-zero shape: output_features={} input_features={}",
                self.output_features, self.input_features
            )));
        }
        let expected_row_stride = self.expected_row_stride_bytes()?;
        if self.row_stride_bytes != expected_row_stride {
            return Err(invalid_arguments(format!(
                "strict GGUF QK256 row stride mismatch: expected {expected_row_stride}, got {}",
                self.row_stride_bytes
            )));
        }
        let expected_packed = self.expected_packed_bytes()?;
        if self.packed_weights.len() != expected_packed {
            return Err(invalid_arguments(format!(
                "strict GGUF QK256 packed payload mismatch: expected {expected_packed}, got {}",
                self.packed_weights.len()
            )));
        }
        Ok(())
    }
}

/// Per-kernel stats recorded by the CUDA BitNet linear primitive.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaBitnetKernelInvocationStats {
    /// Stable kernel ID.
    pub kernel_id: String,
    /// Successful primitive invocations.
    pub invocations: u64,
    /// CPU fallback invocations. Strict CUDA proof requires this to stay zero.
    pub fallback_invocations: u64,
    /// Host-to-device activation bytes for primitive calls.
    pub host_to_device_bytes: u64,
    /// Device-to-host output bytes for primitive calls.
    pub device_to_host_bytes: u64,
    /// CUDA kernel launches.
    pub kernel_launches: u64,
    /// Optional measured kernel time in milliseconds.
    pub kernel_time_ms: Option<f64>,
    /// True when the associated weight handle was upload-once.
    pub weights_uploaded_once: bool,
    /// True if a per-token weight upload was recorded.
    pub per_token_weight_upload: bool,
}

impl CudaBitnetKernelInvocationStats {
    /// Construct empty stats for a kernel ID.
    pub fn new(kernel_id: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            invocations: 0,
            fallback_invocations: 0,
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            kernel_launches: 0,
            kernel_time_ms: None,
            weights_uploaded_once: false,
            per_token_weight_upload: false,
        }
    }

    #[cfg(feature = "cuda")]
    fn record_i2s_gemv(
        &mut self,
        host_to_device_bytes: u64,
        device_to_host_bytes: u64,
        weights_uploaded_once: bool,
        per_token_weight_upload: bool,
    ) {
        self.kernel_id = CUDA_BITNET_I2S_GEMV_KERNEL_ID.to_string();
        self.invocations += 1;
        self.kernel_launches += 1;
        self.host_to_device_bytes += host_to_device_bytes;
        self.device_to_host_bytes += device_to_host_bytes;
        self.weights_uploaded_once = weights_uploaded_once;
        self.per_token_weight_upload = per_token_weight_upload;
    }

    #[cfg(feature = "cuda")]
    fn record_qk256_gemv(
        &mut self,
        host_to_device_bytes: u64,
        device_to_host_bytes: u64,
        kernel_time_ms: Option<f64>,
        weights_uploaded_once: bool,
        per_token_weight_upload: bool,
    ) {
        self.kernel_id = CUDA_QK256_GEMV_KERNEL_ID.to_string();
        self.invocations += 1;
        self.kernel_launches += 1;
        self.host_to_device_bytes += host_to_device_bytes;
        self.device_to_host_bytes += device_to_host_bytes;
        if let Some(kernel_time_ms) = kernel_time_ms {
            self.kernel_time_ms =
                Some(self.kernel_time_ms.unwrap_or(0.0) + kernel_time_ms.max(0.0));
        }
        self.weights_uploaded_once = weights_uploaded_once;
        self.per_token_weight_upload = per_token_weight_upload;
    }
}

impl Default for CudaBitnetKernelInvocationStats {
    fn default() -> Self {
        Self::new(CUDA_BITNET_I2S_GEMV_KERNEL_ID)
    }
}

/// Reusable CUDA BitNet linear backend surface.
pub trait CudaBitnetLinearBackend {
    /// Upload I2_S weights once and return a stable CUDA handle.
    fn upload_i2s_weights(
        &mut self,
        tensor_name: &str,
        weights: &PackedI2sWeights,
    ) -> Result<CudaWeightHandle>;

    /// Upload strict GGUF no-scale QK256 weights once and return a stable CUDA handle.
    fn upload_qk256_weights(
        &mut self,
        tensor_name: &str,
        weights: &PackedQk256Weights,
    ) -> Result<CudaWeightHandle>;

    /// Run one I2_S GEMV using a CUDA-resident weight handle.
    fn i2s_gemv(
        &mut self,
        weights: &CudaWeightHandle,
        activation: &[f32],
        output: &mut [f32],
        stats: &mut CudaBitnetKernelInvocationStats,
    ) -> Result<()>;

    /// Run one strict GGUF no-scale QK256 GEMV using a CUDA-resident weight handle.
    fn qk256_gemv(
        &mut self,
        weights: &CudaWeightHandle,
        activation: &[f32],
        output: &mut [f32],
        seq_len: usize,
        bitnet_i8s_weight_scale: Option<f32>,
        stats: &mut CudaBitnetKernelInvocationStats,
    ) -> Result<()>;
}

/// Receipt-oriented summary of persistent CUDA BitNet lifetime state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBitnetReceiptFields {
    /// Requested backend label expected for this proof lane.
    pub requested_backend: &'static str,
    /// Selected backend label expected for this proof lane.
    pub selected_backend: &'static str,
    /// Runtime API for this proof lane.
    pub runtime_api: &'static str,
    /// True when a persistent CUDA context is owned by the BitNet context.
    pub cuda_context_persistent: bool,
    /// True when a persistent CUDA stream is owned by the BitNet context.
    pub cuda_stream_persistent: bool,
    /// Number of cached CUDA weight handles.
    pub weight_handle_count: usize,
    /// True when all cached weights were packed during strict model load.
    pub packed_at_load: bool,
    /// True when all cached weights were uploaded or registered once.
    pub weights_uploaded_once: bool,
    /// True when any decode-time/per-token weight upload was recorded.
    pub per_token_weight_upload: bool,
    /// Current reusable activation workspace capacity.
    pub activation_workspace_bytes: usize,
    /// True when an existing workspace allocation has been reused.
    pub activation_workspace_reused: bool,
    /// This infrastructure PR must not claim full BitNet CUDA inference.
    pub full_inference_claim: bool,
}

/// Persistent CUDA BitNet context, weight cache, workspace, and stats.
pub struct CudaBitnetContext {
    device: CudaBitnetDeviceInfo,
    #[cfg(feature = "cuda")]
    context: Option<Arc<CudaContext>>,
    #[cfg(feature = "cuda")]
    stream: Option<Arc<CudaStream>>,
    weight_cache: HashMap<String, CudaWeightHandle>,
    workspace: CudaActivationWorkspace,
    stats: CudaBitnetRuntimeStats,
    #[cfg(feature = "cuda")]
    device_weight_buffers: HashMap<CudaWeightId, CudaDeviceWeightBuffers>,
    #[cfg(feature = "cuda")]
    workspace_buffers: CudaActivationDeviceBuffers,
    #[cfg(feature = "cuda")]
    i2s_matmul_module: Option<Arc<CudaModule>>,
    #[cfg(feature = "cuda")]
    i2s_matmul_function: Option<CudaFunction>,
    #[cfg(feature = "cuda")]
    qk256_gemv_module: Option<Arc<CudaModule>>,
    #[cfg(feature = "cuda")]
    qk256_gemv_function: Option<CudaFunction>,
}

#[cfg(feature = "cuda")]
struct CudaDeviceWeightBuffers {
    packed: CudaSlice<u8>,
    _scale_bytes: Option<CudaSlice<u8>>,
    i2s_scales: Option<CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CudaActivationDeviceBuffers {
    activation: Option<CudaSlice<u8>>,
    output: Option<CudaSlice<u8>>,
    scratch: Option<CudaSlice<u8>>,
}

impl CudaBitnetContext {
    /// Create a CUDA-backed persistent BitNet context for `device_index`.
    ///
    /// This creates only a CUDA context and stream. It does not compile kernels
    /// or route transformer inference.
    #[cfg(feature = "cuda")]
    pub fn new(device_index: usize) -> Result<Self> {
        let context = CudaContext::new(device_index).map_err(|err| KernelError::GpuError {
            reason: format!("failed to create persistent CUDA BitNet context: {err:?}"),
        })?;
        let stream = context.default_stream();
        let device = query_cuda_bitnet_device_info(device_index, &context)?;

        Ok(Self {
            device,
            context: Some(context),
            stream: Some(stream),
            weight_cache: HashMap::new(),
            workspace: CudaActivationWorkspace::default(),
            stats: CudaBitnetRuntimeStats::default(),
            device_weight_buffers: HashMap::new(),
            workspace_buffers: CudaActivationDeviceBuffers::default(),
            i2s_matmul_module: None,
            i2s_matmul_function: None,
            qk256_gemv_module: None,
            qk256_gemv_function: None,
        })
    }

    /// Return an explicit unavailable error when the crate is built without CUDA.
    #[cfg(not(feature = "cuda"))]
    pub fn new(device_index: usize) -> Result<Self> {
        let _ = device_index;
        Err(KernelError::DeviceUnavailable {
            reason: "persistent CUDA BitNet context requires the cuda feature".to_string(),
        }
        .into())
    }

    /// Construct metadata-only context state for CPU-only tests and receipt planning.
    pub fn new_metadata_only(device: CudaBitnetDeviceInfo) -> Self {
        Self {
            device,
            #[cfg(feature = "cuda")]
            context: None,
            #[cfg(feature = "cuda")]
            stream: None,
            weight_cache: HashMap::new(),
            workspace: CudaActivationWorkspace::default(),
            stats: CudaBitnetRuntimeStats::default(),
            #[cfg(feature = "cuda")]
            device_weight_buffers: HashMap::new(),
            #[cfg(feature = "cuda")]
            workspace_buffers: CudaActivationDeviceBuffers::default(),
            #[cfg(feature = "cuda")]
            i2s_matmul_module: None,
            #[cfg(feature = "cuda")]
            i2s_matmul_function: None,
            #[cfg(feature = "cuda")]
            qk256_gemv_module: None,
            #[cfg(feature = "cuda")]
            qk256_gemv_function: None,
        }
    }

    /// Return the selected CUDA device identity.
    pub const fn device(&self) -> &CudaBitnetDeviceInfo {
        &self.device
    }

    /// Return the persistent CUDA context when this instance is CUDA-backed.
    #[cfg(feature = "cuda")]
    pub const fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.context.as_ref()
    }

    /// Return the persistent CUDA stream when this instance is CUDA-backed.
    #[cfg(feature = "cuda")]
    pub const fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.stream.as_ref()
    }

    /// Return the current weight cache.
    pub const fn weight_cache(&self) -> &HashMap<String, CudaWeightHandle> {
        &self.weight_cache
    }

    /// Return an uploaded weight handle by tensor name.
    pub fn weight_handle(&self, tensor_name: &str) -> Option<&CudaWeightHandle> {
        self.weight_cache.get(tensor_name)
    }

    /// Return the reusable activation workspace metadata.
    pub const fn workspace(&self) -> &CudaActivationWorkspace {
        &self.workspace
    }

    /// Return receipt-oriented runtime stats.
    pub const fn stats(&self) -> &CudaBitnetRuntimeStats {
        &self.stats
    }

    /// Upload or register a packed BitNet weight once and return a stable handle.
    ///
    /// Repeating the call with identical metadata returns the existing handle and
    /// does not increment upload counters. Reusing a tensor name with different
    /// metadata is rejected to avoid stale decode-time handles.
    pub fn upload_weight_once(
        &mut self,
        tensor_name: impl Into<String>,
        shape: CudaTensorShape,
        kernel_family: CudaBitnetKernelFamily,
        packed_weights: &[u8],
        scale_bytes: &[u8],
    ) -> Result<CudaWeightHandle> {
        let tensor_name = tensor_name.into();
        validate_weight_upload(&tensor_name, packed_weights)?;

        if let Some(existing) = self.weight_cache.get(&tensor_name) {
            validate_existing_handle(existing, &shape, kernel_family, packed_weights, scale_bytes)?;
            return Ok(existing.clone());
        }

        let id = CudaWeightId(NEXT_WEIGHT_ID.fetch_add(1, Ordering::Relaxed));
        let handle = CudaWeightHandle {
            id,
            tensor_name: tensor_name.clone(),
            shape,
            kernel_family,
            packed_bytes: packed_weights.len(),
            scale_bytes: scale_bytes.len(),
            block_size: None,
            packed_at_load: true,
            uploaded_once: true,
        };

        #[cfg(feature = "cuda")]
        if let Some(stream) = &self.stream {
            let packed =
                stream.memcpy_stod(packed_weights).map_err(|err| KernelError::GpuError {
                    reason: format!("failed to upload CUDA BitNet weight '{tensor_name}': {err:?}"),
                })?;
            let scales = if scale_bytes.is_empty() {
                None
            } else {
                Some(stream.memcpy_stod(scale_bytes).map_err(|err| KernelError::GpuError {
                    reason: format!(
                        "failed to upload CUDA BitNet scales for '{tensor_name}': {err:?}"
                    ),
                })?)
            };
            self.device_weight_buffers.insert(
                id,
                CudaDeviceWeightBuffers { packed, _scale_bytes: scales, i2s_scales: None },
            );
        }

        let upload_bytes =
            u64::try_from(handle.total_bytes()).map_err(|_| KernelError::InvalidArguments {
                reason: format!(
                    "CUDA weight '{}' byte count exceeds receipt counter range",
                    handle.tensor_name
                ),
            })?;
        self.stats.weight_uploads += 1;
        self.stats.weight_upload_bytes += upload_bytes;
        self.weight_cache.insert(tensor_name, handle.clone());
        Ok(handle)
    }

    /// Ensure the reusable activation workspace can satisfy the requested sizes.
    pub fn ensure_activation_workspace(
        &mut self,
        activation_bytes: usize,
        output_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<&CudaActivationWorkspace> {
        let grows = activation_bytes > self.workspace.activation_bytes
            || output_bytes > self.workspace.output_bytes
            || scratch_bytes > self.workspace.scratch_bytes;

        if grows {
            self.workspace.activation_bytes = self.workspace.activation_bytes.max(activation_bytes);
            self.workspace.output_bytes = self.workspace.output_bytes.max(output_bytes);
            self.workspace.scratch_bytes = self.workspace.scratch_bytes.max(scratch_bytes);
            self.workspace.growth_count += 1;
            self.stats.workspace_growths += 1;
            self.reallocate_activation_workspace()?;
        } else {
            self.workspace.reuse_count += 1;
            self.stats.workspace_reuses += 1;
        }

        Ok(&self.workspace)
    }

    /// Return receipt fields proving lifetime behavior without inference claims.
    pub fn receipt_fields(&self) -> CudaBitnetReceiptFields {
        let weight_handle_count = self.weight_cache.len();
        let packed_at_load = weight_handle_count > 0
            && self.weight_cache.values().all(|handle| handle.packed_at_load);
        let weights_uploaded_once = weight_handle_count > 0
            && self.weight_cache.values().all(|handle| handle.uploaded_once)
            && self.stats.weight_uploads == weight_handle_count as u64;

        CudaBitnetReceiptFields {
            requested_backend: "nvidia-rtx-5070-ti-cuda",
            selected_backend: "nvidia-rtx-5070-ti-cuda",
            runtime_api: "cuda",
            cuda_context_persistent: self.has_persistent_cuda_context(),
            cuda_stream_persistent: self.has_persistent_cuda_stream(),
            weight_handle_count,
            packed_at_load,
            weights_uploaded_once,
            per_token_weight_upload: self.stats.per_token_weight_uploads > 0,
            activation_workspace_bytes: self.workspace.total_bytes(),
            activation_workspace_reused: self.workspace.reuse_count > 0,
            full_inference_claim: false,
        }
    }

    #[cfg(feature = "cuda")]
    fn has_persistent_cuda_context(&self) -> bool {
        self.context.is_some()
    }

    #[cfg(not(feature = "cuda"))]
    const fn has_persistent_cuda_context(&self) -> bool {
        false
    }

    #[cfg(feature = "cuda")]
    fn has_persistent_cuda_stream(&self) -> bool {
        self.stream.is_some()
    }

    #[cfg(not(feature = "cuda"))]
    const fn has_persistent_cuda_stream(&self) -> bool {
        false
    }

    #[cfg(feature = "cuda")]
    fn reallocate_activation_workspace(&mut self) -> Result<()> {
        let Some(stream) = &self.stream else {
            return Ok(());
        };

        if self.workspace.activation_bytes > 0 {
            self.workspace_buffers.activation =
                Some(stream.alloc_zeros(self.workspace.activation_bytes).map_err(|err| {
                    KernelError::GpuError {
                        reason: format!("failed to allocate CUDA activation workspace: {err:?}"),
                    }
                })?);
        }
        if self.workspace.output_bytes > 0 {
            self.workspace_buffers.output =
                Some(stream.alloc_zeros(self.workspace.output_bytes).map_err(|err| {
                    KernelError::GpuError {
                        reason: format!("failed to allocate CUDA output workspace: {err:?}"),
                    }
                })?);
        }
        if self.workspace.scratch_bytes > 0 {
            self.workspace_buffers.scratch =
                Some(stream.alloc_zeros(self.workspace.scratch_bytes).map_err(|err| {
                    KernelError::GpuError {
                        reason: format!("failed to allocate CUDA scratch workspace: {err:?}"),
                    }
                })?);
        }

        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    const fn reallocate_activation_workspace(&mut self) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_i2s_matmul_function(&mut self) -> Result<()> {
        if self.i2s_matmul_function.is_some() {
            return Ok(());
        }

        let context = self.context.as_ref().ok_or_else(|| KernelError::DeviceUnavailable {
            reason: "CUDA I2S GEMV requires a CUDA-backed BitNet context".to_string(),
        })?;
        let ptx = compile_cuda_bitnet_ptx(I2S_MATMUL_KERNEL_SRC, "BitNet I2S GEMV")?;
        let module = context.load_module(ptx).map_err(|err| KernelError::GpuError {
            reason: format!("failed to load CUDA I2S GEMV module: {err:?}"),
        })?;
        let function = module.load_function(I2S_MATMUL_FUNCTION_NAME).map_err(|err| {
            KernelError::GpuError {
                reason: format!("failed to load CUDA I2S GEMV kernel: {err:?}"),
            }
        })?;

        self.i2s_matmul_module = Some(module);
        self.i2s_matmul_function = Some(function);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_qk256_gemv_function(&mut self) -> Result<()> {
        if self.qk256_gemv_function.is_some() {
            return Ok(());
        }

        let context = self.context.as_ref().ok_or_else(|| KernelError::DeviceUnavailable {
            reason: "CUDA QK256 GEMV requires a CUDA-backed BitNet context".to_string(),
        })?;
        let ptx = compile_cuda_bitnet_ptx(CUDA_QK256_GEMV_KERNEL_SRC, "BitNet QK256 GEMV")?;
        let module = context.load_module(ptx).map_err(|err| KernelError::GpuError {
            reason: format!("failed to load CUDA QK256 GEMV module: {err:?}"),
        })?;
        let function = module.load_function(CUDA_QK256_GEMV_KERNEL_ID).map_err(|err| {
            KernelError::GpuError {
                reason: format!("failed to load CUDA QK256 GEMV kernel: {err:?}"),
            }
        })?;

        self.qk256_gemv_module = Some(module);
        self.qk256_gemv_function = Some(function);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_i2s_gemv_cuda(
        &mut self,
        weights: &CudaWeightHandle,
        activation: &[f32],
        output: &mut [f32],
        config: &I2sMatmulConfig,
    ) -> Result<()> {
        self.ensure_i2s_matmul_function()?;

        let stream = self.stream.as_ref().ok_or_else(|| KernelError::DeviceUnavailable {
            reason: "CUDA I2S GEMV requires a CUDA-backed BitNet stream".to_string(),
        })?;
        let function = self.i2s_matmul_function.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: "CUDA I2S GEMV kernel was not loaded".to_string(),
        })?;
        let buffers = self.device_weight_buffers.get(&weights.id).ok_or_else(|| {
            KernelError::DeviceUnavailable {
                reason: format!(
                    "CUDA I2S GEMV weight '{}' is not resident on the selected device",
                    weights.tensor_name
                ),
            }
        })?;
        let scales = buffers.i2s_scales.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: format!("CUDA I2S GEMV scales for '{}' are not resident", weights.tensor_name),
        })?;

        let activation_dev =
            stream.memcpy_stod(activation).map_err(|err| KernelError::GpuError {
                reason: format!("failed to copy CUDA I2S GEMV activation to device: {err:?}"),
            })?;
        let mut output_dev: CudaSlice<f32> =
            stream.alloc_zeros(config.n).map_err(|err| KernelError::GpuError {
                reason: format!("failed to allocate CUDA I2S GEMV output: {err:?}"),
            })?;

        let launch_config = LaunchConfig {
            grid_dim: config.grid_dim(),
            block_dim: config.block_dim(),
            shared_mem_bytes: config.shared_mem_bytes,
        };
        let mut builder = stream.launch_builder(function);
        builder.arg(&activation_dev);
        builder.arg(&buffers.packed);
        builder.arg(scales);
        builder.arg(&mut output_dev);
        let m_arg = 1_i32;
        let n_arg = i32::try_from(config.n).map_err(|_| KernelError::InvalidArguments {
            reason: format!("CUDA I2S GEMV output dimension exceeds i32: n={}", config.n),
        })?;
        let k_arg = i32::try_from(config.k).map_err(|_| KernelError::InvalidArguments {
            reason: format!("CUDA I2S GEMV input dimension exceeds i32: k={}", config.k),
        })?;
        let block_size_arg =
            i32::try_from(config.block_size).map_err(|_| KernelError::InvalidArguments {
                reason: format!(
                    "CUDA I2S GEMV block size exceeds i32: block_size={}",
                    config.block_size
                ),
            })?;
        builder.arg(&m_arg);
        builder.arg(&n_arg);
        builder.arg(&k_arg);
        builder.arg(&block_size_arg);

        unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
            reason: format!("failed to launch CUDA I2S GEMV kernel: {err:?}"),
        })?;
        stream.synchronize().map_err(|err| KernelError::GpuError {
            reason: format!("failed to synchronize CUDA I2S GEMV kernel: {err:?}"),
        })?;

        let output_host: Vec<f32> =
            stream.memcpy_dtov(&output_dev).map_err(|err| KernelError::GpuError {
                reason: format!("failed to copy CUDA I2S GEMV output to host: {err:?}"),
            })?;
        output[..config.n].copy_from_slice(&output_host[..config.n]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_qk256_gemv_cuda(
        &mut self,
        weights: &CudaWeightHandle,
        activation: &[f32],
        output: &mut [f32],
        config: &Qk256GemvConfig,
    ) -> Result<Option<f64>> {
        self.ensure_qk256_gemv_function()?;

        let stream = self.stream.as_ref().ok_or_else(|| KernelError::DeviceUnavailable {
            reason: "CUDA QK256 GEMV requires a CUDA-backed BitNet stream".to_string(),
        })?;
        let function = self.qk256_gemv_function.as_ref().ok_or_else(|| KernelError::GpuError {
            reason: "CUDA QK256 GEMV kernel was not loaded".to_string(),
        })?;
        let buffers = self.device_weight_buffers.get(&weights.id).ok_or_else(|| {
            KernelError::DeviceUnavailable {
                reason: format!(
                    "CUDA QK256 GEMV weight '{}' is not resident on the selected device",
                    weights.tensor_name
                ),
            }
        })?;

        let input_len = checked_mul(config.seq_len, config.k, "QK256 activation")?;
        let output_len = checked_mul(config.seq_len, config.n_out, "QK256 output")?;
        let activation_dev =
            stream.memcpy_stod(&activation[..input_len]).map_err(|err| KernelError::GpuError {
                reason: format!("failed to copy CUDA QK256 GEMV activation to device: {err:?}"),
            })?;
        let mut output_dev: CudaSlice<f32> =
            stream.alloc_zeros(output_len).map_err(|err| KernelError::GpuError {
                reason: format!("failed to allocate CUDA QK256 GEMV output: {err:?}"),
            })?;

        let launch_config = LaunchConfig {
            grid_dim: config.grid_dim(),
            block_dim: config.block_dim(),
            shared_mem_bytes: config.shared_mem_bytes,
        };
        let mut builder = stream.launch_builder(function);
        builder.arg(&buffers.packed);
        builder.arg(&activation_dev);
        builder.arg(&mut output_dev);
        let seq_len_arg =
            i32::try_from(config.seq_len).map_err(|_| KernelError::InvalidArguments {
                reason: format!("CUDA QK256 GEMV seq_len exceeds i32: {}", config.seq_len),
            })?;
        let n_out_arg = i32::try_from(config.n_out).map_err(|_| KernelError::InvalidArguments {
            reason: format!("CUDA QK256 GEMV n_out exceeds i32: {}", config.n_out),
        })?;
        let k_arg = i32::try_from(config.k).map_err(|_| KernelError::InvalidArguments {
            reason: format!("CUDA QK256 GEMV k exceeds i32: {}", config.k),
        })?;
        let row_stride_arg =
            i32::try_from(config.row_stride_bytes).map_err(|_| KernelError::InvalidArguments {
                reason: format!(
                    "CUDA QK256 GEMV row_stride_bytes exceeds i32: {}",
                    config.row_stride_bytes
                ),
            })?;
        builder.arg(&seq_len_arg);
        builder.arg(&n_out_arg);
        builder.arg(&k_arg);
        builder.arg(&row_stride_arg);
        let bitnet_i8s_weight_scale_arg = config.bitnet_i8s_weight_scale.unwrap_or(1.0);
        let use_bitnet_i8s_arg = i32::from(config.bitnet_i8s_weight_scale.is_some());
        builder.arg(&bitnet_i8s_weight_scale_arg);
        builder.arg(&use_bitnet_i8s_arg);

        let start_event =
            stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(|err| {
                KernelError::GpuError {
                    reason: format!("failed to record CUDA QK256 GEMV start event: {err:?}"),
                }
            })?;
        unsafe { builder.launch(launch_config) }.map_err(|err| KernelError::GpuError {
            reason: format!("failed to launch CUDA QK256 GEMV kernel: {err:?}"),
        })?;
        let end_event =
            stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(|err| {
                KernelError::GpuError {
                    reason: format!("failed to record CUDA QK256 GEMV end event: {err:?}"),
                }
            })?;
        let kernel_time_ms =
            f64::from(start_event.elapsed_ms(&end_event).map_err(|err| KernelError::GpuError {
                reason: format!("failed to measure CUDA QK256 GEMV event time: {err:?}"),
            })?);
        stream.synchronize().map_err(|err| KernelError::GpuError {
            reason: format!("failed to synchronize CUDA QK256 GEMV kernel: {err:?}"),
        })?;

        let output_host: Vec<f32> =
            stream.memcpy_dtov(&output_dev).map_err(|err| KernelError::GpuError {
                reason: format!("failed to copy CUDA QK256 GEMV output to host: {err:?}"),
            })?;
        output[..output_len].copy_from_slice(&output_host[..output_len]);
        Ok(Some(kernel_time_ms))
    }
}

impl CudaBitnetLinearBackend for CudaBitnetContext {
    fn upload_i2s_weights(
        &mut self,
        tensor_name: &str,
        weights: &PackedI2sWeights,
    ) -> Result<CudaWeightHandle> {
        weights.validate()?;
        validate_weight_upload(tensor_name, &weights.packed_weights)?;
        let shape = weights.shape()?;

        if let Some(existing) = self.weight_cache.get(tensor_name) {
            validate_existing_i2s_handle(existing, &shape, weights)?;
            return Ok(existing.clone());
        }

        let id = CudaWeightId(NEXT_WEIGHT_ID.fetch_add(1, Ordering::Relaxed));
        let handle = CudaWeightHandle {
            id,
            tensor_name: tensor_name.to_string(),
            shape,
            kernel_family: CudaBitnetKernelFamily::I2s,
            packed_bytes: weights.packed_weights.len(),
            scale_bytes: weights.scale_bytes_len(),
            block_size: Some(weights.block_size),
            packed_at_load: true,
            uploaded_once: true,
        };

        #[cfg(feature = "cuda")]
        if let Some(stream) = &self.stream {
            let packed = stream.memcpy_stod(&weights.packed_weights).map_err(|err| {
                KernelError::GpuError {
                    reason: format!(
                        "failed to upload CUDA I2S weight '{}': {err:?}",
                        handle.tensor_name
                    ),
                }
            })?;
            let scales =
                stream.memcpy_stod(&weights.scales).map_err(|err| KernelError::GpuError {
                    reason: format!(
                        "failed to upload CUDA I2S scales for '{}': {err:?}",
                        handle.tensor_name
                    ),
                })?;
            self.device_weight_buffers.insert(
                id,
                CudaDeviceWeightBuffers { packed, _scale_bytes: None, i2s_scales: Some(scales) },
            );
        }

        let upload_bytes =
            u64::try_from(handle.total_bytes()).map_err(|_| KernelError::InvalidArguments {
                reason: format!(
                    "CUDA I2S weight '{}' byte count exceeds receipt counter range",
                    handle.tensor_name
                ),
            })?;
        self.stats.weight_uploads += 1;
        self.stats.weight_upload_bytes += upload_bytes;
        self.weight_cache.insert(handle.tensor_name.clone(), handle.clone());
        Ok(handle)
    }

    fn upload_qk256_weights(
        &mut self,
        tensor_name: &str,
        weights: &PackedQk256Weights,
    ) -> Result<CudaWeightHandle> {
        weights.validate_strict_gguf_no_scale()?;
        validate_weight_upload(tensor_name, &weights.packed_weights)?;
        let shape = weights.shape()?;

        if let Some(existing) = self.weight_cache.get(tensor_name) {
            validate_existing_qk256_handle(existing, &shape, weights)?;
            return Ok(existing.clone());
        }

        let id = CudaWeightId(NEXT_WEIGHT_ID.fetch_add(1, Ordering::Relaxed));
        let handle = CudaWeightHandle {
            id,
            tensor_name: tensor_name.to_string(),
            shape,
            kernel_family: CudaBitnetKernelFamily::Qk256,
            packed_bytes: weights.packed_weights.len(),
            scale_bytes: weights.scale_bytes_len(),
            block_size: Some(QK256_BLOCK_COLS),
            packed_at_load: weights.packed_at_load,
            uploaded_once: true,
        };

        #[cfg(feature = "cuda")]
        if let Some(stream) = &self.stream {
            let packed = stream.memcpy_stod(&weights.packed_weights).map_err(|err| {
                KernelError::GpuError {
                    reason: format!(
                        "failed to upload CUDA QK256 weight '{}': {err:?}",
                        handle.tensor_name
                    ),
                }
            })?;
            self.device_weight_buffers.insert(
                id,
                CudaDeviceWeightBuffers { packed, _scale_bytes: None, i2s_scales: None },
            );
        }

        let upload_bytes =
            u64::try_from(handle.total_bytes()).map_err(|_| KernelError::InvalidArguments {
                reason: format!(
                    "CUDA QK256 weight '{}' byte count exceeds receipt counter range",
                    handle.tensor_name
                ),
            })?;
        self.stats.weight_uploads += 1;
        self.stats.weight_upload_bytes += upload_bytes;
        self.weight_cache.insert(handle.tensor_name.clone(), handle.clone());
        Ok(handle)
    }

    fn i2s_gemv(
        &mut self,
        weights: &CudaWeightHandle,
        activation: &[f32],
        output: &mut [f32],
        stats: &mut CudaBitnetKernelInvocationStats,
    ) -> Result<()> {
        let gemv_shape = validate_i2s_gemv_args(weights, activation, output)?;
        let output_features = gemv_shape.output_features;
        let input_features = gemv_shape.input_features;
        #[cfg(feature = "cuda")]
        let block_size = gemv_shape.block_size;
        #[cfg(not(feature = "cuda"))]
        let _ = gemv_shape.block_size;
        let activation_bytes = checked_f32_bytes(input_features, "I2S activation")?;
        let output_bytes = checked_f32_bytes(output_features, "I2S output")?;
        self.ensure_activation_workspace(activation_bytes, output_bytes, 0)?;

        #[cfg(feature = "cuda")]
        {
            let config =
                I2sMatmulConfig::for_shape(1, output_features, input_features, block_size)?;
            self.launch_i2s_gemv_cuda(weights, &activation[..input_features], output, &config)?;
            stats.record_i2s_gemv(
                u64::try_from(activation_bytes).map_err(|_| KernelError::InvalidArguments {
                    reason: "I2S activation byte count exceeds receipt counter range".to_string(),
                })?,
                u64::try_from(output_bytes).map_err(|_| KernelError::InvalidArguments {
                    reason: "I2S output byte count exceeds receipt counter range".to_string(),
                })?,
                weights.uploaded_once,
                self.stats.per_token_weight_uploads > 0,
            );
            return Ok(());
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = weights;
            let _ = activation;
            let _ = output;
            let _ = stats;
            Err(KernelError::DeviceUnavailable {
                reason: "CUDA I2S GEMV requires the cuda feature".to_string(),
            }
            .into())
        }
    }

    fn qk256_gemv(
        &mut self,
        weights: &CudaWeightHandle,
        activation: &[f32],
        output: &mut [f32],
        seq_len: usize,
        bitnet_i8s_weight_scale: Option<f32>,
        stats: &mut CudaBitnetKernelInvocationStats,
    ) -> Result<()> {
        let gemv_shape = validate_qk256_gemv_args(weights, activation, output, seq_len)?;
        if let Some(scale) = bitnet_i8s_weight_scale
            && !scale.is_finite()
        {
            return Err(KernelError::InvalidArguments {
                reason: format!("QK256 GEMV BitNet I8_S weight scale is not finite: {scale}"),
            }
            .into());
        }
        let output_features = gemv_shape.output_features;
        let input_features = gemv_shape.input_features;
        let activation_len = checked_mul(seq_len, input_features, "QK256 activation")?;
        let output_len = checked_mul(seq_len, output_features, "QK256 output")?;
        let activation_bytes = checked_f32_bytes(activation_len, "QK256 activation")?;
        let output_bytes = checked_f32_bytes(output_len, "QK256 output")?;
        self.ensure_activation_workspace(activation_bytes, output_bytes, 0)?;

        #[cfg(feature = "cuda")]
        {
            let mut config = Qk256GemvConfig::for_shape(seq_len, output_features, input_features)?;
            config.bitnet_i8s_weight_scale = bitnet_i8s_weight_scale;
            let kernel_time_ms = self.launch_qk256_gemv_cuda(
                weights,
                &activation[..activation_len],
                output,
                &config,
            )?;
            stats.record_qk256_gemv(
                u64::try_from(activation_bytes).map_err(|_| KernelError::InvalidArguments {
                    reason: "QK256 activation byte count exceeds receipt counter range".to_string(),
                })?,
                u64::try_from(output_bytes).map_err(|_| KernelError::InvalidArguments {
                    reason: "QK256 output byte count exceeds receipt counter range".to_string(),
                })?,
                kernel_time_ms,
                weights.uploaded_once,
                self.stats.per_token_weight_uploads > 0,
            );
            return Ok(());
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = weights;
            let _ = activation;
            let _ = output;
            let _ = stats;
            let _ = bitnet_i8s_weight_scale;
            Err(KernelError::DeviceUnavailable {
                reason: "CUDA QK256 GEMV requires the cuda feature".to_string(),
            }
            .into())
        }
    }
}

struct I2sGemvShape {
    output_features: usize,
    input_features: usize,
    block_size: usize,
}

struct Qk256GemvShape {
    output_features: usize,
    input_features: usize,
}

fn validate_weight_upload(tensor_name: &str, packed_weights: &[u8]) -> Result<()> {
    if tensor_name.trim().is_empty() {
        return Err(invalid_arguments("CUDA weight tensor name must be non-empty"));
    }
    if packed_weights.is_empty() {
        return Err(invalid_arguments("CUDA packed weight payload must be non-empty"));
    }
    Ok(())
}

fn validate_existing_handle(
    existing: &CudaWeightHandle,
    shape: &CudaTensorShape,
    kernel_family: CudaBitnetKernelFamily,
    packed_weights: &[u8],
    scale_bytes: &[u8],
) -> Result<()> {
    let matches = existing.shape == *shape
        && existing.kernel_family == kernel_family
        && existing.packed_bytes == packed_weights.len()
        && existing.scale_bytes == scale_bytes.len()
        && existing.block_size.is_none()
        && existing.packed_at_load
        && existing.uploaded_once;

    if matches {
        Ok(())
    } else {
        Err(invalid_arguments(format!(
            "CUDA weight '{}' is already cached with different metadata",
            existing.tensor_name
        )))
    }
}

fn validate_existing_i2s_handle(
    existing: &CudaWeightHandle,
    shape: &CudaTensorShape,
    weights: &PackedI2sWeights,
) -> Result<()> {
    let matches = existing.shape == *shape
        && existing.kernel_family == CudaBitnetKernelFamily::I2s
        && existing.packed_bytes == weights.packed_weights.len()
        && existing.scale_bytes == weights.scale_bytes_len()
        && existing.block_size == Some(weights.block_size)
        && existing.packed_at_load
        && existing.uploaded_once;

    if matches {
        Ok(())
    } else {
        Err(invalid_arguments(format!(
            "CUDA I2S weight '{}' is already cached with different metadata",
            existing.tensor_name
        )))
    }
}

fn validate_existing_qk256_handle(
    existing: &CudaWeightHandle,
    shape: &CudaTensorShape,
    weights: &PackedQk256Weights,
) -> Result<()> {
    let matches = existing.shape == *shape
        && existing.kernel_family == CudaBitnetKernelFamily::Qk256
        && existing.packed_bytes == weights.packed_weights.len()
        && existing.scale_bytes == weights.scale_bytes_len()
        && existing.block_size == Some(QK256_BLOCK_COLS)
        && existing.packed_at_load == weights.packed_at_load
        && existing.uploaded_once;

    if matches {
        Ok(())
    } else {
        Err(invalid_arguments(format!(
            "CUDA QK256 weight '{}' is already cached with different metadata",
            existing.tensor_name
        )))
    }
}

fn validate_i2s_gemv_args(
    weights: &CudaWeightHandle,
    activation: &[f32],
    output: &[f32],
) -> Result<I2sGemvShape> {
    if weights.kernel_family != CudaBitnetKernelFamily::I2s {
        return Err(invalid_arguments(format!(
            "CUDA I2S GEMV requires an I2S weight handle, got {}",
            weights.kernel_family.as_str()
        )));
    }
    if weights.shape.dims.len() != 2 {
        return Err(invalid_arguments(format!(
            "CUDA I2S GEMV requires matrix weights, got {:?}",
            weights.shape.dims
        )));
    }
    let output_features = weights.shape.dims[0];
    let input_features = weights.shape.dims[1];
    let block_size = weights.block_size.ok_or_else(|| {
        invalid_arguments(format!(
            "CUDA I2S GEMV weight '{}' is missing block size metadata",
            weights.tensor_name
        ))
    })?;

    if activation.len() < input_features {
        return Err(invalid_arguments(format!(
            "CUDA I2S GEMV activation too small: expected at least {input_features}, got {}",
            activation.len()
        )));
    }
    if output.len() < output_features {
        return Err(invalid_arguments(format!(
            "CUDA I2S GEMV output too small: expected at least {output_features}, got {}",
            output.len()
        )));
    }
    Ok(I2sGemvShape { output_features, input_features, block_size })
}

fn validate_qk256_gemv_args(
    weights: &CudaWeightHandle,
    activation: &[f32],
    output: &[f32],
    seq_len: usize,
) -> Result<Qk256GemvShape> {
    if weights.kernel_family != CudaBitnetKernelFamily::Qk256 {
        return Err(invalid_arguments(format!(
            "CUDA QK256 GEMV requires a QK256 weight handle, got {}",
            weights.kernel_family.as_str()
        )));
    }
    if weights.shape.dims.len() != 2 {
        return Err(invalid_arguments(format!(
            "CUDA QK256 GEMV requires matrix weights, got {:?}",
            weights.shape.dims
        )));
    }
    if seq_len == 0 {
        return Err(invalid_arguments("CUDA QK256 GEMV seq_len must be non-zero"));
    }

    let output_features = weights.shape.dims[0];
    let input_features = weights.shape.dims[1];
    let expected_activation = checked_mul(seq_len, input_features, "QK256 activation")?;
    let expected_output = checked_mul(seq_len, output_features, "QK256 output")?;
    if activation.len() < expected_activation {
        return Err(invalid_arguments(format!(
            "CUDA QK256 GEMV activation too small: expected at least {expected_activation}, got {}",
            activation.len()
        )));
    }
    if output.len() < expected_output {
        return Err(invalid_arguments(format!(
            "CUDA QK256 GEMV output too small: expected at least {expected_output}, got {}",
            output.len()
        )));
    }
    Ok(Qk256GemvShape { output_features, input_features })
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        KernelError::InvalidArguments { reason: format!("{label} length overflow: {lhs} * {rhs}") }
            .into()
    })
}

fn checked_f32_bytes(count: usize, label: &str) -> Result<usize> {
    count.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
        KernelError::InvalidArguments {
            reason: format!("{label} byte count overflow for {count} f32 values"),
        }
        .into()
    })
}

fn invalid_arguments(reason: impl Into<String>) -> bitnet_common::BitNetError {
    KernelError::InvalidArguments { reason: reason.into() }.into()
}

#[cfg(feature = "cuda")]
fn compile_cuda_bitnet_ptx(source: &str, label: &str) -> Result<Ptx> {
    let _hook_guard = NVRTC_COMPILE_LOCK.lock().ok();
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let compile_result = std::panic::catch_unwind(|| compile_ptx(source));
    std::panic::set_hook(previous_hook);

    match compile_result {
        Ok(Ok(ptx)) => Ok(ptx),
        Ok(Err(err)) => {
            Err(KernelError::GpuError { reason: format!("failed to compile {label} PTX: {err:?}") }
                .into())
        }
        Err(payload) => Err(KernelError::GpuError {
            reason: format!(
                "failed to compile {label} PTX because NVRTC was unavailable: {}",
                panic_payload_message(&*payload)
            ),
        }
        .into()),
    }
}

#[cfg(feature = "cuda")]
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(feature = "cuda")]
fn query_cuda_bitnet_device_info(
    device_index: usize,
    context: &CudaContext,
) -> Result<CudaBitnetDeviceInfo> {
    let device_name = context.name().map_err(|err| KernelError::GpuError {
        reason: format!("failed to query CUDA device name: {err:?}"),
    })?;
    let major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|err| KernelError::GpuError {
            reason: format!("failed to query CUDA compute capability major: {err:?}"),
        })?;
    let minor = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|err| KernelError::GpuError {
            reason: format!("failed to query CUDA compute capability minor: {err:?}"),
        })?;
    let total_memory = unsafe { cu_device::total_mem(context.cu_device()) }.map_err(|err| {
        KernelError::GpuError { reason: format!("failed to query CUDA total memory: {err:?}") }
    })?;

    let compute_major = u32::try_from(major).map_err(|_| KernelError::GpuError {
        reason: format!("invalid CUDA compute capability major value: {major}"),
    })?;
    let compute_minor = u32::try_from(minor).map_err(|_| KernelError::GpuError {
        reason: format!("invalid CUDA compute capability minor value: {minor}"),
    })?;
    let vram_bytes = u64::try_from(total_memory).map_err(|_| KernelError::GpuError {
        reason: format!("invalid CUDA total memory value: {total_memory}"),
    })?;

    Ok(CudaBitnetDeviceInfo {
        device_index,
        device_name,
        compute_capability: (compute_major, compute_minor),
        vram_bytes: Some(vram_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> CudaBitnetContext {
        CudaBitnetContext::new_metadata_only(CudaBitnetDeviceInfo {
            device_index: 0,
            device_name: "NVIDIA GeForce RTX 5070 Ti".to_string(),
            compute_capability: (12, 0),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
        })
    }

    fn i2s_fixture() -> PackedI2sWeights {
        PackedI2sWeights::new(
            3,
            5,
            32,
            vec![
                pack_i2s_row([1, 0, -1, 1, 0]),
                pack_i2s_tail([0]),
                pack_i2s_row([-1, 1, 0, 0, 1]),
                pack_i2s_tail([1]),
                pack_i2s_row([0, -1, 1, 0, -1]),
                pack_i2s_tail([-1]),
            ],
            vec![1.0, 0.5, 2.0],
        )
        .unwrap()
    }

    fn qk256_fixture() -> PackedQk256Weights {
        PackedQk256Weights::from_strict_gguf_no_scale(3, 300, 128, vec![0xaa; 3 * 128]).unwrap()
    }

    fn pack_i2s_row(vals: [i8; 5]) -> u8 {
        pack_i2s_nibble([vals[0], vals[1], vals[2], vals[3]])
    }

    fn pack_i2s_tail(vals: [i8; 1]) -> u8 {
        pack_i2s_nibble([vals[0], 0, 0, 0])
    }

    fn pack_i2s_nibble(vals: [i8; 4]) -> u8 {
        vals.iter().enumerate().fold(0_u8, |byte, (index, value)| {
            let code = match value {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            byte | (code << (index * 2))
        })
    }

    #[test]
    fn tensor_shape_rejects_empty_or_zero_dims() {
        assert!(CudaTensorShape::new(Vec::<usize>::new()).is_err());
        assert!(CudaTensorShape::new(vec![4, 0]).is_err());
        assert_eq!(CudaTensorShape::matrix(4, 8).unwrap().element_count(), 32);
    }

    #[test]
    fn packed_i2s_weights_accept_tail_bits() {
        let weights = i2s_fixture();

        assert_eq!(weights.output_features, 3);
        assert_eq!(weights.input_features, 5);
        assert_eq!(weights.packed_row_bytes(), 2);
        assert_eq!(weights.blocks_per_output(), 1);
        assert_eq!(weights.expected_packed_bytes(), 6);
        assert_eq!(weights.expected_scale_count(), 3);
        assert_eq!(weights.shape().unwrap().dims, vec![3, 5]);
    }

    #[test]
    fn packed_i2s_weights_reject_incomplete_payloads() {
        assert!(PackedI2sWeights::new(3, 5, 32, vec![0; 5], vec![1.0; 3]).is_err());
        assert!(PackedI2sWeights::new(3, 5, 32, vec![0; 6], vec![1.0; 2]).is_err());
        assert!(PackedI2sWeights::new(3, 5, 64, vec![0; 6], vec![1.0; 3]).is_err());
    }

    #[test]
    fn packed_qk256_weights_validate_strict_gguf_no_scale_layout() {
        let weights = qk256_fixture();

        assert_eq!(weights.output_features, 3);
        assert_eq!(weights.input_features, 300);
        assert_eq!(weights.blocks_per_output(), 2);
        assert_eq!(weights.row_stride_bytes, 128);
        assert_eq!(weights.expected_row_stride_bytes().unwrap(), 128);
        assert_eq!(weights.expected_packed_bytes().unwrap(), 384);
        assert_eq!(weights.scale_bytes_len(), 0);
        assert!(weights.packed_at_load);
        assert_eq!(weights.shape().unwrap().dims, vec![3, 300]);
    }

    #[test]
    fn packed_qk256_weights_reject_non_strict_gguf_layouts() {
        assert!(PackedQk256Weights::from_strict_gguf_no_scale(0, 300, 128, vec![0; 384]).is_err());
        assert!(PackedQk256Weights::from_strict_gguf_no_scale(3, 0, 128, vec![0; 384]).is_err());
        assert!(PackedQk256Weights::from_strict_gguf_no_scale(3, 300, 64, vec![0; 192]).is_err());
        assert!(PackedQk256Weights::from_strict_gguf_no_scale(3, 300, 128, vec![0; 383]).is_err());
    }

    #[test]
    fn upload_weight_once_reuses_existing_handle() {
        let mut context = test_context();
        let shape = CudaTensorShape::matrix(2, 8).unwrap();
        let first = context
            .upload_weight_once(
                "layers.0.feed_forward.w1",
                shape.clone(),
                CudaBitnetKernelFamily::I2s,
                &[1, 2, 3, 4],
                &[7, 8],
            )
            .unwrap();
        let second = context
            .upload_weight_once(
                "layers.0.feed_forward.w1",
                shape,
                CudaBitnetKernelFamily::I2s,
                &[1, 2, 3, 4],
                &[7, 8],
            )
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(context.weight_cache().len(), 1);
        assert_eq!(context.stats().weight_uploads, 1);
        assert!(context.weight_handle("layers.0.feed_forward.w1").is_some());
    }

    #[test]
    fn upload_weight_once_rejects_metadata_mismatch() {
        let mut context = test_context();
        context
            .upload_weight_once(
                "layers.0.attention.wq",
                CudaTensorShape::matrix(2, 8).unwrap(),
                CudaBitnetKernelFamily::Qk256,
                &[1, 2, 3, 4],
                &[],
            )
            .unwrap();

        let result = context.upload_weight_once(
            "layers.0.attention.wq",
            CudaTensorShape::matrix(2, 16).unwrap(),
            CudaBitnetKernelFamily::Qk256,
            &[1, 2, 3, 4],
            &[],
        );

        assert!(result.is_err());
    }

    #[test]
    fn upload_i2s_weights_reuses_handle_without_second_upload() {
        let mut context = test_context();
        let weights = i2s_fixture();
        let first = context.upload_i2s_weights("layers.0.feed_forward.w1", &weights).unwrap();
        let second = context.upload_i2s_weights("layers.0.feed_forward.w1", &weights).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.kernel_family, CudaBitnetKernelFamily::I2s);
        assert_eq!(first.block_size, Some(32));
        assert_eq!(first.packed_bytes, weights.packed_weights.len());
        assert_eq!(first.scale_bytes, weights.scale_bytes_len());
        assert!(first.packed_at_load);
        assert!(first.uploaded_once);
        assert_eq!(context.stats().weight_uploads, 1);
        assert_eq!(
            context.stats().weight_upload_bytes,
            u64::try_from(weights.packed_weights.len() + weights.scale_bytes_len()).unwrap()
        );
    }

    #[test]
    fn upload_i2s_weights_rejects_cached_metadata_mismatch() {
        let mut context = test_context();
        let weights = i2s_fixture();
        context.upload_i2s_weights("layers.0.feed_forward.w1", &weights).unwrap();
        let different = PackedI2sWeights::new(4, 5, 32, vec![0; 8], vec![1.0; 4]).unwrap();

        assert!(context.upload_i2s_weights("layers.0.feed_forward.w1", &different).is_err());
    }

    #[test]
    fn upload_qk256_weights_reuses_handle_without_second_upload() {
        let mut context = test_context();
        let weights = qk256_fixture();
        let first = context.upload_qk256_weights("layers.0.attention.wq", &weights).unwrap();
        let second = context.upload_qk256_weights("layers.0.attention.wq", &weights).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.kernel_family, CudaBitnetKernelFamily::Qk256);
        assert_eq!(first.block_size, Some(QK256_BLOCK_COLS));
        assert_eq!(first.packed_bytes, weights.packed_weights.len());
        assert_eq!(first.scale_bytes, 0);
        assert!(first.packed_at_load);
        assert!(first.uploaded_once);
        assert_eq!(context.stats().weight_uploads, 1);
        assert_eq!(
            context.stats().weight_upload_bytes,
            u64::try_from(weights.packed_weights.len()).unwrap()
        );
    }

    #[test]
    fn upload_qk256_weights_rejects_cached_metadata_mismatch() {
        let mut context = test_context();
        let weights = qk256_fixture();
        context.upload_qk256_weights("layers.0.attention.wq", &weights).unwrap();
        let different =
            PackedQk256Weights::from_strict_gguf_no_scale(4, 300, 128, vec![0xbb; 4 * 128])
                .unwrap();

        assert!(context.upload_qk256_weights("layers.0.attention.wq", &different).is_err());
    }

    #[test]
    fn i2s_gemv_rejects_wrong_family_before_launch() {
        let mut context = test_context();
        let qk256 = qk256_fixture();
        let handle = context.upload_qk256_weights("layers.0.attention.wq", &qk256).unwrap();
        let mut output = vec![0.0; 3];
        let mut stats = CudaBitnetKernelInvocationStats::default();

        let result = context.i2s_gemv(&handle, &[1.0; 5], &mut output, &mut stats);

        assert!(result.is_err());
        assert_eq!(stats.invocations, 0);
        assert_eq!(stats.fallback_invocations, 0);
    }

    #[test]
    fn activation_workspace_grows_then_reuses_capacity() {
        let mut context = test_context();
        context.ensure_activation_workspace(128, 256, 64).unwrap();
        context.ensure_activation_workspace(64, 128, 0).unwrap();

        assert_eq!(context.workspace().activation_bytes, 128);
        assert_eq!(context.workspace().output_bytes, 256);
        assert_eq!(context.workspace().scratch_bytes, 64);
        assert_eq!(context.workspace().growth_count, 1);
        assert_eq!(context.workspace().reuse_count, 1);
        assert_eq!(context.stats().workspace_growths, 1);
        assert_eq!(context.stats().workspace_reuses, 1);
    }

    #[test]
    fn receipt_fields_record_upload_once_without_inference_claim() {
        let mut context = test_context();
        context
            .upload_weight_once(
                "layers.0.feed_forward.w2",
                CudaTensorShape::matrix(4, 8).unwrap(),
                CudaBitnetKernelFamily::I2s,
                &[0xaa, 0xbb, 0xcc],
                &[],
            )
            .unwrap();
        context.ensure_activation_workspace(1024, 512, 256).unwrap();
        context.ensure_activation_workspace(512, 256, 128).unwrap();

        let fields = context.receipt_fields();
        assert_eq!(fields.requested_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(fields.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(fields.runtime_api, "cuda");
        assert_eq!(fields.weight_handle_count, 1);
        assert!(fields.packed_at_load);
        assert!(fields.weights_uploaded_once);
        assert!(!fields.per_token_weight_upload);
        assert!(fields.activation_workspace_reused);
        assert!(!fields.full_inference_claim);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn qk256_invocation_stats_accumulate_transfer_bytes_and_event_time() {
        let mut stats = CudaBitnetKernelInvocationStats::new(CUDA_QK256_GEMV_KERNEL_ID);

        stats.record_qk256_gemv(1024, 512, Some(0.25), true, false);
        stats.record_qk256_gemv(2048, 1024, Some(0.5), true, false);

        assert_eq!(stats.kernel_id, CUDA_QK256_GEMV_KERNEL_ID);
        assert_eq!(stats.invocations, 2);
        assert_eq!(stats.kernel_launches, 2);
        assert_eq!(stats.fallback_invocations, 0);
        assert_eq!(stats.host_to_device_bytes, 3072);
        assert_eq!(stats.device_to_host_bytes, 1536);
        assert_eq!(stats.kernel_time_ms, Some(0.75));
        assert!(stats.weights_uploaded_once);
        assert!(!stats.per_token_weight_upload);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn i2s_gemv_reports_unavailable_without_cuda_feature() {
        let mut context = test_context();
        let weights = i2s_fixture();
        let handle = context.upload_i2s_weights("layers.0.feed_forward.w1", &weights).unwrap();
        let mut output = vec![0.0; weights.output_features];
        let mut stats = CudaBitnetKernelInvocationStats::default();

        let result = context.i2s_gemv(&handle, &[1.0; 5], &mut output, &mut stats);

        assert!(result.is_err());
        assert_eq!(stats.invocations, 0);
        assert_eq!(stats.fallback_invocations, 0);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_context_creation_reports_unavailable_without_cuda_feature() {
        assert!(CudaBitnetContext::new(0).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn i2s_gemv_requires_cuda_backed_context_with_cuda_feature() {
        let mut context = test_context();
        let weights = i2s_fixture();
        let handle = context.upload_i2s_weights("layers.0.feed_forward.w1", &weights).unwrap();
        let mut output = vec![0.0; weights.output_features];
        let mut stats = CudaBitnetKernelInvocationStats::default();

        let result = context.i2s_gemv(&handle, &[1.0; 5], &mut output, &mut stats);

        assert!(result.is_err());
        assert_eq!(stats.invocations, 0);
        assert_eq!(stats.fallback_invocations, 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn live_i2s_gemv_matches_cpu_reference_when_enabled() {
        if std::env::var("BITNET_RUN_CUDA_BITNET_I2S_GEMV").as_deref() != Ok("1") {
            eprintln!("skipping live CUDA BitNet I2S GEMV; set BITNET_RUN_CUDA_BITNET_I2S_GEMV=1");
            return;
        }

        let device_index = std::env::var("BITNET_RTX5070TI_CUDA_DEVICE_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let weights = i2s_fixture();
        let activation = vec![0.5, -1.0, 2.0, 3.0, -0.25];
        let mut expected = vec![0.0; weights.output_features];
        let config = super::super::quantized_matmul::I2sMatmulConfig::for_shape(
            1,
            weights.output_features,
            weights.input_features,
            weights.block_size,
        )
        .unwrap();
        super::super::quantized_matmul::i2s_matmul_cpu(
            &activation,
            &weights.packed_weights,
            &weights.scales,
            &mut expected,
            &config,
        )
        .unwrap();

        let mut context = CudaBitnetContext::new(device_index).unwrap();
        let handle = context.upload_i2s_weights("layers.0.feed_forward.w1", &weights).unwrap();
        let mut actual = vec![0.0; weights.output_features];
        let mut stats = CudaBitnetKernelInvocationStats::default();

        context.i2s_gemv(&handle, &activation, &mut actual, &mut stats).unwrap();

        for (expected, actual) in expected.iter().zip(&actual) {
            assert!((expected - actual).abs() <= f32::EPSILON);
        }
        assert_eq!(stats.kernel_id, CUDA_BITNET_I2S_GEMV_KERNEL_ID);
        assert_eq!(stats.invocations, 1);
        assert_eq!(stats.kernel_launches, 1);
        assert_eq!(stats.fallback_invocations, 0);
        assert!(stats.weights_uploaded_once);
        assert!(!stats.per_token_weight_upload);
    }
}
