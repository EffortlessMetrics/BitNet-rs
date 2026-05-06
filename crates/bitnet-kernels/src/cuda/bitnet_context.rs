//! Persistent CUDA BitNet context and weight-handle scaffolding.
//!
//! This module owns CUDA lifetime state for the BitNet-specific CUDA path:
//! device identity, one long-lived CUDA context/stream pair, upload-once weight
//! handles, reusable activation workspace metadata, and receipt-friendly stats.
//! It does not route transformer inference or launch BitNet kernels.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, result::device as cu_device, sys::CUdevice_attribute,
};

static NEXT_WEIGHT_ID: AtomicU64 = AtomicU64::new(1);

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
}

#[cfg(feature = "cuda")]
struct CudaDeviceWeightBuffers {
    _packed: CudaSlice<u8>,
    _scales: Option<CudaSlice<u8>>,
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
            self.device_weight_buffers
                .insert(id, CudaDeviceWeightBuffers { _packed: packed, _scales: scales });
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
        && existing.scale_bytes == scale_bytes.len();

    if matches {
        Ok(())
    } else {
        Err(invalid_arguments(format!(
            "CUDA weight '{}' is already cached with different metadata",
            existing.tensor_name
        )))
    }
}

fn invalid_arguments(reason: impl Into<String>) -> bitnet_common::BitNetError {
    KernelError::InvalidArguments { reason: reason.into() }.into()
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

    #[test]
    fn tensor_shape_rejects_empty_or_zero_dims() {
        assert!(CudaTensorShape::new(Vec::<usize>::new()).is_err());
        assert!(CudaTensorShape::new(vec![4, 0]).is_err());
        assert_eq!(CudaTensorShape::matrix(4, 8).unwrap().element_count(), 32);
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
        assert!(fields.weights_uploaded_once);
        assert!(!fields.per_token_weight_upload);
        assert!(fields.activation_workspace_reused);
        assert!(!fields.full_inference_claim);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_context_creation_reports_unavailable_without_cuda_feature() {
        assert!(CudaBitnetContext::new(0).is_err());
    }
}
