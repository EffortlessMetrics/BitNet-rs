//! OpenCL 3.0 Unified Shared Memory (USM) support.
//!
//! Detects USM capability at runtime via `CL_DEVICE_SVM_CAPABILITIES` and
//! provides shared memory allocation wrappers (`clSVMAlloc`/`clSVMFree`).
//! When USM is unavailable, falls back to explicit buffer copy paths.

use std::alloc::Layout;
use std::fmt;
use std::ptr::NonNull;

// ---------------------------------------------------------------------------
// SVM / USM capability flags (from OpenCL spec CL_DEVICE_SVM_CAPABILITIES)
// ---------------------------------------------------------------------------

/// Lightweight bitflags implementation (no external dependency needed).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SvmCapabilities(u64);

impl SvmCapabilities {
    pub const COARSE_GRAIN_BUFFER: Self = Self(1 << 0);
    pub const FINE_GRAIN_BUFFER: Self = Self(1 << 1);
    pub const FINE_GRAIN_SYSTEM: Self = Self(1 << 2);
    pub const ATOMICS: Self = Self(1 << 3);
    pub const NONE: Self = Self(0);

    /// Build from raw OpenCL bitfield value.
    #[inline]
    pub const fn from_raw(bits: u64) -> Self {
        Self(bits)
    }

    /// Return the raw bitfield.
    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Check whether `flag` is set.
    #[inline]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }

    /// True when **any** SVM/USM capability is reported.
    #[inline]
    pub const fn supports_usm(self) -> bool {
        self.0 != 0
    }

    /// True when zero-copy host access is possible (fine-grain buffer or system).
    #[inline]
    pub const fn supports_zero_copy(self) -> bool {
        self.contains(Self::FINE_GRAIN_BUFFER) || self.contains(Self::FINE_GRAIN_SYSTEM)
    }
}

impl fmt::Debug for SvmCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.contains(Self::COARSE_GRAIN_BUFFER) {
            parts.push("COARSE_GRAIN_BUFFER");
        }
        if self.contains(Self::FINE_GRAIN_BUFFER) {
            parts.push("FINE_GRAIN_BUFFER");
        }
        if self.contains(Self::FINE_GRAIN_SYSTEM) {
            parts.push("FINE_GRAIN_SYSTEM");
        }
        if self.contains(Self::ATOMICS) {
            parts.push("ATOMICS");
        }
        if parts.is_empty() {
            write!(f, "SvmCapabilities(NONE)")
        } else {
            write!(f, "SvmCapabilities({})", parts.join(" | "))
        }
    }
}

impl fmt::Display for SvmCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Transfer mode selection
// ---------------------------------------------------------------------------

/// Transfer strategy between host and device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Unified/shared address space – no explicit copy needed.
    ZeroCopy,
    /// Explicit `clEnqueueReadBuffer` / `clEnqueueWriteBuffer` round-trips.
    ExplicitCopy,
}

impl TransferMode {
    /// Select the best transfer mode for the given capabilities.
    pub fn select(caps: SvmCapabilities) -> Self {
        if caps.supports_zero_copy() {
            Self::ZeroCopy
        } else {
            Self::ExplicitCopy
        }
    }
}

// ---------------------------------------------------------------------------
// USM allocator
// ---------------------------------------------------------------------------

/// Error type for USM allocation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsmError {
    /// The device does not support SVM / USM at all.
    UsmNotSupported,
    /// `clSVMAlloc` returned null (out of memory or invalid alignment).
    AllocationFailed {
        requested_bytes: usize,
        alignment: usize,
    },
    /// Attempted to free a null pointer.
    FreeNull,
    /// The provided layout has zero size.
    ZeroSizedAllocation,
}

impl fmt::Display for UsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UsmNotSupported => write!(f, "device does not support SVM/USM"),
            Self::AllocationFailed {
                requested_bytes,
                alignment,
            } => {
                write!(
                    f,
                    "USM allocation failed: {requested_bytes} bytes, alignment {alignment}"
                )
            }
            Self::FreeNull => write!(f, "attempted to free a null USM pointer"),
            Self::ZeroSizedAllocation => write!(f, "zero-sized USM allocation requested"),
        }
    }
}

impl std::error::Error for UsmError {}

/// Handle to a USM (SVM) allocation.
///
/// Wraps the raw pointer returned by `clSVMAlloc` together with the layout
/// so that [`UsmAllocator::free`] can release it correctly.
#[derive(Debug)]
pub struct UsmAllocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl UsmAllocation {
    /// Raw pointer to the allocation (never null).
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Layout used for this allocation.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
    }
}

/// Zero-copy USM allocator wrapping `clSVMAlloc` / `clSVMFree`.
///
/// When the device lacks USM support, [`UsmAllocator::new`] succeeds but
/// [`UsmAllocator::alloc`] returns [`UsmError::UsmNotSupported`].  
/// Callers should check [`UsmAllocator::transfer_mode`] and use the explicit
/// buffer-copy path when it reports [`TransferMode::ExplicitCopy`].
#[derive(Debug)]
pub struct UsmAllocator {
    capabilities: SvmCapabilities,
    transfer_mode: TransferMode,
    /// Running total of bytes currently allocated (for diagnostics).
    allocated_bytes: std::sync::atomic::AtomicUsize,
}

impl UsmAllocator {
    /// Create a new allocator for a device with the given SVM capabilities.
    pub fn new(capabilities: SvmCapabilities) -> Self {
        let transfer_mode = TransferMode::select(capabilities);
        log::info!(
            "USM allocator initialised: caps={capabilities:?}, mode={transfer_mode:?}"
        );
        Self {
            capabilities,
            transfer_mode,
            allocated_bytes: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Detected SVM capabilities.
    #[inline]
    pub fn capabilities(&self) -> SvmCapabilities {
        self.capabilities
    }

    /// Selected transfer mode.
    #[inline]
    pub fn transfer_mode(&self) -> TransferMode {
        self.transfer_mode
    }

    /// Total bytes currently allocated through this allocator.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Allocate `layout.size()` bytes of shared memory.
    ///
    /// Returns [`UsmError::UsmNotSupported`] when the device lacks any SVM
    /// capability, and [`UsmError::AllocationFailed`] if the underlying
    /// `clSVMAlloc` returns null (simulated here as a host allocation).
    pub fn alloc(&self, layout: Layout) -> Result<UsmAllocation, UsmError> {
        if layout.size() == 0 {
            return Err(UsmError::ZeroSizedAllocation);
        }
        if !self.capabilities.supports_usm() {
            return Err(UsmError::UsmNotSupported);
        }

        // In a real implementation this calls clSVMAlloc(context, flags, size, alignment).
        // We simulate with std::alloc to keep the crate dependency-free for now.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(UsmError::AllocationFailed {
            requested_bytes: layout.size(),
            alignment: layout.align(),
        })?;

        self.allocated_bytes
            .fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);

        Ok(UsmAllocation { ptr, layout })
    }

    /// Free a previous USM allocation.
    ///
    /// # Safety
    ///
    /// `allocation` must have been returned by [`Self::alloc`] on the same
    /// allocator, and must not be used after this call.
    pub unsafe fn free(&self, allocation: UsmAllocation) -> Result<(), UsmError> {
        // In a real implementation this calls clSVMFree(context, ptr).
        unsafe {
            std::alloc::dealloc(allocation.ptr.as_ptr(), allocation.layout);
        }
        self.allocated_bytes
            .fetch_sub(allocation.layout.size(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Explicit buffer-copy fallback
// ---------------------------------------------------------------------------

/// Explicit host↔device buffer copy (fallback when USM is unavailable).
///
/// In a full OpenCL integration this wraps `clEnqueueWriteBuffer` /
/// `clEnqueueReadBuffer`.  The simulation here copies between two host
/// buffers to exercise the API surface.
#[derive(Debug)]
pub struct ExplicitBuffer {
    data: Vec<u8>,
}

impl ExplicitBuffer {
    /// Create a device-side buffer of `size` bytes (zero-initialised).
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Number of bytes in the buffer.
    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Write host data into the device buffer (simulated `clEnqueueWriteBuffer`).
    ///
    /// Returns the number of bytes copied (min of source and buffer size).
    pub fn write_from_host(&mut self, host_data: &[u8]) -> usize {
        let n = host_data.len().min(self.data.len());
        self.data[..n].copy_from_slice(&host_data[..n]);
        n
    }

    /// Read device buffer into host memory (simulated `clEnqueueReadBuffer`).
    ///
    /// Returns the number of bytes copied (min of dest and buffer size).
    pub fn read_to_host(&self, dest: &mut [u8]) -> usize {
        let n = dest.len().min(self.data.len());
        dest[..n].copy_from_slice(&self.data[..n]);
        n
    }
}

// ---------------------------------------------------------------------------
// Unified data-access path
// ---------------------------------------------------------------------------

/// Unified data path that transparently uses USM zero-copy or explicit copy.
#[derive(Debug)]
pub enum DataPath {
    /// Zero-copy path backed by USM shared allocation.
    ZeroCopy {
        allocator: std::sync::Arc<UsmAllocator>,
    },
    /// Fallback path using explicit buffer copies.
    ExplicitCopy,
}

impl DataPath {
    /// Create the best data path for the given capabilities.
    pub fn new(caps: SvmCapabilities) -> Self {
        if caps.supports_zero_copy() {
            Self::ZeroCopy {
                allocator: std::sync::Arc::new(UsmAllocator::new(caps)),
            }
        } else {
            Self::ExplicitCopy
        }
    }

    /// True when zero-copy path is active.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self, Self::ZeroCopy { .. })
    }

    /// Write `host_data` to a device-visible buffer and return the bytes
    /// that are now accessible from the device side.
    ///
    /// - **ZeroCopy**: allocates shared memory, copies once, pointer is
    ///   accessible from both host and device.
    /// - **ExplicitCopy**: writes into an explicit buffer that must be
    ///   enqueued for transfer.
    pub fn upload(&self, host_data: &[u8]) -> Result<Vec<u8>, UsmError> {
        match self {
            Self::ZeroCopy { allocator } => {
                let layout =
                    Layout::from_size_align(host_data.len(), 64).map_err(|_| {
                        UsmError::AllocationFailed {
                            requested_bytes: host_data.len(),
                            alignment: 64,
                        }
                    })?;
                let alloc = allocator.alloc(layout)?;
                // Copy data into USM region (accessible on both sides).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host_data.as_ptr(),
                        alloc.as_ptr(),
                        host_data.len(),
                    );
                }
                let result = unsafe {
                    std::slice::from_raw_parts(alloc.as_ptr(), host_data.len()).to_vec()
                };
                unsafe { allocator.free(alloc)? };
                Ok(result)
            }
            Self::ExplicitCopy => {
                let mut buf = ExplicitBuffer::new(host_data.len());
                buf.write_from_host(host_data);
                let mut out = vec![0u8; host_data.len()];
                buf.read_to_host(&mut out);
                Ok(out)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime capability detection
// ---------------------------------------------------------------------------

/// Detect SVM capabilities for the device.
///
/// In a full implementation this calls
/// `clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, ...)`.
/// Here we accept the raw bitfield directly for testing.
pub fn detect_usm_capabilities(raw_device_caps: u64) -> SvmCapabilities {
    SvmCapabilities::from_raw(raw_device_caps)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svm_capabilities_none() {
        let caps = SvmCapabilities::NONE;
        assert!(!caps.supports_usm());
        assert!(!caps.supports_zero_copy());
        assert_eq!(caps.bits(), 0);
    }

    #[test]
    fn test_svm_capabilities_coarse_grain_only() {
        let caps = SvmCapabilities::COARSE_GRAIN_BUFFER;
        assert!(caps.supports_usm());
        assert!(!caps.supports_zero_copy());
        assert!(caps.contains(SvmCapabilities::COARSE_GRAIN_BUFFER));
        assert!(!caps.contains(SvmCapabilities::FINE_GRAIN_BUFFER));
    }

    #[test]
    fn test_svm_capabilities_fine_grain_enables_zero_copy() {
        let caps = SvmCapabilities::from_raw(
            SvmCapabilities::FINE_GRAIN_BUFFER.bits()
                | SvmCapabilities::COARSE_GRAIN_BUFFER.bits(),
        );
        assert!(caps.supports_usm());
        assert!(caps.supports_zero_copy());
    }

    #[test]
    fn test_svm_capabilities_system_fine_grain_enables_zero_copy() {
        let caps = SvmCapabilities::FINE_GRAIN_SYSTEM;
        assert!(caps.supports_zero_copy());
    }

    #[test]
    fn test_transfer_mode_selection() {
        assert_eq!(
            TransferMode::select(SvmCapabilities::NONE),
            TransferMode::ExplicitCopy
        );
        assert_eq!(
            TransferMode::select(SvmCapabilities::COARSE_GRAIN_BUFFER),
            TransferMode::ExplicitCopy
        );
        assert_eq!(
            TransferMode::select(SvmCapabilities::FINE_GRAIN_BUFFER),
            TransferMode::ZeroCopy
        );
        assert_eq!(
            TransferMode::select(SvmCapabilities::FINE_GRAIN_SYSTEM),
            TransferMode::ZeroCopy
        );
    }

    #[test]
    fn test_usm_allocator_alloc_and_free() {
        let caps = SvmCapabilities::FINE_GRAIN_BUFFER;
        let allocator = UsmAllocator::new(caps);
        assert_eq!(allocator.allocated_bytes(), 0);

        let layout = Layout::from_size_align(1024, 64).unwrap();
        let alloc = allocator.alloc(layout).unwrap();
        assert_eq!(alloc.size(), 1024);
        assert_eq!(allocator.allocated_bytes(), 1024);

        unsafe { allocator.free(alloc).unwrap() };
        assert_eq!(allocator.allocated_bytes(), 0);
    }

    #[test]
    fn test_usm_allocator_rejects_unsupported_device() {
        let allocator = UsmAllocator::new(SvmCapabilities::NONE);
        let layout = Layout::from_size_align(256, 8).unwrap();
        assert!(matches!(
            allocator.alloc(layout),
            Err(UsmError::UsmNotSupported)
        ));
    }

    #[test]
    fn test_usm_allocator_rejects_zero_size() {
        let allocator = UsmAllocator::new(SvmCapabilities::FINE_GRAIN_BUFFER);
        let layout = Layout::from_size_align(0, 1).unwrap();
        assert!(matches!(
            allocator.alloc(layout),
            Err(UsmError::ZeroSizedAllocation)
        ));
    }

    #[test]
    fn test_explicit_buffer_roundtrip() {
        let src = b"hello, explicit copy path!";
        let mut buf = ExplicitBuffer::new(src.len());
        let written = buf.write_from_host(src);
        assert_eq!(written, src.len());

        let mut dest = vec![0u8; src.len()];
        let read = buf.read_to_host(&mut dest);
        assert_eq!(read, src.len());
        assert_eq!(&dest, src);
    }

    #[test]
    fn test_explicit_buffer_partial_write() {
        let mut buf = ExplicitBuffer::new(4);
        let written = buf.write_from_host(b"ab");
        assert_eq!(written, 2);
    }

    #[test]
    fn test_data_path_zero_copy_roundtrip() {
        let caps = SvmCapabilities::FINE_GRAIN_BUFFER;
        let path = DataPath::new(caps);
        assert!(path.is_zero_copy());

        let data = b"zero-copy test payload";
        let result = path.upload(data).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_data_path_explicit_copy_roundtrip() {
        let path = DataPath::new(SvmCapabilities::NONE);
        assert!(!path.is_zero_copy());

        let data = b"explicit copy test payload";
        let result = path.upload(data).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_detect_usm_capabilities_from_raw() {
        let raw = 0b0111; // coarse + fine buffer + fine system
        let caps = detect_usm_capabilities(raw);
        assert!(caps.contains(SvmCapabilities::COARSE_GRAIN_BUFFER));
        assert!(caps.contains(SvmCapabilities::FINE_GRAIN_BUFFER));
        assert!(caps.contains(SvmCapabilities::FINE_GRAIN_SYSTEM));
        assert!(!caps.contains(SvmCapabilities::ATOMICS));
        assert!(caps.supports_zero_copy());
    }

    #[test]
    fn test_svm_capabilities_debug_display() {
        let caps = SvmCapabilities::from_raw(
            SvmCapabilities::FINE_GRAIN_BUFFER.bits() | SvmCapabilities::ATOMICS.bits(),
        );
        let dbg = format!("{caps:?}");
        assert!(dbg.contains("FINE_GRAIN_BUFFER"));
        assert!(dbg.contains("ATOMICS"));
        assert!(!dbg.contains("COARSE"));
    }
}
