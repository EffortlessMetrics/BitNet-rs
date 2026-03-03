//! Alignment utilities for SIMD and cache-line aligned allocations.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// Supported alignment levels for memory allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alignment {
    /// 16-byte alignment (SSE).
    Align16 = 16,
    /// 32-byte alignment (AVX2).
    Align32 = 32,
    /// 64-byte alignment (AVX-512 / cache line).
    Align64 = 64,
}

impl Alignment {
    /// Returns the alignment value in bytes.
    #[must_use]
    pub const fn as_bytes(self) -> usize {
        self as usize
    }

    /// Returns `true` if `addr` is aligned to this level.
    #[must_use]
    pub const fn is_aligned(self, addr: usize) -> bool {
        addr.is_multiple_of(self.as_bytes())
    }
}

/// An owned, aligned heap allocation.
///
/// Deallocates the backing memory on drop.
pub struct AlignedAlloc {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedAlloc {
    /// Allocates `size` bytes with the given alignment.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails (OOM) or if `size` is zero.
    #[must_use]
    pub fn new(size: usize, alignment: Alignment) -> Self {
        assert!(size > 0, "allocation size must be non-zero");
        let layout = Layout::from_size_align(size, alignment.as_bytes()).expect("invalid layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");
        Self { ptr, layout }
    }

    /// Returns a pointer to the allocation.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the allocation.
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the usable size in bytes.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns the alignment in bytes.
    #[must_use]
    pub const fn alignment(&self) -> usize {
        self.layout.align()
    }

    /// Returns the allocation as a byte slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: pointer is valid for `layout.size()` bytes and is initialised (zeroed).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    /// Returns the allocation as a mutable byte slice.
    #[must_use]
    pub const fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: pointer is valid, unique, and initialised.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedAlloc {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: The allocation is exclusively owned and the raw pointer is not aliased.
unsafe impl Send for AlignedAlloc {}
// SAFETY: &AlignedAlloc only exposes shared (&[u8]) access.
unsafe impl Sync for AlignedAlloc {}
