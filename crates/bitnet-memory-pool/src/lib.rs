//! CPU memory pool allocator optimized for neural network inference workloads.
//!
//! Provides arena, slab, and size-class pool allocators with:
//! - RAII-based allocation with automatic return to pool on drop
//! - Cache-line and SIMD-aligned allocations (16/32/64 byte)
//! - Thread-safe operation via `Mutex`
//! - Memory statistics tracking (allocated, freed, peak, fragmentation)

mod align;
mod arena;
mod pool;
mod slab;
mod stats;

pub use align::{AlignedAlloc, Alignment};
pub use arena::ArenaAllocator;
pub use pool::{MemoryPool, PoolGuard};
pub use slab::SlabAllocator;
pub use stats::PoolStats;

#[cfg(test)]
mod tests;
