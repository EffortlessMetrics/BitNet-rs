//! OpenCL abstraction layer for BitNet GPU inference.
//!
//! This crate provides a high-level Rust interface over the `opencl3` crate,
//! targeting Intel Arc GPUs via Intel's Compute Runtime.
//!
//! # Architecture
//!
//! ```text
//! OpenClDevice -> OpenClContext -> OpenClQueue
//!                     |              |
//!               OpenClProgram   OpenClBuffer<T>
//! ```
//!
//! # Example (requires hardware)
//!
//! ```rust,no_run
//! use bitnet_opencl::{OpenClContext, OpenClQueue, OpenClBuffer, AccessMode};
//!
//! let ctx = OpenClContext::new_intel().expect("Intel GPU required");
//! let queue = ctx.create_queue().expect("queue creation");
//! let mut buf = OpenClBuffer::<f32>::new(&ctx, 1024, AccessMode::ReadWrite)
//!     .expect("buffer allocation");
//! ```

pub mod async_transfer;
pub mod buffer;
pub mod buffer_pool;
pub mod context;
pub mod device;
pub mod error;
pub mod pinned;
pub mod program;
pub mod queue;

// Re-export primary API types at crate root.
pub use async_transfer::{async_read, async_write, wait_all, TransferEvent};
pub use buffer::{AccessMode, OpenClBuffer};
pub use buffer_pool::{BufferPool, BufferPoolConfig, PoolStats};
pub use context::OpenClContext;
pub use device::OpenClDevice;
pub use error::OpenClError;
pub use pinned::PinnedBuffer;
pub use program::{OpenClProgram, ProgramCache};
pub use queue::OpenClQueue;
