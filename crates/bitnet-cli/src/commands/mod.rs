//! CLI command implementations

#[cfg(feature = "cli-bench")]
pub mod benchmark;
pub mod chat;
pub mod convert;
#[allow(dead_code)]
pub mod eval;
#[cfg(feature = "full-cli")]
pub mod gpu_info;
pub mod inference;
pub mod inspect;
#[cfg(feature = "cli-bench")]
pub mod kernel_bench;
pub mod serve;
pub mod template_util;

#[cfg(feature = "cli-bench")]
pub use benchmark::BenchmarkCommand;
pub use convert::ConvertCommand;
#[cfg(feature = "full-cli")]
pub use gpu_info::GpuInfoCommand;
pub use inference::InferenceCommand;
pub use inspect::InspectCommand;
#[cfg(feature = "cli-bench")]
pub use kernel_bench::KernelBenchCommand;
pub use serve::ServeCommand;
