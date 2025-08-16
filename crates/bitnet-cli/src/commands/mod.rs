//! CLI command implementations

#[cfg(feature = "cli-bench")]
pub mod benchmark;
pub mod convert;
pub mod inference;
pub mod serve;

#[cfg(feature = "cli-bench")]
pub use benchmark::BenchmarkCommand;
pub use convert::ConvertCommand;
pub use inference::InferenceCommand;
pub use serve::ServeCommand;
