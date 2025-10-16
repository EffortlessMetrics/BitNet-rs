//! CLI command implementations

#[cfg(feature = "cli-bench")]
pub mod benchmark;
pub mod chat;
pub mod convert;
#[allow(dead_code)]
pub mod eval;
pub mod inference;
pub mod inspect;
pub mod serve;
pub mod template_util;

#[cfg(feature = "cli-bench")]
pub use benchmark::BenchmarkCommand;
pub use convert::ConvertCommand;
pub use inference::InferenceCommand;
pub use inspect::InspectCommand;
pub use serve::ServeCommand;
