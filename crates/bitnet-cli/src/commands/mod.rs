//! CLI command implementations

pub mod answer_corpus;
pub mod answer_parity;
#[cfg(feature = "cli-bench")]
pub mod benchmark;
pub mod chat;
pub mod convert;
#[allow(dead_code)]
pub mod eval;
pub mod inference;
pub mod inspect;
pub mod reference_compare;
pub mod serve;
pub mod template_util;

pub use answer_corpus::AnswerCorpusCommand;
pub use answer_parity::AnswerParityCommand;
#[cfg(feature = "cli-bench")]
pub use benchmark::BenchmarkCommand;
pub use convert::ConvertCommand;
pub use inference::InferenceCommand;
pub use inspect::InspectCommand;
pub use reference_compare::ReferenceCompareCommand;
pub use serve::ServeCommand;
