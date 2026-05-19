#![recursion_limit = "256"]

//! Benchmark receipts for tracking kernel performance over time.

mod error;
mod receipt;
mod validation;

pub use error::ReceiptError;
pub use receipt::{BenchReceipt, ReceiptStore};
pub use validation::{
    validate_dense_gguf_qwen_benchmark_qualification_receipt_file,
    validate_dense_gguf_qwen_benchmark_qualification_receipt_json,
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_file,
    validate_dense_gguf_qwen_cuda_benchmark_baseline_receipt_json,
    validate_dense_gguf_qwen_repeated_comparator_receipt_file,
    validate_dense_gguf_qwen_repeated_comparator_receipt_json,
    validate_qwen3_cuda_repeated_comparator_receipt_file,
    validate_qwen3_cuda_repeated_comparator_receipt_json,
    validate_rtx5070ti_cuda_benchmark_receipt_file, validate_rtx5070ti_cuda_benchmark_receipt_json,
    validate_strict_bitnet_cuda_benchmark_receipt_file,
    validate_strict_bitnet_cuda_benchmark_receipt_json, validate_strict_cpu_benchmark_receipt_file,
    validate_strict_cpu_benchmark_receipt_json,
    validate_strict_cuda_answer_path_benchmark_receipt_file,
    validate_strict_cuda_answer_path_benchmark_receipt_json,
    validate_strict_cuda_benchmark_qualification_receipt_file,
    validate_strict_cuda_benchmark_qualification_receipt_json,
    validate_strict_cuda_repeated_ask_benchmark_receipt_file,
    validate_strict_cuda_repeated_ask_benchmark_receipt_json,
    validate_strict_cuda_warm_session_benchmark_receipt_file,
    validate_strict_cuda_warm_session_benchmark_receipt_json,
};

#[cfg(test)]
mod tests;
