use bitnet_crossval::backend::CppBackend;
use bitnet_crossval::token_parity::{TokenParityError, format_token_mismatch_error};

fn main() {
    println!("\n=== Testing BitNet Backend Error Message ===\n");

    let bitnet_error = TokenParityError {
        rust_tokens: vec![128000, 128000, 1229, 374],
        cpp_tokens: vec![128000, 1229, 374],
        first_diff_index: 1,
        prompt: "What is 2+2?".to_string(),
        backend: CppBackend::BitNet,
    };

    println!("{}", format_token_mismatch_error(&bitnet_error));

    println!("\n=== Testing LLaMA Backend Error Message ===\n");

    let llama_error = TokenParityError {
        rust_tokens: vec![128000, 1229, 374],
        cpp_tokens: vec![128000, 1229, 891, 374],
        first_diff_index: 2,
        prompt: "Capital of France?".to_string(),
        backend: CppBackend::Llama,
    };

    println!("{}", format_token_mismatch_error(&llama_error));

    // Test utility methods
    println!("\n=== Testing Utility Methods ===\n");
    println!("BitNet setup command: {}", CppBackend::BitNet.setup_command());
    println!("BitNet required libs: {:?}", CppBackend::BitNet.required_libs());
    println!("\nLLaMA setup command: {}", CppBackend::Llama.setup_command());
    println!("LLaMA required libs: {:?}", CppBackend::Llama.required_libs());
}
