//! Basic inference example using BitNet.rs
//!
//! This example demonstrates how to load a BitNet model and perform basic text generation.
//!
//! NOTE: This example is currently outdated and needs to be updated to match the current API.
//! Please use the bitnet-cli for inference instead:
//!   cargo run -p bitnet-cli --features cpu,full-cli -- run --model model.gguf --prompt "Your prompt here"

#[cfg(feature = "examples")]
fn main() {
    eprintln!("This example is currently outdated.");
    eprintln!("Please use the bitnet-cli for inference instead:");
    eprintln!(
        "  cargo run -p bitnet-cli --features cpu,full-cli -- run --model model.gguf --prompt \"Your prompt here\""
    );
    std::process::exit(1);
}

#[cfg(not(feature = "examples"))]
fn main() {}
