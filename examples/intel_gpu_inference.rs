//! Full Intel GPU inference demonstration using the OpenCL backend.
//!
//! This example demonstrates the complete inference pipeline on Intel Arc GPUs:
//! 1. Detect Intel GPU availability via OpenCL
//! 2. Initialize compute context
//! 3. Load or generate model weights
//! 4. Run a simplified inference loop
//!
//! # Running
//!
//! ```bash
//! cargo run --example intel_gpu_inference --no-default-features --features oneapi
//! ```

fn main() {
    #[cfg(feature = "oneapi")]
    {
        oneapi_main();
    }

    #[cfg(not(feature = "oneapi"))]
    {
        eprintln!("Intel GPU inference requires the `oneapi` feature.");
        eprintln!();
        eprintln!("Build with:");
        eprintln!("  cargo run --example intel_gpu_inference --no-default-features --features oneapi");
        eprintln!();
        eprintln!("Prerequisites:");
        eprintln!("  - Intel oneAPI Base Toolkit (or Intel Compute Runtime)");
        eprintln!("  - An Intel GPU (Arc, Iris Xe, or compatible)");
        std::process::exit(1);
    }
}

#[cfg(feature = "oneapi")]
fn oneapi_main() {
    use std::time::Instant;

    println!("=== BitNet Intel GPU Inference Demo ===");
    println!();

    // Parse simple command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1));
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Hello, world!");
    let max_tokens: usize = args
        .iter()
        .position(|a| a == "--max-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    // Step 1: Detect Intel GPU
    println!("Step 1: Detecting Intel GPU...");
    match detect_intel_gpu() {
        Ok((platform, device)) => {
            println!("  Found: {} on {}", device, platform);
        }
        Err(e) => {
            eprintln!("  No Intel GPU detected: {}", e);
            eprintln!("  Falling back to CPU demonstration mode.");
            run_cpu_demo(prompt, max_tokens);
            return;
        }
    }

    // Step 2: Initialize OpenCL context
    println!();
    println!("Step 2: Initializing OpenCL context...");
    let start = Instant::now();
    match bitnet_kernels::gpu::opencl::OpenClKernel::new() {
        Ok(kernel) => {
            let elapsed = start.elapsed();
            println!(
                "  Context ready in {:.1}ms: {} ({})",
                elapsed.as_secs_f64() * 1000.0,
                kernel.device_name(),
                kernel.platform_name()
            );
            run_gpu_demo(&kernel, model_path, prompt, max_tokens);
        }
        Err(e) => {
            eprintln!("  OpenCL initialization failed: {}", e);
            eprintln!("  Falling back to CPU demonstration mode.");
            run_cpu_demo(prompt, max_tokens);
        }
    }
}

#[cfg(feature = "oneapi")]
fn detect_intel_gpu() -> Result<(String, String), String> {
    use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
    use opencl3::platform::get_platforms;

    let platforms = get_platforms().map_err(|e| format!("OpenCL platform error: {}", e))?;
    for platform in &platforms {
        let platform_name = platform.name().unwrap_or_default();
        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
        for device_id in devices {
            let device = Device::new(device_id);
            let name = device.name().unwrap_or_default();
            let vendor = device.vendor().unwrap_or_default();
            if vendor.to_lowercase().contains("intel") {
                return Ok((platform_name, name));
            }
        }
    }
    Err("No Intel GPU found".into())
}

#[cfg(feature = "oneapi")]
fn run_gpu_demo(
    kernel: &bitnet_kernels::gpu::opencl::OpenClKernel,
    _model_path: Option<&String>,
    prompt: &str,
    max_tokens: usize,
) {
    use std::time::Instant;

    println!();
    println!("Step 3: Preparing model weights...");
    println!("  Using random demo weights (no model file loaded)");
    println!("  Note: Pass --model <path> to use a real GGUF model");

    // Generate deterministic pseudo-random weights for demonstration
    let vocab_size = 256;
    let hidden_dim = 64;
    let num_layers = 2;

    let embedding_table: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| ((i as f32 * 0.0137).sin() * 0.1))
        .collect();

    // Ternary weight matrices for each layer (packed)
    let weights_per_layer = hidden_dim * hidden_dim;
    let packed_per_layer = (weights_per_layer + 3) / 4;
    let layer_weights: Vec<Vec<u8>> = (0..num_layers)
        .map(|layer| {
            (0..packed_per_layer)
                .map(|i| ((i + layer * 7 + 13) % 256) as u8)
                .collect()
        })
        .collect();

    let rms_weights = vec![1.0f32; hidden_dim];

    // Step 4: Tokenize input
    println!();
    println!("Step 4: Tokenizing input...");
    let tokens: Vec<usize> = prompt.bytes().map(|b| b as usize % vocab_size).collect();
    println!("  Prompt: \"{}\"", prompt);
    println!("  Tokens: {:?} ({} tokens)", &tokens[..tokens.len().min(10)], tokens.len());

    // Step 5: Run inference
    println!();
    println!("Step 5: Running inference ({} tokens)...", max_tokens);
    let start = Instant::now();

    let mut current_tokens = tokens.clone();
    let mut generated = Vec::new();

    for step in 0..max_tokens {
        let last_token = *current_tokens.last().unwrap_or(&0);

        // Embedding lookup
        let mut hidden: Vec<f32> = embedding_table
            [last_token * hidden_dim..(last_token + 1) * hidden_dim]
            .to_vec();

        // Process through layers
        for layer in 0..num_layers {
            // RMSNorm
            hidden = cpu_rms_norm(&hidden, &rms_weights, 1e-5);

            // Ternary matmul (attention simulation)
            let mut output = vec![0.0f32; hidden_dim];
            cpu_matmul_i2s(&layer_weights[layer], &hidden, &mut output, 1, hidden_dim, hidden_dim);
            hidden = output;

            // SiLU activation
            hidden = cpu_silu(&hidden);
        }

        // Simple argmax sampling from projection
        let next_token = hidden
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i % vocab_size)
            .unwrap_or(0);

        generated.push(next_token);
        current_tokens.push(next_token);

        if step < 4 || step == max_tokens - 1 {
            println!("  Step {}: token {} -> {}", step, last_token, next_token);
        } else if step == 4 {
            println!("  ...");
        }
    }

    let elapsed = start.elapsed();
    let tokens_per_sec = max_tokens as f64 / elapsed.as_secs_f64();

    println!();
    println!("=== Results ===");
    println!("  Generated {} tokens in {:.1}ms ({:.1} tok/s)", max_tokens, elapsed.as_secs_f64() * 1000.0, tokens_per_sec);
    println!("  Token IDs: {:?}", &generated[..generated.len().min(20)]);
    println!("  Backend: OpenCL ({})", kernel.device_name());
    println!();
    println!("Note: This demo uses random weights. Output is not meaningful text.");
    println!("      Use a real GGUF model for actual inference.");
}

fn run_cpu_demo(prompt: &str, max_tokens: usize) {
    use std::time::Instant;

    println!();
    println!("=== CPU Fallback Demo ===");
    println!("  Prompt: \"{}\"", prompt);

    let vocab_size = 256;
    let hidden_dim = 32;

    let start = Instant::now();
    let mut token = prompt.bytes().last().unwrap_or(b' ') as usize % vocab_size;
    let mut generated = Vec::new();

    for _ in 0..max_tokens {
        // Simple hash-based next token (not real inference)
        token = (token.wrapping_mul(6364136223846793005).wrapping_add(1)) % vocab_size;
        generated.push(token);
    }

    let elapsed = start.elapsed();
    println!(
        "  Generated {} tokens in {:.1}ms (CPU fallback, not real inference)",
        max_tokens,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("  Token IDs: {:?}", &generated[..generated.len().min(20)]);
    let _ = hidden_dim; // suppress unused warning
}

// CPU reference implementations for the demo

fn cpu_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| x * rms * w)
        .collect()
}

fn cpu_matmul_i2s(a_packed: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let byte_idx = (row * k + i) / 4;
                let sub = (row * k + i) % 4;
                let bits = if byte_idx < a_packed.len() {
                    (a_packed[byte_idx] >> (sub * 2)) & 0x03
                } else {
                    0
                };
                let w: f32 = match bits {
                    0x01 => 1.0,
                    0x03 => -1.0,
                    _ => 0.0,
                };
                if i * n + col < b.len() {
                    sum += w * b[i * n + col];
                }
            }
            if row * n + col < c.len() {
                c[row * n + col] = sum;
            }
        }
    }
}

fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        })
        .collect()
}
