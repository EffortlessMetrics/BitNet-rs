//! Per-token parity ladder comparison and output helpers.

#![allow(dead_code)] // Used by the xtask binary; the shared xtask library compiles it too.

use std::path::Path;

use anyhow::{Result, bail};

use super::CppBackend;

pub(crate) struct LadderRun<'a> {
    pub ladder: &'a str,
    pub positions: usize,
    pub rust_logits: &'a [Vec<f32>],
    pub cpp_logits: &'a [Vec<f32>],
    pub receipt_path: Option<&'a Path>,
    pub model_path: &'a Path,
    pub tokenizer_path: &'a Path,
    pub backend: &'a CppBackend,
    pub formatted_prompt: &'a str,
    pub prompt: &'a str,
    pub format: &'a str,
    pub cos_tol: f32,
    pub compute_mse: bool,
    pub compute_kl: bool,
    pub compute_topk: bool,
    pub verbose: bool,
}

pub(crate) fn run_ladder(args: LadderRun<'_>) -> Result<()> {
    match args.ladder {
        "positions" => run_positions(args),
        "tokens" => bail!("Ladder mode 'tokens' not yet implemented"),
        "masks" => bail!("Ladder mode 'masks' not yet implemented"),
        "first-logit" => bail!("Ladder mode 'first-logit' not yet implemented"),
        "decode" => bail!("Ladder mode 'decode' not yet implemented"),
        _ => bail!("Unknown ladder mode: {}", args.ladder),
    }
}

fn run_positions(args: LadderRun<'_>) -> Result<()> {
    use bitnet_crossval::logits_compare::compare_per_position_logits;

    let effective_positions = args.positions.min(args.rust_logits.len()).min(args.cpp_logits.len());

    if effective_positions < args.rust_logits.len() || effective_positions < args.cpp_logits.len() {
        if args.verbose {
            eprintln!(
                "Limiting comparison to first {} positions (Rust: {}, C++: {})",
                effective_positions,
                args.rust_logits.len(),
                args.cpp_logits.len()
            );
        }
        println!("📊 Comparing first {} positions...", effective_positions);
    } else {
        println!("📊 Comparing logits per position...");
    }

    let rust_logits_slice = &args.rust_logits[..effective_positions];
    let cpp_logits_slice = &args.cpp_logits[..effective_positions];

    if args.positions > effective_positions {
        eprintln!(
            "Warning: --positions {} exceeds available positions {}",
            args.positions, effective_positions
        );
    }

    let divergence = compare_per_position_logits(rust_logits_slice, cpp_logits_slice);

    if args.verbose && (!args.compute_mse || !args.compute_kl || !args.compute_topk) {
        eprintln!("Note: Selective metrics not yet implemented, using all metrics");
    }

    if let Some(receipt_file) = args.receipt_path {
        if args.verbose {
            eprintln!("📝 Generating parity receipt...");
        }
        generate_parity_receipt(
            args.model_path,
            args.backend,
            args.formatted_prompt,
            rust_logits_slice,
            cpp_logits_slice,
            &divergence,
            args.cos_tol,
            args.compute_mse,
            args.compute_kl,
            args.compute_topk,
            receipt_file,
        )?;
        println!("✓ Receipt written to: {}", receipt_file.display());
    }

    output_comparison_results(
        &divergence,
        args.format,
        args.cos_tol,
        args.model_path,
        args.tokenizer_path,
        args.prompt,
    )
}

#[allow(clippy::too_many_arguments)] // Receipt generation needs the comparison context.
fn generate_parity_receipt(
    model_path: &Path,
    backend: &CppBackend,
    formatted_prompt: &str,
    rust_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
    divergence: &bitnet_crossval::logits_compare::LogitsDivergence,
    cos_tol: f32,
    compute_mse: bool,
    compute_kl: bool,
    compute_topk: bool,
    receipt_path: &Path,
) -> Result<()> {
    use bitnet_crossval::metrics::{kl_divergence, max_abs, mse_row, topk_agree, topk_indices};
    use bitnet_crossval::receipt::{ParityReceipt, PositionMetrics, Thresholds};

    let mut receipt =
        ParityReceipt::new(&model_path.display().to_string(), backend.name(), formatted_prompt);

    let mse_threshold = (1.0 - cos_tol) * (1.0 - cos_tol);
    receipt.set_thresholds(Thresholds { mse: mse_threshold, kl: 0.1, topk: 0.8 });

    let n_positions = rust_logits.len().min(cpp_logits.len());
    for pos in 0..n_positions {
        let rust_row = &rust_logits[pos];
        let cpp_row = &cpp_logits[pos];

        let mse = if compute_mse && rust_row.len() == cpp_row.len() {
            mse_row(rust_row, cpp_row)
        } else {
            divergence.per_token_l2_dist[pos] * divergence.per_token_l2_dist[pos]
                / rust_row.len() as f32
        };

        let max_abs_diff = if rust_row.len() == cpp_row.len() {
            max_abs(rust_row, cpp_row)
        } else {
            divergence.max_absolute_diff
        };

        let kl = if compute_kl && rust_row.len() == cpp_row.len() {
            Some(kl_divergence(rust_row, cpp_row))
        } else {
            None
        };

        let topk_agreement = if compute_topk && rust_row.len() == cpp_row.len() {
            let k = 5.min(rust_row.len());
            Some(topk_agree(rust_row, cpp_row, k))
        } else {
            None
        };

        let k = 5.min(rust_row.len());
        let top5_rust = topk_indices(rust_row, k);
        let top5_cpp = topk_indices(cpp_row, k);

        receipt.add_position(PositionMetrics {
            pos,
            mse,
            max_abs: max_abs_diff,
            kl,
            topk_agree: topk_agreement,
            top5_rust,
            top5_cpp,
        });
    }

    receipt.finalize();
    receipt.write_to_file(receipt_path)?;

    Ok(())
}

fn output_comparison_results(
    divergence: &bitnet_crossval::logits_compare::LogitsDivergence,
    format: &str,
    cos_tol: f32,
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
) -> Result<()> {
    match format {
        "json" => {
            let output = serde_json::json!({
                "first_divergence_token": divergence.first_divergence_token,
                "per_token_cosine_sim": divergence.per_token_cosine_sim,
                "per_token_l2_dist": divergence.per_token_l2_dist,
                "max_absolute_diff": divergence.max_absolute_diff,
                "threshold": cos_tol,
                "status": if divergence.first_divergence_token.is_none() { "ok" } else { "diverged" }
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => output_text_results(divergence, cos_tol, model_path, tokenizer_path, prompt),
    }

    Ok(())
}

fn output_text_results(
    divergence: &bitnet_crossval::logits_compare::LogitsDivergence,
    cos_tol: f32,
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
) {
    for (t, (&cosine, &l2)) in
        divergence.per_token_cosine_sim.iter().zip(divergence.per_token_l2_dist.iter()).enumerate()
    {
        let ok = cosine >= cos_tol;
        let symbol = if ok { "✓" } else { "✗" };
        println!("{} t={} cosine={:.6} l2={:.2e}", symbol, t, cosine, l2);

        if !ok && divergence.first_divergence_token == Some(t) {
            println!("   ↑ First divergence detected at token {}", t);
        }
    }

    println!();
    println!("Max absolute diff: {:.2e}", divergence.max_absolute_diff);

    if let Some(first_div) = divergence.first_divergence_token {
        println!("❌ First divergence at token {}", first_div);
        println!();
        println!("Next steps:");
        println!("  # 1. Capture Rust trace (seq={})", first_div);
        println!(
            "  BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \\"
        );
        println!("    cargo run -p bitnet-cli --features cpu,trace -- run \\");
        println!("    --model {} \\", model_path.display());
        println!("    --tokenizer {} \\", tokenizer_path.display());
        println!("    --prompt \"{}\" \\", prompt);
        println!("    --max-tokens {} --greedy", first_div + 1);
        println!();
        println!(
            "  # 2. Capture C++ trace (seq={}) - see docs/howto/cpp-setup.md if not instrumented",
            first_div
        );
        println!("  BITNET_TRACE_DIR_CPP=/tmp/cpp <cpp-command-here>");
        println!();
        println!("  # 3. Compare traces");
        println!("  cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp");
        println!();
        std::process::exit(1);
    } else {
        println!("✅ All positions match within tolerance");
    }
}
