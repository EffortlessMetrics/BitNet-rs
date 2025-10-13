use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[path = "../common.rs"]
mod common;
use common::{iter_ln_tensors, read_safetensors_bytes, rms_for_tensor};

/// Inspect LayerNorm gamma dtype and RMS straight from a SafeTensors file.
/// Exits non-zero if any LN RMS is outside pattern-aware gates:
///  - mlp.ffn_layernorm.weight:          [0.05, 2.0]
///  - post_attention_layernorm.weight:   [0.25, 2.0]
///  - all other norm weights:            [0.50, 2.0]
#[derive(Parser, Debug)]
#[command(name = "st-ln-inspect")]
struct Cli {
    /// Path to a single .safetensors file
    #[arg(long)]
    input: PathBuf,

    /// Print per-tensor lines
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let buf = read_safetensors_bytes(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;

    let mut total = 0usize;
    let mut bad_ffn = 0usize;
    let mut bad_post = 0usize;
    let mut bad_other = 0usize;

    for (name, t) in iter_ln_tensors(&buf)? {
        total += 1;
        let rms = rms_for_tensor(&t)?;
        let line = format!("{:<72} dtype={:?} rms={:.4}", &name, t.dtype(), rms);

        let is_ffn = name.as_str().contains("ffn_layernorm.weight");
        let is_post = name.as_str().contains("post_attention_layernorm.weight");

        let ok = if is_ffn {
            (0.05..=2.0).contains(&rms)
        } else if is_post {
            (0.25..=2.0).contains(&rms)
        } else {
            (0.50..=2.0).contains(&rms)
        };

        if args.verbose || !ok {
            let mark = if ok { "✅" } else { "❌" };
            println!("{} {}", line, mark);
        }

        if !ok {
            if is_ffn {
                bad_ffn += 1
            } else if is_post {
                bad_post += 1
            } else {
                bad_other += 1
            }
        }
    }

    println!("\n[source] LN gamma tensors: total={}", total);
    println!(
        "[source] out-of-envelope:   ffn={}, post-attn={}, other={}",
        bad_ffn, bad_post, bad_other
    );

    if (bad_ffn + bad_post + bad_other) > 0 {
        std::process::exit(12);
    }
    Ok(())
}
