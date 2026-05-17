use anyhow::{Context, Result, bail};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

use bitnet_common::Device as BNDevice;
use bitnet_inference::InferenceEngine;
use bitnet_models::{GgufReader, ModelLoader};
use bitnet_scoring_core::{NllStats, observe_target_nll, sanitize_logits_in_place};
use candle_core::Device;

#[derive(Args, Debug)]
pub struct ScoreArgs {
    /// GGUF model path
    #[arg(long)]
    pub model: PathBuf,

    /// Optional external SentencePiece model (overrides GGUF)
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Text file, one prompt per line
    #[arg(long)]
    pub file: PathBuf,

    /// Optional cap on tokens evaluated
    #[arg(long, default_value_t = 0)]
    pub max_tokens: usize,

    /// Device to use for inference (cpu, cuda, metal, auto)
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Batch size for scoring
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,

    /// Where to write JSON (stdout if omitted)
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

pub async fn run_score(args: &ScoreArgs) -> Result<()> {
    validate_score_args(args)?;

    // Read GGUF (counts for JSON)
    let gguf_bytes =
        fs::read(&args.model).with_context(|| format!("read {}", args.model.display()))?;
    let gguf = GgufReader::new(&gguf_bytes).context("parse gguf")?;
    let counts = json!({
        "n_kv": gguf.metadata_keys().len(),
        "n_tensors": gguf.tensor_count(),
        "unmapped": 0
    });

    // Load tokenizer (external preferred)
    let tokenizer = if let Some(spm) = &args.tokenizer {
        // Use unified auto-loader for consistency
        bitnet_tokenizers::auto::load_auto(&args.model, Some(spm))?
    } else {
        // Use unified auto-loader for consistency
        bitnet_tokenizers::auto::load_auto(&args.model, None)?
    };

    // Determine device
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" | "gpu" | "vulkan" | "opencl" | "ocl" => Device::cuda_if_available(0)
            .context("GPU backend not available (OpenCL/Vulkan aliases currently map to CUDA)")?,
        "npu" | "metal" => anyhow::bail!("NPU/Metal not supported in this build"),
        "auto" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
        other => anyhow::bail!("invalid device: {other}"),
    };

    // Load model and create inference engine
    let loader = ModelLoader::new(BNDevice::from(&device));
    let model =
        loader.load(&args.model).with_context(|| format!("load model {}", args.model.display()))?;
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let mut engine = InferenceEngine::new(model_arc, tokenizer.clone(), BNDevice::from(&device))
        .context("create inference engine")?;

    // Load dataset
    let data =
        fs::read_to_string(&args.file).with_context(|| format!("read {}", args.file.display()))?;
    let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();

    let start = Instant::now();
    let mut stats = NllStats::default();

    'outer: for chunk in lines.chunks(args.batch_size) {
        for line in chunk {
            let ids = tokenizer.encode(line, false, false).context("tokenize")?;
            if ids.len() < 2 {
                continue;
            }

            let max_steps = if args.max_tokens > 0 {
                args.max_tokens.saturating_sub(stats.tokens).min(ids.len() - 1)
            } else {
                ids.len() - 1
            };

            let mut prefix = Vec::with_capacity(ids.len());
            prefix.push(ids[0]);
            for t in 0..max_steps {
                let mut logits = engine.eval_ids(&prefix).await?;
                sanitize_logits_in_place(&mut logits);
                let target = ids[t + 1] as usize;
                if target >= logits.len() {
                    anyhow::bail!("target index {} out of bounds", target);
                }
                observe_target_nll(&mut stats, &logits, target);
                prefix.push(ids[t + 1]);
                if args.max_tokens > 0 && stats.tokens >= args.max_tokens {
                    break 'outer;
                }
            }
        }
    }

    let tokenizer_origin = if args.tokenizer.is_some() { "external" } else { "embedded" };

    let mean_nll = stats.mean();
    let ppl = stats.perplexity();
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let out = json!({
        "type": "score",
        "model": args.model.display().to_string(),
        "dataset": args.file.display().to_string(),
        "tokens": stats.tokens,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "latency": { "total_ms": latency_ms },
        "tokenizer": {
            "type": "sentencepiece",
            "origin": tokenizer_origin
        },
        "gen_policy": {
            "bos": false,
            "temperature": 0.0,
            "seed": std::env::var("BITNET_SEED").ok()
        },
        "counts": counts
    });

    if let Some(p) = &args.json_out {
        fs::write(p, serde_json::to_string_pretty(&out)?)
            .with_context(|| format!("write {}", p.display()))?;
        println!("Wrote score results to {}", p.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&out)?);
    }
    Ok(())
}

fn validate_score_args(args: &ScoreArgs) -> Result<()> {
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score_args_with_batch_size(batch_size: usize) -> ScoreArgs {
        ScoreArgs {
            model: PathBuf::from("model.gguf"),
            tokenizer: None,
            file: PathBuf::from("prompts.txt"),
            max_tokens: 0,
            device: "cpu".to_string(),
            batch_size,
            json_out: None,
        }
    }

    #[test]
    fn validate_score_args_rejects_zero_batch_size() {
        let err = validate_score_args(&score_args_with_batch_size(0)).unwrap_err();
        assert!(
            err.to_string().contains("--batch-size must be greater than 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_score_args_accepts_positive_batch_size() -> Result<()> {
        validate_score_args(&score_args_with_batch_size(1))?;
        Ok(())
    }
}
