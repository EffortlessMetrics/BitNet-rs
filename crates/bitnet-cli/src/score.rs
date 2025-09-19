use anyhow::{Context, Result};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

use bitnet_common::Device as BNDevice;
use bitnet_inference::InferenceEngine;
use bitnet_models::{GgufReader, ModelLoader};
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
        "cuda" | "gpu" => Device::cuda_if_available(0).context("CUDA not available")?,
        "metal" => anyhow::bail!("Metal not supported in this build"),
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
    let mut total_tokens: usize = 0; // predicted tokens (T-1)
    let mut total_nll: f64 = 0.0;

    'outer: for chunk in lines.chunks(args.batch_size) {
        for line in chunk {
            let ids = tokenizer.encode(line, false, false).context("tokenize")?;
            if ids.len() < 2 {
                continue;
            }

            let max_steps = if args.max_tokens > 0 {
                args.max_tokens.saturating_sub(total_tokens).min(ids.len() - 1)
            } else {
                ids.len() - 1
            };

            let mut prefix = Vec::with_capacity(ids.len());
            prefix.push(ids[0]);
            for t in 0..max_steps {
                let mut logits = engine.eval_ids(&prefix).await?;
                for v in &mut logits {
                    if !v.is_finite() {
                        *v = f32::NEG_INFINITY;
                    }
                }
                let logp = log_softmax_stable(&logits);
                let target = ids[t + 1] as usize;
                total_nll -= logp[target] as f64;
                total_tokens += 1;
                prefix.push(ids[t + 1]);
                if args.max_tokens > 0 && total_tokens >= args.max_tokens {
                    break 'outer;
                }
            }
        }
    }

    let tokenizer_origin = if args.tokenizer.is_some() { "external" } else { "embedded" };

    let mean_nll = if total_tokens > 0 { total_nll / total_tokens as f64 } else { 0.0 };
    let ppl = mean_nll.exp();
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let out = json!({
        "type": "score",
        "model": args.model.display().to_string(),
        "dataset": args.file.display().to_string(),
        "tokens": total_tokens,
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

#[inline]
fn log_softmax_stable(xs: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &v in xs {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for &v in xs {
        sum += (v - m).exp();
    }
    let lse = m + sum.ln();
    xs.iter().map(|&v| v - lse).collect()
}
