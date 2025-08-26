use anyhow::{Context, Result};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

use bitnet_common::Device;
use bitnet_inference::InferenceEngine;
use bitnet_models::{GgufReader, ModelLoader};
use bitnet_tokenizers::Tokenizer;

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
    let tokenizer: Box<dyn Tokenizer> = if let Some(spm) = &args.tokenizer {
        bitnet_tokenizers::load_tokenizer(spm)
            .with_context(|| format!("load tokenizer {}", spm.display()))?
    } else {
        bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
            .context("GGUF has no embedded tokenizer; pass --tokenizer")?
    };

    // Load model and build inference engine (CPU by default)
    let loader = ModelLoader::new(Device::Cpu);
    let model = loader.load(&args.model).context("load model")?;
    let model: Arc<dyn bitnet_models::Model> = model.into();
    let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.into();
    let engine = InferenceEngine::new(model, tokenizer_arc.clone(), Device::Cpu)
        .context("create inference engine")?;
    let tokenizer = tokenizer_arc; // use Arc for encoding below

    // Load dataset
    let data =
        fs::read_to_string(&args.file).with_context(|| format!("read {}", args.file.display()))?;

    // Teacher-forcing over dataset to compute NLL
    let mut total_tokens: usize = 0; // predicted tokens (T-1)
    let mut nll_sum: f64 = 0.0;

    // Helper for stable log-softmax
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

    let start = Instant::now();

    'lines: for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let ids = tokenizer
            .encode(line, true, true)
            .context("tokenize")?;
        if ids.len() < 2 {
            continue;
        }

        let mut prefix = Vec::with_capacity(ids.len());
        prefix.push(ids[0]);
        for t in 1..ids.len() {
            let mut logits = engine
                .logits(&prefix)
                .await
                .context("inference logits")?;

            // Demote NaNs to -inf for robustness
            for v in &mut logits {
                if !v.is_finite() {
                    *v = f32::NEG_INFINITY;
                }
            }

            let logp = log_softmax_stable(&logits);
            let target = ids[t] as usize;
            if let Some(&lp) = logp.get(target) {
                nll_sum -= lp as f64;
                total_tokens += 1;
            }

            prefix.push(ids[t]);

            if args.max_tokens > 0 && total_tokens >= args.max_tokens {
                break 'lines;
            }
        }
    }

    let mean_nll = if total_tokens > 0 { nll_sum / total_tokens as f64 } else { 0.0 };
    let ppl = mean_nll.exp();
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let tokenizer_origin = if args.tokenizer.is_some() { "external" } else { "embedded" };

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
