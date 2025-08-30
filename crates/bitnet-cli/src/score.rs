use anyhow::{Context, Result};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

use bitnet_common::Device;
use bitnet_inference::engine::InferenceEngine;
use bitnet_models::{GgufReader, loader::ModelLoader};
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
    let tokenizer: Arc<dyn Tokenizer> = if let Some(spm) = &args.tokenizer {
        let tok = bitnet_tokenizers::load_tokenizer(spm)
            .with_context(|| format!("load tokenizer {}", spm.display()))?;
        tok.into()
    } else {
        let tok = bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
            .context("GGUF has no embedded tokenizer; pass --tokenizer")?;
        tok.into()
    };

    // Load model and create inference engine (CPU only for now)
    let loader = ModelLoader::new(Device::Cpu);
    let model =
        loader.load(&args.model).with_context(|| format!("load model {}", args.model.display()))?;
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let mut engine = InferenceEngine::new(model_arc, tokenizer.clone(), Device::Cpu)
        .context("create inference engine")?;

    // Load dataset
    let data =
        fs::read_to_string(&args.file).with_context(|| format!("read {}", args.file.display()))?;

    let mut total_tokens: usize = 0;
    let mut total_nll: f64 = 0.0;

    let start = Instant::now();

    'lines: for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let ids =
            tokenizer.encode(line, /*bos*/ false, /*add_special*/ false).context("tokenize")?;
        if ids.len() < 2 {
            continue;
        }

        let mut prefix = vec![ids[0]];
        for t in 1..ids.len() {
            if args.max_tokens > 0 && total_tokens >= args.max_tokens {
                break 'lines;
            }

            let mut logits =
                engine.eval_ids(&prefix).await.context("eval_ids in teacher-forcing")?;

            for v in &mut logits {
                if !v.is_finite() {
                    *v = f32::NEG_INFINITY;
                }
            }

            let target = ids[t] as usize;
            if target >= logits.len() {
                anyhow::bail!("target index {} out of bounds", target);
            }
            // Compute log probability for the target token
            let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f64;
            for &v in &logits {
                sum += ((v - m) as f64).exp();
            }
            let log_sum = m as f64 + sum.ln();
            let lp = logits[target] as f64 - log_sum;

            total_nll -= lp;
            total_tokens += 1;
            prefix.push(ids[t]);
        }
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let mean_nll = if total_tokens > 0 { total_nll / total_tokens as f64 } else { 0.0 };
    let ppl = mean_nll.exp();
    let latency = json!({
        "total_ms": total_ms,
        "per_token_ms": if total_tokens > 0 {
            Some(total_ms / total_tokens as f64)
        } else {
            None::<f64>
        }
    });

    let tokenizer_origin = if args.tokenizer.is_some() { "external" } else { "embedded" };

    let out = json!({
        "type": "score",
        "model": args.model.display().to_string(),
        "dataset": args.file.display().to_string(),
        "tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "latency": latency,
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
