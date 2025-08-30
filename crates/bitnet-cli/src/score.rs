use anyhow::{Context, Result, anyhow};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

use bitnet_common::Device as BnDevice;
use bitnet_inference::InferenceEngine;
use bitnet_models::{GgufReader, ModelLoader};
use bitnet_tokenizers::Tokenizer;
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

    /// Batch size for scoring
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,

    /// Device to run on (cpu, cuda, metal, auto)
    #[arg(long, default_value = "auto")]
    pub device: String,

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
        bitnet_tokenizers::load_tokenizer(spm)
            .with_context(|| format!("load tokenizer {}", spm.display()))?
            .into()
    } else {
        bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
            .context("GGUF has no embedded tokenizer; pass --tokenizer")?
            .into()
    };
    let tokenizer_arc = tokenizer.clone();

    // Determine device and load model
    let device = parse_device(&args.device)?;
    let loader = ModelLoader::new(BnDevice::from(&device));
    let model =
        loader.load(&args.model).with_context(|| format!("load model {}", args.model.display()))?;
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let mut engine =
        InferenceEngine::new(model_arc, tokenizer_arc.clone(), BnDevice::from(&device))?;

    // Load dataset
    let data =
        fs::read_to_string(&args.file).with_context(|| format!("read {}", args.file.display()))?;
    let mut lines_iter = data.lines().filter(|l| !l.trim().is_empty());

    let start = Instant::now();
    let mut total_tokens: usize = 0;
    let mut total_nll: f64 = 0.0;

    loop {
        let mut batch = Vec::with_capacity(args.batch_size);
        for _ in 0..args.batch_size {
            if let Some(line) = lines_iter.next() {
                batch.push(line.to_string());
            } else {
                break;
            }
        }
        if batch.is_empty() {
            break;
        }

        for line in batch {
            let ids = tokenizer.encode(&line, true, true).context("tokenize")?;
            if ids.len() < 2 {
                continue;
            }
            let (sum, tokens) = compute_nll(&mut engine, &ids, tokenizer.pad_token_id()).await?;
            total_tokens += tokens;
            total_nll += sum;
            if args.max_tokens > 0 && total_tokens >= args.max_tokens {
                break;
            }
        }
        if args.max_tokens > 0 && total_tokens >= args.max_tokens {
            break;
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mean_nll = if total_tokens > 0 { total_nll / total_tokens as f64 } else { 0.0 };
    let ppl = mean_nll.exp();

    let tokenizer_origin = if args.tokenizer.is_some() { "external" } else { "embedded" };

    let out = json!({
        "type": "score",
        "model": args.model.display().to_string(),
        "dataset": args.file.display().to_string(),
        "tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "latency": { "total_ms": elapsed_ms },
        "tokenizer": {
            "type": "sentencepiece",
            "origin": tokenizer_origin
        },
        "gen_policy": {
            "bos": true,
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

fn log_softmax_stable(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = xs.iter().map(|x| (*x - max).exp()).sum();
    xs.iter().map(|x| (*x - max) - sum.ln()).collect()
}

async fn compute_nll(
    engine: &mut InferenceEngine,
    tokens: &[u32],
    pad_id: Option<u32>,
) -> Result<(f64, usize)> {
    if tokens.len() < 2 {
        return Ok((0.0, 0));
    }
    let mut prefix: Vec<u32> = Vec::with_capacity(tokens.len());
    prefix.push(tokens[0]);
    let mut sum = 0.0;
    let mut count = 0;

    for t in 1..tokens.len() {
        let mut logits = engine.eval_ids(&prefix).await?;
        for v in &mut logits {
            if !v.is_finite() {
                *v = f32::NEG_INFINITY;
            }
        }
        if let Some(pid) = pad_id {
            if tokens[t] == pid {
                prefix.push(tokens[t]);
                continue;
            }
        }
        let logp = log_softmax_stable(&logits);
        let target = tokens[t] as usize;
        let lp =
            *logp.get(target).ok_or_else(|| anyhow!("target index {} out of bounds", target))?;
        sum -= lp as f64;
        count += 1;
        prefix.push(tokens[t]);
    }

    Ok((sum, count))
}

fn parse_device(dev: &str) -> Result<Device> {
    match dev {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => Device::cuda_if_available(0).context("CUDA not available"),
        "auto" => {
            if let Ok(d) = Device::cuda_if_available(0) {
                Ok(d)
            } else {
                Ok(Device::Cpu)
            }
        }
        other => Err(anyhow!("Invalid device: {}", other)),
    }
}
