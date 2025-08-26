use anyhow::{Context, Result};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc};

use bitnet_common::Device;
use bitnet_inference::InferenceEngine;
use bitnet_models::{GgufReader, ModelLoader};
use bitnet_tokenizers::Tokenizer;
use futures::{FutureExt, future::BoxFuture};

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

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NllStats {
    sum: f64,
    tokens: usize,
}

impl NllStats {
    #[inline]
    fn mean(&self) -> f64 {
        if self.tokens > 0 { self.sum / self.tokens as f64 } else { 0.0 }
    }

    #[inline]
    fn add(&mut self, other: NllStats) {
        self.sum += other.sum;
        self.tokens += other.tokens;
    }
}

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

pub trait LogitSource {
    fn eval_ids<'a>(&'a mut self, ids: &'a [u32]) -> BoxFuture<'a, Result<Vec<f32>>>;
}

impl LogitSource for InferenceEngine {
    fn eval_ids<'a>(&'a mut self, ids: &'a [u32]) -> BoxFuture<'a, Result<Vec<f32>>> {
        InferenceEngine::eval_ids(self, ids).boxed()
    }
}

pub(crate) async fn compute_line_nll<E: LogitSource + Send>(
    engine: &mut E,
    tokens: &[u32],
    max_predict: usize,
) -> Result<NllStats> {
    if tokens.len() < 2 || max_predict == 0 {
        return Ok(NllStats::default());
    }
    let mut stats = NllStats::default();
    let mut prefix: Vec<u32> = Vec::with_capacity(tokens.len());
    prefix.push(tokens[0]);
    for t in 1..tokens.len() {
        let logits = engine.eval_ids(&prefix).await?;
        let logp = log_softmax_stable(&logits);
        let target = tokens[t] as usize;
        if let Some(lp) = logp.get(target) {
            stats.sum -= *lp as f64;
            stats.tokens += 1;
        }
        prefix.push(tokens[t]);
        if stats.tokens >= max_predict {
            break;
        }
    }
    Ok(stats)
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
    let tokenizer_box: Box<dyn Tokenizer> = if let Some(spm) = &args.tokenizer {
        bitnet_tokenizers::load_tokenizer(spm)
            .with_context(|| format!("load tokenizer {}", spm.display()))?
    } else {
        bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
            .context("GGUF has no embedded tokenizer; pass --tokenizer")?
    };
    let tokenizer: Arc<dyn Tokenizer> = tokenizer_box.into();

    // Load model and create inference engine
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
    let mut nll_stats = NllStats::default();

    'outer: for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let ids =
            tokenizer.encode(line, /*bos*/ false, /*add_special*/ false).context("tokenize")?;
        total_tokens += ids.len();

        let remaining = if args.max_tokens > 0 {
            args.max_tokens.saturating_sub(nll_stats.tokens)
        } else {
            usize::MAX
        };
        let line_stats = compute_line_nll(&mut engine, &ids, remaining).await?;
        nll_stats.add(line_stats);

        if args.max_tokens > 0 && nll_stats.tokens >= args.max_tokens {
            break 'outer;
        }
    }

    let tokenizer_origin = if args.tokenizer.is_some() { "external" } else { "embedded" };

    let out = json!({
        "type": "score",
        "model": args.model.display().to_string(),
        "dataset": args.file.display().to_string(),
        "tokens": total_tokens,
        "mean_nll": nll_stats.mean(),
        "ppl": nll_stats.mean().exp(),
        "latency": { "total_ms": serde_json::Value::Null },
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

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEngine;

    impl LogitSource for DummyEngine {
        fn eval_ids<'a>(&'a mut self, _ids: &'a [u32]) -> BoxFuture<'a, Result<Vec<f32>>> {
            async { Ok(vec![0.0; 100]) }.boxed()
        }
    }

    #[tokio::test]
    async fn compute_line_nll_uniform() {
        let mut engine = DummyEngine;
        let stats = compute_line_nll(&mut engine, &[0, 1, 2], usize::MAX).await.unwrap();
        assert!((stats.mean() - (100f64).ln()).abs() < 1e-6);
        assert_eq!(stats.tokens, 2);
    }
}
