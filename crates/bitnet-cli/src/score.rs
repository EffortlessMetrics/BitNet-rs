use anyhow::{Context, Result};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc};

use bitnet_common::Device;
use bitnet_inference::InferenceEngine;
use bitnet_models::{GgufReader, ModelLoader};
use bitnet_tokenizers::Tokenizer;

#[async_trait::async_trait]
trait EvalIds {
    async fn eval_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>>;
}

#[async_trait::async_trait]
impl EvalIds for InferenceEngine {
    async fn eval_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        InferenceEngine::eval_ids(self, ids).await
    }
}

// Stable log-softmax computation
fn log_softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0f32;
    for &v in xs {
        sum += (v - max).exp();
    }
    let lse = max + sum.ln();
    xs.iter().map(|&v| v - lse).collect()
}

// Compute NLL for a single token sequence
async fn compute_line_nll<E: EvalIds + Send>(
    engine: &mut E,
    tokens: &[u32],
) -> Result<(f64, usize)> {
    if tokens.len() < 2 {
        return Ok((0.0, 0));
    }

    let mut total = 0.0f64;
    let mut predicted = 0usize;
    let mut prefix: Vec<u32> = Vec::with_capacity(tokens.len());
    prefix.push(tokens[0]);

    for t in 1..tokens.len() {
        let logits = engine.eval_ids(&prefix).await?;
        let logp = log_softmax(&logits);
        let target = tokens[t] as usize;
        let lp = *logp
            .get(target)
            .ok_or_else(|| anyhow::anyhow!("target index {} out of bounds", target))?;
        total -= lp as f64;
        predicted += 1;
        prefix.push(tokens[t]);
    }

    Ok((total, predicted))
}

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
        let tk = bitnet_tokenizers::load_tokenizer(spm)
            .with_context(|| format!("load tokenizer {}", spm.display()))?;
        tk.into()
    } else {
        let tk = bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
            .context("GGUF has no embedded tokenizer; pass --tokenizer")?;
        tk.into()
    };

    // Load model and create inference engine (CPU only)
    let loader = ModelLoader::new(Device::Cpu);
    let model =
        loader.load(&args.model).with_context(|| format!("load model {}", args.model.display()))?;
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let mut engine =
        InferenceEngine::new(model_arc, tokenizer.clone(), Device::Cpu).context("create engine")?;

    // Load dataset
    let data =
        fs::read_to_string(&args.file).with_context(|| format!("read {}", args.file.display()))?;
    let mut total_tokens: usize = 0;
    let mut total_nll: f64 = 0.0;

    'lines: for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let mut ids =
            tokenizer.encode(line, /*bos*/ false, /*add_special*/ false).context("tokenize")?;

        // Respect max_tokens by truncating if needed
        if args.max_tokens > 0 {
            let remaining = args.max_tokens.saturating_sub(total_tokens);
            if remaining == 0 {
                break;
            }
            if ids.len().saturating_sub(1) > remaining {
                ids.truncate(remaining + 1); // keep initial token
            }
        }

        let (nll, predicted) = compute_line_nll(&mut engine, &ids).await?;
        total_tokens += predicted;
        total_nll += nll;

        if args.max_tokens > 0 && total_tokens >= args.max_tokens {
            break 'lines;
        }
    }

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
    struct DummyEngine {
        logits: Vec<Vec<f32>>,
        idx: usize,
    }

    #[async_trait::async_trait]
    impl EvalIds for DummyEngine {
        async fn eval_ids(&mut self, _ids: &[u32]) -> Result<Vec<f32>> {
            let out = self.logits[self.idx].clone();
            self.idx += 1;
            Ok(out)
        }
    }

    #[tokio::test]
    async fn nll_and_ppl_small_sample() {
        let mut engine = DummyEngine {
            logits: vec![vec![0.0, (4.0f32).ln()], vec![(3.0f32).ln(), (2.0f32).ln()]],
            idx: 0,
        };

        let tokens = vec![0u32, 1, 0];
        let (nll, predicted) = compute_line_nll(&mut engine, &tokens).await.unwrap();
        assert_eq!(predicted, 2);
        let mean = nll / predicted as f64;
        let ppl = mean.exp();
        assert!((mean - 0.3669848).abs() < 1e-6);
        assert!((ppl - 1.443).abs() < 1e-3);
    }
}
