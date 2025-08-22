use anyhow::{Context, Result};
use clap::Args;
use serde_json::json;
use std::{fs, path::PathBuf};

use bitnet_models::GgufReader;
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
    let gguf_bytes = fs::read(&args.model).with_context(|| format!("read {}", args.model.display()))?;
    let gguf = GgufReader::new(&gguf_bytes).context("parse gguf")?;
    let counts = json!({
        "n_kv": gguf.metadata_keys().len(),
        "n_tensors": gguf.tensor_count(),
        "unmapped": 0
    });

    // Load tokenizer (external preferred)
    let tokenizer: Box<dyn Tokenizer> = if let Some(spm) = &args.tokenizer {
        bitnet_tokenizers::load_tokenizer(spm).with_context(|| format!("load tokenizer {}", spm.display()))?
    } else {
        bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
            .context("GGUF has no embedded tokenizer; pass --tokenizer")?
    };

    // Load dataset
    let data = fs::read_to_string(&args.file).with_context(|| format!("read {}", args.file.display()))?;
    let mut total_tokens: usize = 0;

    // TODO: replace stub with real teacher-forcing when logits are exposed.
    // For now we emit structure with null NLL/PPL so JSON consumers stay stable.
    for line in data.lines() {
        if line.trim().is_empty() { continue; }
        let ids = tokenizer.encode(line, /*bos*/false, /*add_special*/false)
            .context("tokenize")?;
        total_tokens += ids.len();
        if args.max_tokens > 0 && total_tokens >= args.max_tokens { break; }
    }

    let tokenizer_origin = if args.tokenizer.is_some() {
        "external"
    } else {
        "embedded"
    };

    let out = json!({
        "type": "score",
        "model": args.model.display().to_string(),
        "dataset": args.file.display().to_string(),
        "tokens": total_tokens,
        "mean_nll": serde_json::Value::Null,
        "ppl": serde_json::Value::Null,
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
        fs::write(p, serde_json::to_string_pretty(&out)?).with_context(|| format!("write {}", p.display()))?;
        println!("Wrote score results to {}", p.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&out)?);
    }
    Ok(())
}