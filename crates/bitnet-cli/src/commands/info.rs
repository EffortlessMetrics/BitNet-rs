//! Model and tokenizer info command

use anyhow::{Result, Context};
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
use tracing::info;

use bitnet_inference::loader::{ModelLoader, parse_model_format};

/// Display model and tokenizer information
#[derive(Debug, Parser)]
pub struct InfoCommand {
    /// Path to the model file
    #[arg(short, long, value_name = "PATH")]
    pub model: PathBuf,

    /// Path to tokenizer (optional, will use embedded if available)
    #[arg(short, long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Model format override (auto, gguf, safetensors)
    #[arg(long, default_value = "auto")]
    pub model_format: String,

    /// Output format (text or json)
    #[arg(long, default_value = "text")]
    pub output_format: String,

    /// Show scoring policy details
    #[arg(long)]
    pub show_policy: bool,

    /// Show ignored tensors
    #[arg(long)]
    pub show_ignored: bool,
}

impl InfoCommand {
    pub async fn run(self) -> Result<()> {
        info!("Loading model information from: {}", self.model.display());

        // Parse format override
        let format_override = parse_model_format(&self.model_format)
            .context("Invalid model format")?;

        // Create loader
        let mut loader = ModelLoader::new(&self.model);

        if let Some(tok_path) = &self.tokenizer {
            loader = loader.with_tokenizer(tok_path);
        }

        if let Some(fmt) = format_override {
            loader = loader.with_format(fmt);
        }

        // Load model and get metadata
        let (_model, _tokenizer, metadata) = loader.load()
            .context("Failed to load model")?;

        // Output based on format
        match self.output_format.as_str() {
            "json" => {
                let output = json!({
                    "model_path": self.model.display().to_string(),
                    "format": metadata.format.name(),
                    "format_source": metadata.format_source,
                    "tokenizer_source": metadata.tokenizer_source,
                    "model_config": {
                        "vocab_size": metadata.model_config.vocab_size,
                        "hidden_size": metadata.model_config.hidden_size,
                        "num_layers": metadata.model_config.num_layers,
                        "num_heads": metadata.model_config.num_heads,
                        "context_length": metadata.model_config.context_length,
                    },
                    "tensors_loaded": metadata.tensors_loaded,
                    "ignored_tensors_count": metadata.ignored_tensors.len(),
                    "scoring_policy": if self.show_policy {
                        json!(metadata.scoring_policy)
                    } else {
                        json!(null)
                    },
                    "ignored_tensors": if self.show_ignored {
                        json!(metadata.ignored_tensors)
                    } else {
                        json!(null)
                    }
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            _ => {
                // Text format
                println!("Model Information");
                println!("================");
                println!("Path: {}", self.model.display());
                println!("Format: {} ({})", metadata.format.name(), metadata.format_source);
                println!("Tokenizer: {}", metadata.tokenizer_source);
                println!();

                println!("Model Configuration");
                println!("------------------");
                println!("Vocab Size: {}", metadata.model_config.vocab_size);
                println!("Hidden Size: {}", metadata.model_config.hidden_size);
                println!("Layers: {}", metadata.model_config.num_layers);
                println!("Attention Heads: {}", metadata.model_config.num_heads);
                println!("Context Length: {}", metadata.model_config.context_length);
                println!();

                println!("Loading Statistics");
                println!("-----------------");
                println!("Tensors Loaded: {}", metadata.tensors_loaded);
                println!("Ignored Tensors: {}", metadata.ignored_tensors.len());

                if self.show_policy {
                    println!();
                    println!("Scoring Policy");
                    println!("-------------");
                    println!("Add BOS: {}", metadata.scoring_policy.add_bos);
                    println!("Append EOS: {}", metadata.scoring_policy.append_eos);
                    println!("Mask Padding: {}", metadata.scoring_policy.mask_pad);
                }

                if self.show_ignored && !metadata.ignored_tensors.is_empty() {
                    println!();
                    println!("Ignored Tensors");
                    println!("--------------");
                    for tensor in &metadata.ignored_tensors[..5.min(metadata.ignored_tensors.len())] {
                        println!("  - {}", tensor);
                    }
                    if metadata.ignored_tensors.len() > 5 {
                        println!("  ... and {} more", metadata.ignored_tensors.len() - 5);
                    }
                }
            }
        }

        Ok(())
    }
}
