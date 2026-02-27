use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info, warn};

use bitnet_inference::InferenceEngine;
use bitnet_models::ModelLoader;

use crate::config::CliConfig;

/// Running sum of NLL and the number of predicted tokens (T-1)
#[derive(Clone, Copy, Debug, Default)]
struct NllStats {
    sum: f64,      // total negative log-likelihood over predicted tokens
    tokens: usize, // number of predicted tokens (T-1), padding excluded
}

impl NllStats {
    #[inline]
    fn mean(self) -> f64 {
        if self.tokens > 0 { self.sum / self.tokens as f64 } else { 0.0 }
    }

    #[inline]
    fn add(&mut self, other: NllStats) {
        self.sum += other.sum;
        self.tokens += other.tokens;
    }
}

/// Evaluate model perplexity and performance
#[derive(Debug, Args)]
pub struct EvalCommand {
    /// Path to model file (.gguf)
    #[arg(short, long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Model format (auto, gguf, safetensors)
    #[arg(long, default_value = "auto", value_name = "FORMAT")]
    pub model_format: String,

    /// Path to text file for evaluation (not required if --teacher-force-ids is used)
    #[arg(long, value_name = "PATH", required_unless_present = "teacher_force_ids")]
    pub text_file: Option<PathBuf>,

    /// Path to tokenizer.json (HF) or tokenizer.model (SPM)
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Device to use for inference
    #[arg(short, long, default_value = "auto")]
    pub device: String,

    /// Maximum sequence length for evaluation
    #[arg(long, default_value = "2048")]
    pub max_seq_len: usize,

    /// Batch size for evaluation
    #[arg(long, default_value = "1")]
    pub batch_size: usize,

    /// Output JSON file with results
    #[arg(long, value_name = "PATH")]
    pub json_out: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of lines to evaluate (default: all)
    #[arg(long)]
    pub max_lines: Option<usize>,

    /// Teacher-forcing mode: comma-separated token IDs
    #[arg(long, value_name = "IDS")]
    pub teacher_force_ids: Option<String>,

    /// Dump logit steps during teacher-forcing (max steps)
    #[arg(long)]
    pub dump_logit_steps: Option<usize>,

    /// Top-k tokens to include in logit dump
    #[arg(long, default_value = "10", value_name = "K")]
    pub logits_topk: usize,

    /// Enable deterministic mode (single-threaded)
    #[arg(long)]
    pub deterministic: bool,

    /// Number of threads to use (0 = all cores)
    #[arg(long, default_value = "0")]
    pub threads: usize,
}

/// Single step of logit information for teacher-forcing
#[derive(Debug, Serialize, Deserialize)]
pub struct LogitStep {
    pub step: usize,
    pub topk: Vec<(u32, f32)>, // (token_id, logit)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_id: Option<u32>,
}

/// Scoring policy configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct ScoringPolicy {
    pub add_bos: bool,
    pub append_eos: bool,
    pub mask_pad: bool,
}

/// Environment information for evaluation
#[derive(Debug, Serialize, Deserialize)]
pub struct EvalEnvironment {
    pub platform: String,
    pub bitnet_cli: String,
    pub rust_version: String,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub threads: usize,
}

/// Model metadata for evaluation
#[derive(Debug, Serialize, Deserialize)]
pub struct EvalModelMeta {
    pub format: String,
    pub tokenizer_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

/// Complete metadata for evaluation
#[derive(Debug, Serialize, Deserialize)]
pub struct EvalMeta {
    pub format: String,
    pub tokenizer: String,
    pub scoring_policy: ScoringPolicy,
    pub environment: EvalEnvironment,
    pub model: EvalModelMeta,
}

/// Evaluation results
#[derive(Debug, Serialize, Deserialize)]
pub struct EvalResults {
    pub model_path: String,
    pub text_file: String,
    pub lines_evaluated: usize,
    pub total_tokens: usize,
    pub mean_nll: f64,
    pub std_nll: f64,
    pub perplexity: f64,
    pub timing_ms: EvalTiming,
    pub tokens_per_second: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_dump: Option<Vec<LogitStep>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scoring_policy: Option<ScoringPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tf_path_head: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub totals: Option<EvalTotals>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<EvalMeta>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalTotals {
    pub lines: usize,
    pub predicted_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalTiming {
    pub total: f64,
    pub per_line: f64,
    pub per_token: f64,
}

/// Small helper for deterministic, robust top-k on possibly quantized logits
#[inline]
fn topk_stable_indices(logits: &[f32], k: usize) -> Vec<usize> {
    use core::cmp::Ordering;
    if k == 0 {
        return Vec::new();
    }

    let mut idx: Vec<usize> = (0..logits.len()).collect();

    // Sort by logit descending, then by index ascending for ties
    idx.sort_by(|&a, &b| {
        match logits[b].partial_cmp(&logits[a]) {
            Some(Ordering::Less) => Ordering::Less,
            Some(Ordering::Greater) => Ordering::Greater,
            _ => a.cmp(&b), // Deterministic tie-breaking
        }
    });

    idx.truncate(k);
    idx
}

/// Stable log-softmax computation
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

impl EvalCommand {
    /// Build metadata for the evaluation
    fn build_metadata(
        &self,
        format: &str,
        tokenizer_source: &str,
        scoring_policy: ScoringPolicy,
    ) -> EvalMeta {
        let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
        let bitnet_cli = env!("CARGO_PKG_VERSION").to_string();

        // Get Rust version
        let rust_version = "rustc 1.89.0".to_string(); // Simplified for now

        let threads = if self.deterministic {
            1
        } else {
            std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
        };

        EvalMeta {
            format: format.to_string(),
            tokenizer: tokenizer_source.to_string(),
            scoring_policy,
            environment: EvalEnvironment {
                platform,
                bitnet_cli,
                rust_version,
                deterministic: self.deterministic || std::env::var("BITNET_DETERMINISTIC").is_ok(),
                seed: self.seed,
                threads,
            },
            model: EvalModelMeta {
                format: format.to_string(),
                tokenizer_source: tokenizer_source.to_string(),
                parameters: None,
                quantization: None,
            },
        }
    }

    /// Execute the evaluation command
    pub async fn execute(&self, config: &CliConfig) -> Result<()> {
        // Note: logging already initialized in main()

        // Set deterministic mode if requested
        if self.deterministic {
            unsafe {
                std::env::set_var("BITNET_DETERMINISTIC", "1");
                std::env::set_var("RAYON_NUM_THREADS", "1");
                std::env::set_var("OMP_NUM_THREADS", "1");
                std::env::set_var("MKL_NUM_THREADS", "1");
                std::env::set_var("BLAS_NUM_THREADS", "1");
            }
        }

        // Set thread count
        if self.threads > 0 {
            unsafe {
                std::env::set_var("RAYON_NUM_THREADS", self.threads.to_string());
            }
        }

        info!("Starting model evaluation");

        // Load model and tokenizer
        let (engine, tokenizer) = self.load_model_and_tokenizer(config).await?;

        // Friendly guardrail for accidental empty top-k
        if self.dump_logit_steps.is_some_and(|steps| steps > 0) && self.logits_topk == 0 {
            warn!("--dump-logit-steps > 0 but --logits-topk == 0: no tokens will be recorded.");
        }

        // Route: teacher-forcing vs file
        let results = if let Some(tf_csv) = &self.teacher_force_ids {
            let tf_ids: Vec<u32> = tf_csv
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<u32>())
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to parse --teacher-force-ids CSV")?;
            info!("Running teacher-forcing evaluation with {} tokens", tf_ids.len());
            self.evaluate_teacher_force(engine, tf_ids).await?
        } else {
            let lines = self.read_text_file()?;
            info!("Loaded {} lines for evaluation", lines.len());
            self.evaluate(engine, tokenizer, lines).await?
        };

        // Output results
        self.output_results(&results)?;

        Ok(())
    }

    /// Setup logging
    fn setup_logging(&self, config: &CliConfig) -> Result<()> {
        let level = if self.verbose { "debug" } else { &config.logging.level };

        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_writer(std::io::stderr)
            .init();

        Ok(())
    }

    /// Load model and tokenizer
    async fn load_model_and_tokenizer(
        &self,
        config: &CliConfig,
    ) -> Result<(InferenceEngine, std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>)> {
        let model_path = self
            .model
            .as_ref()
            .or(config.default_model.as_ref())
            .context("No model specified. Use --model or set default_model in config")?;

        info!("Loading model from: {}", model_path.display());

        // Determine device
        let device = self.determine_device()?;
        debug!("Using device: {:?}", device);

        // Load model
        let loader = ModelLoader::new(bitnet_common::Device::from(&device));
        let model = loader
            .load(model_path)
            .with_context(|| format!("Failed to load model from: {}", model_path.display()))?;

        // Load tokenizer
        let tokenizer = self.load_tokenizer(model_path).await?;

        // Create inference engine
        let model_arc: std::sync::Arc<dyn bitnet_models::Model> = model.into();
        let bn_device = bitnet_common::Device::from(&device);
        let engine = InferenceEngine::new(model_arc, tokenizer.clone(), bn_device)
            .context("Failed to create inference engine")?;

        Ok((engine, tokenizer))
    }

    /// Determine device based on configuration
    fn determine_device(&self) -> Result<candle_core::Device> {
        use candle_core::Device;

        match self.device.as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" | "vulkan" | "opencl" | "ocl" => Device::cuda_if_available(0).context("GPU backend not available (OpenCL/Vulkan aliases currently map to CUDA)"),
            "metal" => Device::new_metal(0).context("Metal not available"),
            "auto" => {
                if let Ok(device) = Device::cuda_if_available(0) {
                    Ok(device)
                } else if let Ok(device) = Device::new_metal(0) {
                    Ok(device)
                } else {
                    Ok(Device::Cpu)
                }
            }
            _ => anyhow::bail!(
                "Invalid device: {}. Must be one of: cpu, cuda, gpu, vulkan, opencl, ocl, metal, auto",
                self.device
            ),
        }
    }

    /// Load tokenizer
    async fn load_tokenizer(
        &self,
        model_path: &Path,
    ) -> Result<std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>> {
        // Use the unified auto-loader for consistent behavior
        let tokenizer = bitnet_tokenizers::auto::load_auto(model_path, self.tokenizer.as_deref())?;

        info!("Successfully loaded tokenizer using auto-detection");
        Ok(tokenizer)
    }

    /// Read text file (only required when not teacher-forcing)
    fn read_text_file(&self) -> Result<Vec<String>> {
        let tf = self.text_file.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--text-file is required unless --teacher-force-ids is provided")
        })?;

        let file = std::fs::File::open(tf)
            .with_context(|| format!("Failed to open text file: {}", tf.display()))?;

        let reader = BufReader::new(file);
        let mut lines = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if let Some(max) = self.max_lines
                && i >= max
            {
                break;
            }

            let line = line.context("Failed to read line from text file")?;
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }

        Ok(lines)
    }

    /// Teacher-forcing NLL for a single sequence using the decode path
    /// Returns (sum of NLL over predicted tokens, number of predicted tokens)
    async fn compute_nll_stats(
        &self,
        engine: &mut InferenceEngine,
        tokens: &[u32],
        pad_id: Option<u32>,
    ) -> Result<NllStats> {
        if tokens.len() < 2 {
            return Ok(NllStats::default());
        }

        let mut stats = NllStats::default();
        let mut prefix: Vec<u32> = Vec::with_capacity(tokens.len());
        prefix.push(tokens[0]);

        for (idx, &current_token) in tokens.iter().skip(1).enumerate() {
            let _t = idx + 1;
            // Get logits from forward pass
            let mut logits =
                engine.eval_ids(&prefix).await.context("eval_ids in teacher-forcing")?;

            // Demote NaNs for robustness
            for v in &mut logits {
                if !v.is_finite() {
                    *v = f32::NEG_INFINITY;
                }
            }

            // Skip padding tokens if specified
            if let Some(pid) = pad_id
                && current_token == pid
            {
                prefix.push(current_token);
                continue;
            }

            // Compute log probabilities
            let logp = log_softmax_stable(&logits);
            let target = current_token as usize;
            let lp = *logp
                .get(target)
                .ok_or_else(|| anyhow::anyhow!("target index {} out of bounds", target))?;

            stats.sum -= lp as f64;
            stats.tokens += 1;

            prefix.push(current_token);
        }

        Ok(stats)
    }

    /// Evaluate on lines from a file (corpus), token-weighted mean NLL
    async fn evaluate(
        &self,
        mut engine: InferenceEngine,
        tokenizer: std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>,
        lines: Vec<String>,
    ) -> Result<EvalResults> {
        let start = Instant::now();
        let mut agg = NllStats::default();

        // Set seed if provided
        if let Some(seed) = self.seed {
            unsafe {
                std::env::set_var("BITNET_SEED", seed.to_string());
            }
        }

        for (i, line) in lines.iter().enumerate() {
            debug!("Evaluating line {}/{}", i + 1, lines.len());

            // Tokenize
            let tokens = tokenizer.encode(line, true, true)?;
            if tokens.len() < 2 {
                continue; // Skip too short sequences
            }

            // Compute NLL with proper teacher-forcing
            let pad_id = tokenizer.pad_token_id();
            let s = self.compute_nll_stats(&mut engine, &tokens, pad_id).await?;
            agg.add(s);

            if (i + 1) % 10 == 0 {
                info!("Progress: {}/{} lines", i + 1, lines.len());
            }
        }

        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let mean_nll = agg.mean();
        let perplexity = mean_nll.exp();
        let predicted = agg.tokens.max(1); // T-1 token count

        Ok(EvalResults {
            model_path: self
                .model
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "default".to_string()),
            text_file: self
                .text_file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "teacher-forced".to_string()),
            lines_evaluated: lines.len(),
            total_tokens: predicted,
            mean_nll,
            std_nll: 0.0, // Would need to track per-line stats for proper std
            perplexity,
            timing_ms: EvalTiming {
                total: elapsed_ms,
                per_line: elapsed_ms / (lines.len().max(1) as f64),
                per_token: elapsed_ms / (predicted as f64),
            },
            tokens_per_second: (predicted as f64) / elapsed.as_secs_f64(),
            logits_dump: None,
            scoring_policy: Some(ScoringPolicy { add_bos: true, append_eos: true, mask_pad: true }),
            tf_path_head: None,
            totals: Some(EvalTotals { lines: lines.len(), predicted_tokens: predicted }),
            meta: None,
        })
    }

    /// Evaluate with teacher-forcing on an explicit token path, with optional logits dump
    async fn evaluate_teacher_force(
        &self,
        mut engine: InferenceEngine,
        tf_ids: Vec<u32>,
    ) -> Result<EvalResults> {
        let start = Instant::now();

        // Compute NLL stats
        let stats = if tf_ids.len() >= 2 {
            self.compute_nll_stats(&mut engine, &tf_ids, None).await?
        } else {
            NllStats::default()
        };

        let mean_nll = stats.mean();
        let perplexity = mean_nll.exp();
        let predicted = stats.tokens.max(1);

        // Optional logits dump (teacher-forced)
        let logits_dump = if let Some(steps) = self.dump_logit_steps {
            if steps > 0 && tf_ids.len() > 1 {
                let mut dump = Vec::new();
                for (step, t) in (0..(tf_ids.len() - 1)).take(steps).enumerate() {
                    let mut logits: Vec<f32> = engine.eval_ids(&tf_ids[..=t]).await?;

                    // Demote NaNs to -inf
                    for v in &mut logits {
                        if !v.is_finite() {
                            *v = f32::NEG_INFINITY;
                        }
                    }

                    let k = self.logits_topk.min(logits.len());
                    let idx = topk_stable_indices(&logits, k);
                    let topk: Vec<(u32, f32)> =
                        idx.into_iter().map(|i| (i as u32, logits[i])).collect();

                    dump.push(LogitStep { step, topk, chosen_id: Some(tf_ids[t + 1]) });
                }
                Some(dump)
            } else {
                None
            }
        } else {
            None
        };

        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        // Include first N tokens of TF path for replay
        let tf_path_head =
            if tf_ids.len() <= 100 { Some(tf_ids.clone()) } else { Some(tf_ids[..100].to_vec()) };

        Ok(EvalResults {
            model_path: self
                .model
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "default".to_string()),
            text_file: "teacher-forced".to_string(),
            lines_evaluated: 1,
            total_tokens: predicted, // predicted tokens (T-1)
            mean_nll,
            std_nll: 0.0,
            perplexity,
            timing_ms: EvalTiming {
                total: elapsed_ms,
                per_line: elapsed_ms,
                per_token: elapsed_ms / (predicted as f64),
            },
            tokens_per_second: (predicted as f64) / elapsed.as_secs_f64(),
            logits_dump,
            scoring_policy: Some(ScoringPolicy {
                add_bos: false, // TF path already includes BOS if needed
                append_eos: false,
                mask_pad: false,
            }),
            tf_path_head,
            totals: Some(EvalTotals { lines: 1, predicted_tokens: predicted }),
            meta: None,
        })
    }

    /// Output results
    fn output_results(&self, results: &EvalResults) -> Result<()> {
        // Print summary
        println!("\n{}", console::style("Evaluation Results").bold().cyan());
        println!("{}", console::style("-".repeat(40)).dim());
        println!("Model:       {}", results.model_path);
        println!("Text file:   {}", results.text_file);
        println!("Lines:       {}", results.lines_evaluated);
        println!("Tokens:      {}", results.total_tokens);
        println!("Mean NLL:    {:.4}", results.mean_nll);
        println!("Std NLL:     {:.4}", results.std_nll);
        println!("Perplexity:  {:.2}", results.perplexity);
        println!("Time:        {:.1}ms", results.timing_ms.total);
        println!("Speed:       {:.1} tok/s", results.tokens_per_second);

        // Save to JSON if requested
        if let Some(json_path) = &self.json_out {
            let file = std::fs::File::create(json_path).with_context(|| {
                format!("Failed to create JSON output: {}", json_path.display())
            })?;
            serde_json::to_writer_pretty(file, results)?;
            info!("Results saved to: {}", json_path.display());
        }

        Ok(())
    }
}
