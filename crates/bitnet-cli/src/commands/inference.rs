//! Inference command implementation

use anyhow::{Context, Result};
use clap::Args;
use console::style;
// use futures::StreamExt; // Removed unused import
use humansize::{DECIMAL, format_size};
use humantime::format_duration;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{fs, time::timeout};
use tracing::{debug, error, info, warn};

use bitnet_inference::{InferenceConfig, InferenceEngine, SamplingConfig};
use bitnet_models::ModelLoader;
use bitnet_tokenizers::{Tokenizer, TokenizerBuilder};
use candle_core::Device;

use crate::config::CliConfig;

/// Generation configuration for inference
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

/// Inference command arguments
#[derive(Args, Debug)]
pub struct InferenceCommand {
    /// Path to the model file
    #[arg(short, long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Model format (auto, gguf, safetensors)
    #[arg(long, default_value = "auto", value_name = "FORMAT")]
    pub model_format: String,

    /// Input prompt (if not provided, interactive mode is used)
    #[arg(short, long, value_name = "TEXT")]
    pub prompt: Option<String>,

    /// Input file containing prompts (one per line)
    #[arg(long, value_name = "PATH")]
    pub input_file: Option<PathBuf>,

    /// Output file for results
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Device to use for inference (cpu, cuda, auto)
    #[arg(short, long, value_name = "DEVICE")]
    pub device: Option<String>,

    /// Quantization type (i2s, tl1, tl2, auto)
    #[arg(short, long, value_name = "TYPE")]
    pub quantization: Option<String>,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "512", value_name = "N")]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy, higher = more random)
    #[arg(long, default_value = "0.7", value_name = "TEMP")]
    pub temperature: f32,

    /// Top-k sampling parameter
    #[arg(long, value_name = "K")]
    pub top_k: Option<usize>,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, value_name = "P")]
    pub top_p: Option<f32>,

    /// Repetition penalty
    #[arg(long, default_value = "1.1", value_name = "PENALTY")]
    pub repetition_penalty: f32,

    /// Random seed for reproducible generation
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,

    /// Enable greedy decoding (temperature=0, top_p=1, top_k=0)
    #[arg(long)]
    pub greedy: bool,

    /// Force deterministic execution (single-threaded, deterministic ops)
    #[arg(long)]
    pub deterministic: bool,

    /// Number of threads to use (default: all cores)
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Enable streaming output
    #[arg(long)]
    pub stream: bool,

    /// Batch size for processing multiple prompts
    #[arg(long, default_value = "1", value_name = "SIZE")]
    pub batch_size: usize,

    /// Number of parallel workers for batch processing
    #[arg(long, value_name = "N")]
    pub workers: Option<usize>,

    /// Enable interactive mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Show performance metrics
    #[arg(long)]
    pub metrics: bool,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format (text, json, jsonl)
    #[arg(long, default_value = "text", value_name = "FORMAT")]
    pub format: String,

    /// System prompt for chat models
    #[arg(long, value_name = "TEXT")]
    pub system_prompt: Option<String>,

    /// Chat template to use
    #[arg(long, value_name = "TEMPLATE")]
    pub chat_template: Option<String>,

    /// Path to tokenizer.json (HF) or tokenizer.model (SPM)
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Disable BOS insertion
    #[arg(long, default_value_t = false)]
    pub no_bos: bool,

    /// Disable EOS insertion
    #[arg(long, default_value_t = false)]
    pub no_eos: bool,

    /// Stop sequences
    #[arg(long, value_name = "SEQ")]
    pub stop: Vec<String>,

    /// Timeout for inference (in seconds)
    #[arg(long, value_name = "SECONDS")]
    pub timeout: Option<u64>,

    /// Dump top-k logits for first N decode steps (for testing)
    #[arg(long, value_name = "N")]
    pub dump_logits: Option<usize>,

    /// Number of top logits to dump per step
    #[arg(long, default_value = "10", value_name = "K")]
    pub logits_topk: usize,
}

/// Inference result for JSON output
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResult {
    pub prompt: String,
    pub generated_text: String,
    pub counts: TokenCounts,
    pub timing_ms: TimingMetrics,
    pub throughput_tps: ThroughputMetrics,
    pub memory_used: Option<u64>,
    pub model_info: ModelInfo,
    pub tokenizer_info: TokenizerInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_dump: Option<Vec<LogitStep>>,
}

/// Logit information for a single decode step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitStep {
    pub step: usize,
    /// Top-k tokens with their logits: [(token_id, logit)]
    pub topk: Vec<(u32, f32)>,
    /// The token that was actually chosen at this step
    pub chosen_id: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenCounts {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TimingMetrics {
    pub tokenize: f64,
    pub prefill: f64,
    pub decode: f64,
    pub total: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub prefill: f64,
    pub decode: f64,
    pub e2e: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub source: String,
    pub vocab_size: usize,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub quantization: String,
    pub device: String,
    pub parameters: Option<u64>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
}

/// Performance metrics
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub total_tokens: usize,
    pub total_time: Duration,
    pub memory_peak: u64,
    pub memory_current: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl InferenceCommand {
    /// Execute the inference command
    pub async fn execute(&self, config: &CliConfig) -> Result<()> {
        // Setup deterministic environment if requested
        self.setup_environment()?;

        // Setup logging and progress reporting
        let _guard = self.setup_logging(config)?;

        // Validate arguments
        self.validate_args()?;

        // Load model and tokenizer
        let (engine, tokenizer) = self.load_model_and_tokenizer(config).await?;

        // Execute based on mode
        match (self.interactive, self.input_file.as_ref(), self.prompt.as_ref()) {
            (true, _, _) => self.run_interactive_mode(engine, tokenizer).await,
            (false, Some(file), _) => self.run_batch_mode(engine, tokenizer, file).await,
            (false, None, Some(prompt)) => {
                self.run_single_inference(engine, tokenizer, prompt).await
            }
            (false, None, None) => {
                anyhow::bail!("Must provide either --prompt, --input-file, or --interactive");
            }
        }
    }

    /// Setup environment for deterministic execution
    fn setup_environment(&self) -> Result<()> {
        use std::env;

        // Set thread count if specified
        if let Some(threads) = self.threads {
            unsafe { std::env::set_var("RAYON_NUM_THREADS", threads.to_string()) };
            unsafe { std::env::set_var("OMP_NUM_THREADS", threads.to_string()) };
            unsafe { std::env::set_var("MKL_NUM_THREADS", threads.to_string()) };
            unsafe { std::env::set_var("BLAS_NUM_THREADS", threads.to_string()) };
            debug!("Set thread count to {}", threads);
        }

        // Enable deterministic mode if requested
        if self.deterministic {
            unsafe { std::env::set_var("BITNET_DETERMINISTIC", "1") };
            unsafe { std::env::set_var("CANDLE_DETERMINISTIC", "1") };

            // Force single-threaded execution for full determinism
            if self.threads.is_none() {
                unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };
                unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
                unsafe { std::env::set_var("MKL_NUM_THREADS", "1") };
                unsafe { std::env::set_var("BLAS_NUM_THREADS", "1") };
            }

            debug!("Enabled deterministic mode");
        }

        // Set seed in environment if provided
        if let Some(seed) = self.seed {
            unsafe { std::env::set_var("BITNET_SEED", seed.to_string()) };
            debug!("Set seed to {}", seed);
        }

        Ok(())
    }

    /// Setup logging based on configuration
    fn setup_logging(&self, config: &CliConfig) -> Result<()> {
        let level = if self.verbose { "debug" } else { &config.logging.level };

        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

        match config.logging.format.as_str() {
            "json" => {
                tracing_subscriber::fmt()
                    .json()
                    .with_env_filter(filter)
                    .with_target(false)
                    .with_timer(tracing_subscriber::fmt::time::uptime())
                    .init();
            }
            "compact" => {
                tracing_subscriber::fmt()
                    .compact()
                    .with_env_filter(filter)
                    .with_target(false)
                    .init();
            }
            _ => {
                tracing_subscriber::fmt()
                    .pretty()
                    .with_env_filter(filter)
                    .with_target(false)
                    .init();
            }
        }

        Ok(())
    }

    /// Validate command arguments
    fn validate_args(&self) -> Result<()> {
        // Validate format
        match self.format.as_str() {
            "text" | "json" | "jsonl" => {}
            _ => {
                anyhow::bail!("Invalid format: {}. Must be one of: text, json, jsonl", self.format)
            }
        }

        // Validate temperature
        if self.temperature < 0.0 {
            anyhow::bail!("Temperature must be non-negative");
        }

        // Validate top_p
        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                anyhow::bail!("Top-p must be between 0.0 and 1.0");
            }
        }

        // Validate repetition penalty
        if self.repetition_penalty <= 0.0 {
            anyhow::bail!("Repetition penalty must be positive");
        }

        // Validate batch size
        if self.batch_size == 0 {
            anyhow::bail!("Batch size must be greater than 0");
        }

        Ok(())
    }

    /// Load model and tokenizer
    async fn load_model_and_tokenizer(
        &self,
        config: &CliConfig,
    ) -> Result<(InferenceEngine, Arc<dyn bitnet_tokenizers::Tokenizer>)> {
        let model_path = self
            .model
            .as_ref()
            .or(config.default_model.as_ref())
            .context("No model specified. Use --model or set default_model in config")?;

        info!("Loading model from: {}", model_path.display());

        // Show loading progress
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
        pb.set_message("Loading model...");
        pb.enable_steady_tick(Duration::from_millis(100));

        // Determine device
        let device = self.determine_device(config)?;
        debug!("Using device: {:?}", device);

        // Load model
        let loader = ModelLoader::new(bitnet_common::Device::from(&device));
        let model = loader
            .load(model_path)
            .with_context(|| format!("Failed to load model from: {}", model_path.display()))?;

        pb.set_message("Loading tokenizer...");

        // Load tokenizer (try to infer from model or use default)
        let tokenizer = self.load_tokenizer(model_path).await?;

        pb.set_message("Initializing inference engine...");

        // Create inference engine
        let model_arc: Arc<dyn bitnet_models::Model> = model.into();
        let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.clone().into();
        let bn_device = bitnet_common::Device::from(&device);
        let engine = InferenceEngine::new(model_arc, tokenizer_arc, bn_device)
            .context("Failed to create inference engine")?;

        pb.finish_with_message(style("✓ Model loaded successfully").green().to_string());

        Ok((engine, tokenizer))
    }

    /// Determine device to use
    fn determine_device(&self, config: &CliConfig) -> Result<Device> {
        let device_str = self.device.as_ref().unwrap_or(&config.default_device);

        match device_str.as_str() {
            "cpu" | "auto" => {
                info!("Using CPU device");
                Ok(Device::Cpu)
            }
            "cuda" => {
                warn!("CUDA support not yet implemented, falling back to CPU");
                Ok(Device::Cpu)
            }
            _ => anyhow::bail!("Invalid device: {}. Must be one of: cpu, cuda, auto", device_str),
        }
    }

    /// Load tokenizer
    async fn load_tokenizer(
        &self,
        model_path: &PathBuf,
    ) -> Result<Arc<dyn bitnet_tokenizers::Tokenizer>> {
        // If explicit tokenizer path provided, use it
        if let Some(tokenizer_path) = &self.tokenizer {
            debug!("Loading tokenizer from: {}", tokenizer_path.display());
            return TokenizerBuilder::from_file(tokenizer_path)
                .context("Failed to load tokenizer from file");
        }

        // Try GGUF-embedded tokenizer (placeholder for now)
        if let Some(tokenizer) = bitnet_tokenizers::try_from_gguf_metadata(|| {
            anyhow::bail!("GGUF tokenizer not implemented")
        }) {
            debug!("Using GGUF-embedded tokenizer");
            return Ok(tokenizer);
        }

        // Try to load tokenizer from model directory
        let tokenizer_path =
            model_path.parent().map(|p| p.join("tokenizer.json")).filter(|p| p.exists());

        if let Some(tokenizer_path) = tokenizer_path {
            debug!("Loading tokenizer from: {}", tokenizer_path.display());
            TokenizerBuilder::from_file(&tokenizer_path)
                .context("Failed to load tokenizer from file")
        } else {
            anyhow::bail!(
                "No tokenizer found. Pass --tokenizer <tokenizer.json|tokenizer.model> or ensure tokenizer.json exists in model directory."
            )
        }
    }

    /// Run single inference
    async fn run_single_inference(
        &self,
        mut engine: InferenceEngine,
        _tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer>,
        prompt: &str,
    ) -> Result<()> {
        let start_time = Instant::now();
        let config = self.create_generation_config()?;

        if self.stream {
            self.run_streaming_inference(&mut engine, prompt, &config).await?;
        } else {
            let result =
                self.run_batch_inference(&mut engine, &[prompt.to_string()], &config).await?;
            self.output_results(&result).await?;
        }

        if self.metrics {
            self.show_performance_metrics(start_time, 1).await?;
        }

        Ok(())
    }

    /// Run streaming inference
    async fn run_streaming_inference(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<()> {
        // Placeholder implementation - would use actual streaming
        println!("{}", style("Streaming inference not yet fully implemented").yellow());
        // Placeholder implementation - would use actual streaming
        let result =
            "This is a placeholder response. Streaming inference not yet fully implemented."
                .to_string();

        print!("{}", style("Generated: ").bold());
        io::stdout().flush()?;

        // Simulate streaming by printing character by character
        let start_time = Instant::now();
        for (i, char) in result.chars().enumerate() {
            print!("{}", char);
            io::stdout().flush()?;

            // Small delay to simulate streaming
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        println!(); // New line after generation

        if self.metrics {
            let elapsed = start_time.elapsed();
            let token_count = result.split_whitespace().count();
            let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();
            println!("\n{}", style("Performance:").bold());
            println!("  Tokens generated: {}", token_count);
            println!("  Time taken: {}", format_duration(elapsed));
            println!("  Tokens/second: {:.2}", tokens_per_sec);
        }

        Ok(())
    }

    /// Run batch inference
    async fn run_batch_inference(
        &self,
        engine: &mut InferenceEngine,
        prompts: &[String],
        config: &GenerationConfig,
    ) -> Result<Vec<InferenceResult>> {
        let mut results = Vec::new();
        let total_prompts = prompts.len();

        // Setup progress bar
        let pb = ProgressBar::new(total_prompts as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        // Process in batches
        for (batch_idx, batch) in prompts.chunks(self.batch_size).enumerate() {
            let batch_start = Instant::now();

            // Process batch (parallel if workers > 1)
            let batch_results = if let Some(workers) = self.workers {
                self.process_batch_parallel(engine, batch, config, workers).await?
            } else {
                self.process_batch_sequential(engine, batch, config).await?
            };

            results.extend(batch_results);

            let batch_elapsed = batch_start.elapsed();
            pb.inc(batch.len() as u64);

            if self.verbose {
                let batch_tokens_per_sec: f64 = results[batch_idx * self.batch_size..]
                    .iter()
                    .map(|r| r.throughput_tps.decode)
                    .sum::<f64>()
                    / batch.len() as f64;

                debug!(
                    "Batch {} completed in {:.2}s, avg {:.2} tokens/sec",
                    batch_idx + 1,
                    batch_elapsed.as_secs_f64(),
                    batch_tokens_per_sec
                );
            }
        }

        pb.finish_with_message("Inference completed");
        Ok(results)
    }

    /// Process batch sequentially with proper timing
    async fn process_batch_sequential(
        &self,
        engine: &mut InferenceEngine,
        batch: &[String],
        config: &GenerationConfig,
    ) -> Result<Vec<InferenceResult>> {
        let mut results = Vec::new();
        let tokenizer = engine.tokenizer();

        for prompt in batch {
            // 1. Tokenize (measure)
            let t0 = Instant::now();
            let prompt_ids = tokenizer.encode(prompt, !self.no_bos, false)?;
            let t_tok_ms = t0.elapsed().as_secs_f64() * 1e3;

            // 2. Prefill (measure)
            let t1 = Instant::now();
            engine.prefill(&prompt_ids)?;
            let t_prefill_ms = t1.elapsed().as_secs_f64() * 1e3;

            // 3. Decode loop (measure)
            let t2 = Instant::now();

            // Setup logits capture
            let logits_collector = if self.dump_logits.is_some() {
                let collector = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
                let collector_clone = collector.clone();
                let logits_topk = self.logits_topk;

                // Create callback that captures chosen tokens
                let cb =
                    std::sync::Arc::new(move |step: usize, topk: Vec<(u32, f32)>, chosen: u32| {
                        if let Ok(mut sink) = collector_clone.lock() {
                            sink.push(LogitStep { step, topk, chosen_id: Some(chosen) });
                        }
                    })
                        as std::sync::Arc<dyn Fn(usize, Vec<(u32, f32)>, u32) + Send + Sync>;

                Some((collector, cb))
            } else {
                None
            };

            // Create generation config with logits callback
            let mut gen_config = config.clone();
            if let Some((_, cb)) = &logits_collector {
                gen_config.logits_tap_steps = self.dump_logits.unwrap_or(0);
                gen_config.logits_topk = self.logits_topk;
                gen_config.logits_cb = Some(cb.clone());
            }

            // Generate with the engine
            let generated_ids = engine.generate_tokens(&prompt_ids, &gen_config).await?;
            let t_decode_ms = t2.elapsed().as_secs_f64() * 1e3;

            // 4. Decode to text
            let generated_text = tokenizer.decode(&generated_ids)?;

            // 5. Calculate metrics
            let prompt_tokens = prompt_ids.len();
            let generated_tokens = generated_ids.len();
            let total_tokens = prompt_tokens + generated_tokens;
            let t_total_ms = t_tok_ms + t_prefill_ms + t_decode_ms;

            let prefill_tps =
                if t_prefill_ms > 0.0 { prompt_tokens as f64 / (t_prefill_ms / 1e3) } else { 0.0 };
            let decode_tps =
                if t_decode_ms > 0.0 { generated_tokens as f64 / (t_decode_ms / 1e3) } else { 0.0 };
            let e2e_tps =
                if t_total_ms > 0.0 { total_tokens as f64 / (t_total_ms / 1e3) } else { 0.0 };

            // Collect logits if requested
            let logits_dump = if let Some((collector, _)) = logits_collector {
                if let Ok(sink) = collector.lock() { Some(sink.clone()) } else { None }
            } else {
                None
            };

            results.push(InferenceResult {
                prompt: prompt.clone(),
                generated_text,
                counts: TokenCounts { prompt_tokens, generated_tokens, total_tokens },
                timing_ms: TimingMetrics {
                    tokenize: t_tok_ms,
                    prefill: t_prefill_ms,
                    decode: t_decode_ms,
                    total: t_total_ms,
                },
                throughput_tps: ThroughputMetrics {
                    prefill: prefill_tps,
                    decode: decode_tps,
                    e2e: e2e_tps,
                },
                memory_used: self.get_memory_usage(),
                model_info: self.get_model_info(),
                tokenizer_info: self.get_tokenizer_info(tokenizer.as_ref()),
                logits_dump,
            });
        }

        Ok(results)
    }

    /// Process batch in parallel (placeholder - would need thread-safe engine)
    async fn process_batch_parallel(
        &self,
        engine: &mut InferenceEngine,
        batch: &[String],
        config: &GenerationConfig,
        _workers: usize,
    ) -> Result<Vec<InferenceResult>> {
        // For now, fall back to sequential processing
        // In a full implementation, this would clone engines or use a thread pool
        warn!("Parallel processing not yet implemented, falling back to sequential");
        self.process_batch_sequential(engine, batch, config).await
    }

    /// Run interactive mode
    async fn run_interactive_mode(
        &self,
        mut engine: InferenceEngine,
        tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer>,
    ) -> Result<()> {
        println!("{}", style("BitNet Interactive Mode").bold().cyan());
        println!("Type your prompts below. Press Ctrl+C to exit, Ctrl+D for new session.\n");

        let mut conversation_history = Vec::new();
        let config = self.create_generation_config()?;

        loop {
            // Get user input
            print!("{} ", style(">").bold().green());
            io::stdout().flush()?;

            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(0) => break, // EOF (Ctrl+D)
                Ok(_) => {
                    let input = input.trim();
                    if input.is_empty() {
                        continue;
                    }

                    // Handle special commands
                    match input {
                        "/help" => {
                            self.show_interactive_help();
                            continue;
                        }
                        "/clear" => {
                            conversation_history.clear();
                            println!("Conversation history cleared.");
                            continue;
                        }
                        "/metrics" => {
                            // Show current metrics
                            continue;
                        }
                        "/exit" | "/quit" => break,
                        _ => {}
                    }

                    // Prepare prompt with history if using chat template
                    let prompt = if let Some(template) = &self.chat_template {
                        self.format_chat_prompt(template, &conversation_history, input)?
                    } else {
                        input.to_string()
                    };

                    // Generate response
                    let start_time = Instant::now();

                    // Placeholder implementation
                    let response = format!("Response to: {}", input);

                    if self.stream {
                        print!("{} ", style("Assistant:").bold().blue());
                        io::stdout().flush()?;

                        // Simulate streaming
                        for char in response.chars() {
                            print!("{}", char);
                            io::stdout().flush()?;
                            tokio::time::sleep(Duration::from_millis(30)).await;
                        }
                        println!(); // New line
                    } else {
                        println!("{} {}", style("Assistant:").bold().blue(), response);
                    }

                    conversation_history.push((input.to_string(), response));

                    if self.metrics {
                        let elapsed = start_time.elapsed();
                        println!(
                            "  {} {}",
                            style("Time:").dim(),
                            style(format_duration(elapsed)).dim()
                        );
                    }

                    println!(); // Extra line for readability
                }
                Err(e) => {
                    error!("Failed to read input: {}", e);
                    break;
                }
            }
        }

        println!("\n{}", style("Goodbye!").bold().cyan());
        Ok(())
    }

    /// Run batch mode from file
    async fn run_batch_mode(
        &self,
        mut engine: InferenceEngine,
        _tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer>,
        input_file: &PathBuf,
    ) -> Result<()> {
        info!("Processing prompts from: {}", input_file.display());

        // Read prompts from file
        let content = fs::read_to_string(input_file)
            .await
            .with_context(|| format!("Failed to read input file: {}", input_file.display()))?;

        let prompts: Vec<String> = content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        if prompts.is_empty() {
            anyhow::bail!("No prompts found in input file");
        }

        info!("Found {} prompts to process", prompts.len());

        let config = self.create_generation_config()?;
        let start_time = Instant::now();

        let results = self.run_batch_inference(&mut engine, &prompts, &config).await?;

        self.output_results(&results).await?;

        if self.metrics {
            self.show_performance_metrics(start_time, prompts.len()).await?;
        }

        Ok(())
    }

    /// Create generation configuration
    fn create_generation_config(&self) -> Result<GenerationConfig> {
        // Apply greedy decoding if requested
        let (temperature, top_k, top_p, repetition_penalty) = if self.greedy {
            (0.0, 0, 1.0, 1.0) // Force greedy: no sampling, no penalties
        } else {
            (
                self.temperature,
                self.top_k.unwrap_or(40) as u32,
                self.top_p.unwrap_or(1.0),
                self.repetition_penalty,
            )
        };

        let sampling =
            SamplingConfig { temperature, top_k, top_p, repetition_penalty, seed: self.seed };

        Ok(GenerationConfig {
            max_new_tokens: self.max_tokens,
            sampling,
            stop_sequences: self.stop.clone(),
            stream: self.stream,
        })
    }

    /// Output results in the specified format
    async fn output_results(&self, results: &[InferenceResult]) -> Result<()> {
        let output: Box<dyn Write> = if let Some(output_path) = &self.output {
            Box::new(std::fs::File::create(output_path).with_context(|| {
                format!("Failed to create output file: {}", output_path.display())
            })?)
        } else {
            Box::new(io::stdout())
        };

        match self.format.as_str() {
            "json" => {
                serde_json::to_writer_pretty(output, results)?;
            }
            "jsonl" => {
                let mut writer = output;
                for result in results {
                    serde_json::to_writer(&mut writer, result)?;
                    writeln!(writer)?;
                }
            }
            _ => {
                let mut writer = output;
                for (i, result) in results.iter().enumerate() {
                    if results.len() > 1 {
                        writeln!(writer, "=== Result {} ===", i + 1)?;
                        writeln!(writer, "Prompt: {}", result.prompt)?;
                    }
                    writeln!(writer, "{}", result.generated_text)?;
                    if self.metrics {
                        writeln!(
                            writer,
                            "Tokens: {}, Time: {:.2}s, Speed: {:.2} tok/s",
                            result.counts.generated_tokens,
                            result.timing_ms.total / 1e3,
                            result.throughput_tps.decode
                        )?;
                    }
                    if results.len() > 1 {
                        writeln!(writer)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Show performance metrics
    async fn show_performance_metrics(
        &self,
        start_time: Instant,
        total_prompts: usize,
    ) -> Result<()> {
        let total_time = start_time.elapsed();

        println!("\n{}", style("Performance Metrics:").bold());
        println!("  Total prompts: {}", total_prompts);
        println!("  Total time: {}", format_duration(total_time));
        println!(
            "  Average time per prompt: {}",
            format_duration(total_time / total_prompts as u32)
        );

        if let Some(memory) = self.get_memory_usage() {
            println!("  Memory usage: {}", format_size(memory, DECIMAL));
        }

        Ok(())
    }

    /// Get current memory usage (placeholder)
    fn get_memory_usage(&self) -> Option<u64> {
        // This would integrate with system memory monitoring
        None
    }

    /// Get model information
    fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            path: self.model.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
            quantization: self.quantization.clone().unwrap_or_default(),
            device: self.device.clone().unwrap_or_default(),
            parameters: None,  // Would be extracted from model
            vocab_size: None,  // Would be extracted from model
            hidden_size: None, // Would be extracted from model
        }
    }

    /// Get tokenizer information
    fn get_tokenizer_info(&self, tokenizer: &dyn bitnet_tokenizers::Tokenizer) -> TokenizerInfo {
        // Determine source from tokenizer type or path
        let source = if self.tokenizer.is_some() {
            let path = self.tokenizer.as_ref().unwrap();
            if path.extension().map_or(false, |e| e == "json") {
                "hf_json".to_string()
            } else if path.extension().map_or(false, |e| e == "model") {
                "spm".to_string()
            } else {
                "unknown".to_string()
            }
        } else {
            "gguf".to_string() // Assume GGUF if no explicit tokenizer
        };

        TokenizerInfo {
            source,
            vocab_size: tokenizer.vocab_size(),
            bos_id: tokenizer.bos_token_id(),
            eos_id: tokenizer.eos_token_id(),
        }
    }

    /// Format chat prompt with template
    fn format_chat_prompt(
        &self,
        _template: &str,
        history: &[(String, String)],
        current_input: &str,
    ) -> Result<String> {
        // Simple template formatting - in practice would use a proper template engine
        let mut prompt = String::new();

        if let Some(system_prompt) = &self.system_prompt {
            prompt.push_str(&format!("System: {}\n\n", system_prompt));
        }

        for (user_msg, assistant_msg) in history {
            prompt.push_str(&format!("User: {}\nAssistant: {}\n\n", user_msg, assistant_msg));
        }

        prompt.push_str(&format!("User: {}\nAssistant:", current_input));

        Ok(prompt)
    }

    /// Show interactive help
    fn show_interactive_help(&self) {
        println!("{}", style("Interactive Commands:").bold());
        println!("  /help     - Show this help message");
        println!("  /clear    - Clear conversation history");
        println!("  /metrics  - Show current performance metrics");
        println!("  /exit     - Exit interactive mode");
        println!("  Ctrl+C    - Exit");
        println!("  Ctrl+D    - New session");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device};
    use bitnet_models::Model;
    use bitnet_tokenizers::Tokenizer;
    use std::{sync::Arc, thread::sleep, time::Duration};

    struct MockModel {
        config: BitNetConfig,
    }

    impl MockModel {
        fn new() -> Self {
            Self { config: BitNetConfig::default() }
        }
    }

    impl Model for MockModel {
        fn config(&self) -> &BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn std::any::Any,
        ) -> Result<ConcreteTensor, BitNetError> {
            // Introduce delay to make prefill latency measurable
            sleep(Duration::from_millis(10));
            Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
        }

        fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
            let seq_len = tokens.len();
            let hidden_dim = self.config.model.hidden_size;
            Ok(ConcreteTensor::mock(vec![seq_len, hidden_dim]))
        }

        fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
            Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
        }
    }

    struct MockTokenizer;
    impl MockTokenizer {
        fn new() -> Self {
            Self
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_bos: bool,
            _add_special: bool,
        ) -> Result<Vec<u32>, BitNetError> {
            Ok((0..text.len().min(10)).map(|i| i as u32 + 1).collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String, BitNetError> {
            Ok(format!("decoded_{}_tokens", tokens.len()))
        }

        fn vocab_size(&self) -> usize {
            50257
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(50256)
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(50257)
        }

        fn token_to_piece(&self, token: u32) -> Option<String> {
            Some(format!("<token_{}>", token))
        }
    }

    #[tokio::test]
    async fn prefill_is_executed() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();

        let cmd = InferenceCommand {
            model: None,
            model_format: "auto".into(),
            prompt: None,
            input_file: None,
            output: None,
            device: None,
            quantization: None,
            max_tokens: 16,
            temperature: 0.7,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.1,
            seed: None,
            greedy: false,
            deterministic: false,
            threads: None,
            stream: false,
            batch_size: 1,
            workers: None,
            interactive: false,
            metrics: false,
            verbose: false,
            format: "text".into(),
            system_prompt: None,
            chat_template: None,
            tokenizer: None,
            no_bos: false,
            no_eos: false,
            stop: Vec::new(),
            timeout: None,
            dump_logits: None,
            logits_topk: 10,
        };

        let gen_config = GenerationConfig {
            max_new_tokens: 1,
            sampling: SamplingConfig::default(),
            stop_sequences: vec![],
            stream: false,
        };

        let results = cmd
            .process_batch_sequential(&mut engine, &["hello".to_string()], &gen_config)
            .await
            .unwrap();

        assert!(results[0].timing_ms.prefill >= 5.0, "prefill should record latency");
    }
}
