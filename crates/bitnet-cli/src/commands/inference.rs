//! Inference command implementation

use anyhow::{Context, Result};
use clap::Args;
use console::style;
use futures::StreamExt;
use humansize::{format_size, DECIMAL};
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

use bitnet_inference::{BitNetInferenceEngine, InferenceConfig, InferenceEngine, SamplingConfig};
use bitnet_models::ModelLoader;
use bitnet_tokenizers::TokenizerBuilder;
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

    /// Stop sequences
    #[arg(long, value_name = "SEQ")]
    pub stop: Vec<String>,

    /// Timeout for inference (in seconds)
    #[arg(long, value_name = "SECONDS")]
    pub timeout: Option<u64>,
}

/// Inference result for JSON output
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResult {
    pub prompt: String,
    pub generated_text: String,
    pub tokens_generated: usize,
    pub time_taken: Duration,
    pub tokens_per_second: f64,
    pub memory_used: Option<u64>,
    pub model_info: ModelInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub quantization: String,
    pub device: String,
    pub parameters: Option<u64>,
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
    ) -> Result<(BitNetInferenceEngine, Arc<dyn bitnet_tokenizers::Tokenizer>)> {
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
        let loader = ModelLoader::new(device.clone());
        let model = loader
            .load(model_path)
            .with_context(|| format!("Failed to load model from: {}", model_path.display()))?;

        pb.set_message("Loading tokenizer...");

        // Load tokenizer (try to infer from model or use default)
        let tokenizer = self.load_tokenizer(model_path).await?;

        pb.set_message("Initializing inference engine...");

        // Create inference engine
        let inference_config = InferenceConfig::default();
        let engine = BitNetInferenceEngine::with_auto_backend(model, inference_config)
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
        // Try to load tokenizer from model directory or use default
        let tokenizer_path =
            model_path.parent().map(|p| p.join("tokenizer.json")).filter(|p| p.exists());

        if let Some(tokenizer_path) = tokenizer_path {
            debug!("Loading tokenizer from: {}", tokenizer_path.display());
            TokenizerBuilder::from_file(&tokenizer_path)
                .context("Failed to load tokenizer from file")
        } else {
            debug!("Using default GPT-2 tokenizer");
            TokenizerBuilder::from_pretrained("gpt2").context("Failed to load default tokenizer")
        }
    }

    /// Run single inference
    async fn run_single_inference(
        &self,
        mut engine: BitNetInferenceEngine,
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
        engine: &mut BitNetInferenceEngine,
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
        engine: &mut BitNetInferenceEngine,
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
                    .map(|r| r.tokens_per_second)
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

    /// Process batch sequentially
    async fn process_batch_sequential(
        &self,
        _engine: &mut BitNetInferenceEngine,
        batch: &[String],
        _config: &GenerationConfig,
    ) -> Result<Vec<InferenceResult>> {
        let mut results = Vec::new();

        for prompt in batch {
            let start_time = Instant::now();

            // Placeholder implementation
            let generated = format!("Generated response for: {}", prompt);

            let elapsed = start_time.elapsed();
            let tokens_generated = generated.split_whitespace().count(); // Rough token count
            let tokens_per_second = tokens_generated as f64 / elapsed.as_secs_f64();

            results.push(InferenceResult {
                prompt: prompt.clone(),
                generated_text: generated,
                tokens_generated,
                time_taken: elapsed,
                tokens_per_second,
                memory_used: self.get_memory_usage(),
                model_info: self.get_model_info(),
            });
        }

        Ok(results)
    }

    /// Process batch in parallel (placeholder - would need thread-safe engine)
    async fn process_batch_parallel(
        &self,
        engine: &mut BitNetInferenceEngine,
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
        mut engine: BitNetInferenceEngine,
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
        mut engine: BitNetInferenceEngine,
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
        let sampling = SamplingConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: self.seed,
        };

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
                            result.tokens_generated,
                            result.time_taken.as_secs_f64(),
                            result.tokens_per_second
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
            parameters: None, // Would be extracted from model
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
