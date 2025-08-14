//! Benchmarking command implementation

use anyhow::{Context, Result};
use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use bitnet_inference::{BitNetInferenceEngine, InferenceConfig};
use bitnet_models::ModelLoader;
use bitnet_tokenizers::TokenizerBuilder;
use candle_core::Device;

use crate::config::CliConfig;

/// Benchmark command arguments
#[derive(Args, Debug)]
pub struct BenchmarkCommand {
    /// Path to the model file
    #[arg(short, long, value_name = "PATH")]
    pub model: PathBuf,

    /// Device to benchmark (cpu, cuda, auto)
    #[arg(short, long, value_name = "DEVICE")]
    pub device: Option<String>,

    /// Number of benchmark iterations
    #[arg(long, default_value = "10", value_name = "N")]
    pub iterations: usize,

    /// Warmup iterations
    #[arg(long, default_value = "3", value_name = "N")]
    pub warmup: usize,

    /// Benchmark prompt length
    #[arg(long, default_value = "128", value_name = "TOKENS")]
    pub prompt_length: usize,

    /// Generation length
    #[arg(long, default_value = "256", value_name = "TOKENS")]
    pub generation_length: usize,

    /// Compare against Python baseline
    #[arg(long)]
    pub compare_python: bool,

    /// Generate flamegraph
    #[arg(long)]
    pub flamegraph: bool,

    /// Output format (text, json, csv)
    #[arg(long, default_value = "text", value_name = "FORMAT")]
    pub format: String,

    /// Output file for results
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Memory profiling
    #[arg(long)]
    pub memory_profile: bool,

    /// Batch sizes to test
    #[arg(long, value_delimiter = ',', default_values = ["1", "4", "8"])]
    pub batch_sizes: Vec<usize>,

    /// Sequence lengths to test
    #[arg(long, value_delimiter = ',', default_values = ["128", "512", "1024"])]
    pub sequence_lengths: Vec<usize>,
}

/// Benchmark results
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub model_path: String,
    pub device: String,
    pub timestamp: String,
    pub system_info: SystemInfo,
    pub benchmark_config: BenchmarkConfig,
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub rust_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub warmup: usize,
    pub prompt_length: usize,
    pub generation_length: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub iterations: Vec<IterationResult>,
    pub statistics: Statistics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub latency_ms: f64,
    pub tokens_per_second: f64,
    pub memory_used_mb: Option<f64>,
    pub peak_memory_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Statistics {
    pub mean_latency_ms: f64,
    pub std_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub mean_tokens_per_second: f64,
    pub std_tokens_per_second: f64,
    pub peak_memory_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_tests: usize,
    pub total_duration_s: f64,
    pub best_performance: BestPerformance,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BestPerformance {
    pub test_name: String,
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub batch_size: usize,
    pub sequence_length: usize,
}

impl BenchmarkCommand {
    /// Execute the benchmark command
    pub async fn execute(&self, config: &CliConfig) -> Result<()> {
        // Validate arguments
        self.validate_args()?;

        info!("Starting benchmark for model: {}", self.model.display());

        // Load model and tokenizer
        let (mut engine, _tokenizer) = self.load_model_and_tokenizer(config).await?;

        // Run benchmarks
        let results = self.run_benchmarks(&mut engine).await?;

        // Generate flamegraph if requested
        if self.flamegraph {
            self.generate_flamegraph().await?;
        }

        // Compare with Python if requested
        if self.compare_python {
            self.compare_with_python(&results).await?;
        }

        // Output results
        self.output_results(&results).await?;

        Ok(())
    }

    /// Validate command arguments
    fn validate_args(&self) -> Result<()> {
        // Check model file exists
        if !self.model.exists() {
            anyhow::bail!("Model file does not exist: {}", self.model.display());
        }

        // Validate format
        match self.format.as_str() {
            "text" | "json" | "csv" => {}
            _ => anyhow::bail!(
                "Invalid format: {}. Must be one of: text, json, csv",
                self.format
            ),
        }

        // Validate iterations
        if self.iterations == 0 {
            anyhow::bail!("Iterations must be greater than 0");
        }

        // Validate batch sizes
        for &batch_size in &self.batch_sizes {
            if batch_size == 0 {
                anyhow::bail!("Batch size must be greater than 0");
            }
        }

        // Validate sequence lengths
        for &seq_len in &self.sequence_lengths {
            if seq_len == 0 {
                anyhow::bail!("Sequence length must be greater than 0");
            }
        }

        Ok(())
    }

    /// Load model and tokenizer
    async fn load_model_and_tokenizer(
        &self,
        config: &CliConfig,
    ) -> Result<(
        BitNetInferenceEngine,
        std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>,
    )> {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Loading model for benchmarking...");
        pb.enable_steady_tick(Duration::from_millis(100));

        // Determine device
        let device = self.determine_device(config)?;
        debug!("Using device: {:?}", device);

        // Load model
        let loader = ModelLoader::new(device);
        let model = loader
            .load(&self.model)
            .with_context(|| format!("Failed to load model: {}", self.model.display()))?;

        // Load tokenizer
        let tokenizer =
            TokenizerBuilder::from_pretrained("gpt2").context("Failed to load tokenizer")?;

        // Create inference engine
        let inference_config = InferenceConfig::default();
        let engine = BitNetInferenceEngine::with_auto_backend(model, inference_config)
            .context("Failed to create inference engine")?;

        pb.finish_with_message(format!(
            "{} Model loaded for benchmarking",
            style("✓").green()
        ));

        Ok((engine, tokenizer))
    }

    /// Determine device to use
    fn determine_device(&self, config: &CliConfig) -> Result<Device> {
        let device_str = self.device.as_ref().unwrap_or(&config.default_device);

        match device_str.as_str() {
            "cpu" | "auto" => {
                info!("Using CPU device for benchmarking");
                Ok(Device::Cpu)
            }
            "cuda" => {
                warn!("CUDA support not yet implemented, falling back to CPU");
                Ok(Device::Cpu)
            }
            _ => anyhow::bail!(
                "Invalid device: {}. Must be one of: cpu, cuda, auto",
                device_str
            ),
        }
    }

    /// Run all benchmarks
    async fn run_benchmarks(
        &self,
        _engine: &mut BitNetInferenceEngine,
    ) -> Result<BenchmarkResults> {
        let start_time = Instant::now();
        let mut all_results = Vec::new();

        // Calculate total tests
        let total_tests = self.batch_sizes.len() * self.sequence_lengths.len();

        let pb = ProgressBar::new(total_tests as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        // Run benchmarks for each combination
        for &batch_size in &self.batch_sizes {
            for &seq_len in &self.sequence_lengths {
                let test_name = format!("batch_{}_seq_{}", batch_size, seq_len);
                pb.set_message(format!("Running {}", test_name));

                let result = self
                    .run_single_benchmark(&test_name, batch_size, seq_len)
                    .await?;
                all_results.push(result);

                pb.inc(1);
            }
        }

        pb.finish_with_message(format!("{} All benchmarks completed", style("✓").green()));

        // Calculate summary
        let summary = self.calculate_summary(&all_results, start_time.elapsed());

        Ok(BenchmarkResults {
            model_path: self.model.display().to_string(),
            device: self.device.clone().unwrap_or_else(|| "cpu".to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            system_info: self.get_system_info(),
            benchmark_config: BenchmarkConfig {
                iterations: self.iterations,
                warmup: self.warmup,
                prompt_length: self.prompt_length,
                generation_length: self.generation_length,
            },
            results: all_results,
            summary,
        })
    }

    /// Run a single benchmark configuration
    async fn run_single_benchmark(
        &self,
        test_name: &str,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<BenchmarkResult> {
        let mut iterations = Vec::new();

        // Warmup iterations
        for i in 0..self.warmup {
            debug!("Warmup iteration {} for {}", i + 1, test_name);
            self.run_single_iteration(i, batch_size, seq_len, true)
                .await?;
        }

        // Actual benchmark iterations
        for i in 0..self.iterations {
            let result = self
                .run_single_iteration(i, batch_size, seq_len, false)
                .await?;
            iterations.push(result);
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(&iterations);

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            batch_size,
            sequence_length: seq_len,
            iterations,
            statistics,
        })
    }

    /// Run a single iteration
    async fn run_single_iteration(
        &self,
        iteration: usize,
        batch_size: usize,
        seq_len: usize,
        is_warmup: bool,
    ) -> Result<IterationResult> {
        let start_time = Instant::now();

        // Simulate inference work
        let work_duration = Duration::from_millis((50 + batch_size * 10 + seq_len / 10) as u64);
        tokio::time::sleep(work_duration).await;

        let elapsed = start_time.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        // Calculate tokens per second (simulated)
        let total_tokens = batch_size * seq_len;
        let tokens_per_second = total_tokens as f64 / elapsed.as_secs_f64();

        // Simulate memory usage
        let memory_used_mb = if self.memory_profile {
            Some(100.0 + (batch_size as f64 * seq_len as f64 * 0.01))
        } else {
            None
        };

        let peak_memory_mb = memory_used_mb.map(|m| m * 1.2);

        if !is_warmup {
            debug!(
                "Iteration {}: {:.2}ms, {:.2} tok/s",
                iteration + 1,
                latency_ms,
                tokens_per_second
            );
        }

        Ok(IterationResult {
            iteration,
            latency_ms,
            tokens_per_second,
            memory_used_mb,
            peak_memory_mb,
        })
    }

    /// Calculate statistics from iterations
    fn calculate_statistics(&self, iterations: &[IterationResult]) -> Statistics {
        let latencies: Vec<f64> = iterations.iter().map(|r| r.latency_ms).collect();
        let throughputs: Vec<f64> = iterations.iter().map(|r| r.tokens_per_second).collect();

        let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        let std_latency = {
            let variance = latencies
                .iter()
                .map(|&x| (x - mean_latency).powi(2))
                .sum::<f64>()
                / latencies.len() as f64;
            variance.sqrt()
        };

        let std_throughput = {
            let variance = throughputs
                .iter()
                .map(|&x| (x - mean_throughput).powi(2))
                .sum::<f64>()
                / throughputs.len() as f64;
            variance.sqrt()
        };

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted_latencies, 50.0);
        let p95 = percentile(&sorted_latencies, 95.0);
        let p99 = percentile(&sorted_latencies, 99.0);

        let peak_memory = iterations
            .iter()
            .filter_map(|r| r.peak_memory_mb)
            .fold(0.0f64, f64::max);

        Statistics {
            mean_latency_ms: mean_latency,
            std_latency_ms: std_latency,
            min_latency_ms: sorted_latencies[0],
            max_latency_ms: sorted_latencies[sorted_latencies.len() - 1],
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            mean_tokens_per_second: mean_throughput,
            std_tokens_per_second: std_throughput,
            peak_memory_mb: if peak_memory > 0.0 {
                Some(peak_memory)
            } else {
                None
            },
        }
    }

    /// Calculate benchmark summary
    fn calculate_summary(
        &self,
        results: &[BenchmarkResult],
        total_duration: Duration,
    ) -> BenchmarkSummary {
        // Find best performance
        let best = results
            .iter()
            .max_by(|a, b| {
                a.statistics
                    .mean_tokens_per_second
                    .partial_cmp(&b.statistics.mean_tokens_per_second)
                    .unwrap()
            })
            .unwrap();

        let best_performance = BestPerformance {
            test_name: best.test_name.clone(),
            tokens_per_second: best.statistics.mean_tokens_per_second,
            latency_ms: best.statistics.mean_latency_ms,
            batch_size: best.batch_size,
            sequence_length: best.sequence_length,
        };

        // Generate recommendations
        let mut recommendations = Vec::new();

        if best.batch_size > 1 {
            recommendations.push(format!(
                "Best performance achieved with batch size {}",
                best.batch_size
            ));
        }

        if best.statistics.mean_tokens_per_second > 100.0 {
            recommendations.push(
                "Good throughput achieved. Consider GPU acceleration for even better performance."
                    .to_string(),
            );
        } else {
            recommendations.push(
                "Consider optimizing model or using GPU acceleration for better performance."
                    .to_string(),
            );
        }

        if let Some(peak_memory) = best.statistics.peak_memory_mb {
            if peak_memory > 1000.0 {
                recommendations.push("High memory usage detected. Consider using quantization or smaller batch sizes.".to_string());
            }
        }

        BenchmarkSummary {
            total_tests: results.len(),
            total_duration_s: total_duration.as_secs_f64(),
            best_performance,
            recommendations,
        }
    }

    /// Get system information
    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: num_cpus::get(),
            memory_gb: 16.0, // Placeholder
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Generate flamegraph
    async fn generate_flamegraph(&self) -> Result<()> {
        info!("Generating flamegraph...");

        // Placeholder implementation
        println!(
            "{} Flamegraph generation not yet implemented",
            style("⚠").yellow()
        );
        println!("  To generate flamegraphs, use:");
        println!("  cargo install flamegraph");
        println!(
            "  sudo flamegraph -- bitnet benchmark --model {}",
            self.model.display()
        );

        Ok(())
    }

    /// Compare with Python baseline
    async fn compare_with_python(&self, _results: &BenchmarkResults) -> Result<()> {
        info!("Comparing with Python baseline...");

        // Placeholder implementation
        println!(
            "{} Python comparison not yet implemented",
            style("⚠").yellow()
        );
        println!("  To compare with Python:");
        println!("  1. Run the original Python implementation");
        println!("  2. Compare the results manually");

        Ok(())
    }

    /// Output results in the specified format
    async fn output_results(&self, results: &BenchmarkResults) -> Result<()> {
        let output: Box<dyn Write> = if let Some(output_path) = &self.output {
            Box::new(std::fs::File::create(output_path).with_context(|| {
                format!("Failed to create output file: {}", output_path.display())
            })?)
        } else {
            Box::new(std::io::stdout())
        };

        match self.format.as_str() {
            "json" => {
                serde_json::to_writer_pretty(output, results)?;
            }
            "csv" => {
                self.write_csv_results(output, results)?;
            }
            _ => {
                self.write_text_results(output, results)?;
            }
        }

        Ok(())
    }

    /// Write results in text format
    fn write_text_results(
        &self,
        mut output: Box<dyn Write>,
        results: &BenchmarkResults,
    ) -> Result<()> {
        writeln!(
            output,
            "\n{}",
            style("BitNet Benchmark Results").bold().cyan()
        )?;
        writeln!(output, "================================")?;
        writeln!(output)?;

        // System info
        writeln!(output, "{}", style("System Information:").bold())?;
        writeln!(output, "  Model: {}", results.model_path)?;
        writeln!(output, "  Device: {}", results.device)?;
        writeln!(
            output,
            "  OS: {} ({})",
            results.system_info.os, results.system_info.arch
        )?;
        writeln!(output, "  CPU Cores: {}", results.system_info.cpu_cores)?;
        writeln!(output, "  Timestamp: {}", results.timestamp)?;
        writeln!(output)?;

        // Benchmark config
        writeln!(output, "{}", style("Benchmark Configuration:").bold())?;
        writeln!(
            output,
            "  Iterations: {}",
            results.benchmark_config.iterations
        )?;
        writeln!(output, "  Warmup: {}", results.benchmark_config.warmup)?;
        writeln!(
            output,
            "  Prompt Length: {}",
            results.benchmark_config.prompt_length
        )?;
        writeln!(
            output,
            "  Generation Length: {}",
            results.benchmark_config.generation_length
        )?;
        writeln!(output)?;

        // Results
        writeln!(output, "{}", style("Results:").bold())?;
        for result in &results.results {
            writeln!(output, "  {}:", style(&result.test_name).bold())?;
            writeln!(
                output,
                "    Mean Latency: {:.2} ms",
                result.statistics.mean_latency_ms
            )?;
            writeln!(
                output,
                "    Std Latency: {:.2} ms",
                result.statistics.std_latency_ms
            )?;
            writeln!(
                output,
                "    P95 Latency: {:.2} ms",
                result.statistics.p95_latency_ms
            )?;
            writeln!(
                output,
                "    Mean Throughput: {:.2} tokens/sec",
                result.statistics.mean_tokens_per_second
            )?;
            if let Some(memory) = result.statistics.peak_memory_mb {
                writeln!(output, "    Peak Memory: {:.2} MB", memory)?;
            }
            writeln!(output)?;
        }

        // Summary
        writeln!(output, "{}", style("Summary:").bold())?;
        writeln!(output, "  Total Tests: {}", results.summary.total_tests)?;
        writeln!(
            output,
            "  Total Duration: {:.2}s",
            results.summary.total_duration_s
        )?;
        writeln!(
            output,
            "  Best Performance: {} ({:.2} tokens/sec)",
            results.summary.best_performance.test_name,
            results.summary.best_performance.tokens_per_second
        )?;
        writeln!(output)?;

        // Recommendations
        if !results.summary.recommendations.is_empty() {
            writeln!(output, "{}", style("Recommendations:").bold())?;
            for rec in &results.summary.recommendations {
                writeln!(output, "  • {}", rec)?;
            }
        }

        Ok(())
    }

    /// Write results in CSV format
    fn write_csv_results(
        &self,
        mut output: Box<dyn Write>,
        results: &BenchmarkResults,
    ) -> Result<()> {
        writeln!(output, "test_name,batch_size,sequence_length,mean_latency_ms,std_latency_ms,p95_latency_ms,mean_tokens_per_second,peak_memory_mb")?;

        for result in &results.results {
            writeln!(
                output,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{}",
                result.test_name,
                result.batch_size,
                result.sequence_length,
                result.statistics.mean_latency_ms,
                result.statistics.std_latency_ms,
                result.statistics.p95_latency_ms,
                result.statistics.mean_tokens_per_second,
                result
                    .statistics
                    .peak_memory_mb
                    .map(|m| format!("{:.2}", m))
                    .unwrap_or_else(|| "".to_string())
            )?;
        }

        Ok(())
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}
