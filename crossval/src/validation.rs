//! Comprehensive validation framework for BitNet.rs vs bitnet.cpp
//!
//! This module provides exhaustive testing across multiple dimensions:
//! - Accuracy: Token generation, perplexity, logit comparison
//! - Performance: Throughput, latency, memory usage
//! - Compatibility: Model loading, edge cases, format variants
//! - Determinism: Reproducible results with seeding

use crate::{CrossvalError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Comprehensive validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub model_path: PathBuf,
    pub model_metadata: ModelMetadata,
    
    // Loading validation
    pub rust_load: LoadResult,
    pub cpp_load: LoadResult,
    
    // Accuracy metrics
    pub accuracy: AccuracyMetrics,
    
    // Performance metrics
    pub performance: PerformanceMetrics,
    
    // Memory metrics
    pub memory: MemoryMetrics,
    
    // Compatibility
    pub compatibility: CompatibilityReport,
    
    // Overall status
    pub status: ValidationStatus,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub file_size: u64,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub data_offset: u64,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub context_length: Option<u32>,
    pub vocab_size: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadResult {
    pub success: bool,
    pub load_time_ms: f64,
    pub memory_used_mb: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    // Token generation
    pub token_match_rate: f64,  // % of tokens that match exactly
    pub edit_distance: f64,     // Average edit distance between sequences
    
    // Logit comparison
    pub logit_mse: Option<f64>,        // Mean squared error of logits
    pub logit_cosine_sim: Option<f64>, // Cosine similarity of logit vectors
    pub top_k_accuracy: HashMap<usize, f64>, // Top-K accuracy (k=1,5,10)
    
    // Perplexity
    pub rust_perplexity: Option<f64>,
    pub cpp_perplexity: Option<f64>,
    pub perplexity_delta: Option<f64>,
    
    // Statistical tests
    pub ks_test_pvalue: Option<f64>,  // Kolmogorov-Smirnov test
    pub deterministic: bool,           // Same output with same seed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Throughput
    pub rust_tokens_per_sec: f64,
    pub cpp_tokens_per_sec: f64,
    pub speedup_factor: f64,  // rust_tps / cpp_tps
    
    // Latency
    pub rust_first_token_ms: f64,
    pub cpp_first_token_ms: f64,
    pub rust_p50_ms: f64,
    pub cpp_p50_ms: f64,
    pub rust_p99_ms: f64,
    pub cpp_p99_ms: f64,
    
    // Batch performance
    pub batch_sizes_tested: Vec<usize>,
    pub batch_throughput: HashMap<usize, (f64, f64)>, // batch_size -> (rust_tps, cpp_tps)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub rust_peak_mb: f64,
    pub cpp_peak_mb: f64,
    pub rust_model_size_mb: f64,
    pub cpp_model_size_mb: f64,
    pub rust_inference_overhead_mb: f64,
    pub cpp_inference_overhead_mb: f64,
    pub memory_efficiency_ratio: f64, // cpp_peak / rust_peak
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub model_format_compatible: bool,
    pub api_compatible: bool,
    pub output_compatible: bool,
    pub edge_cases_handled: Vec<EdgeCaseResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCaseResult {
    pub case_name: String,
    pub rust_result: String,
    pub cpp_result: String,
    pub compatible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Pass,           // All metrics within tolerance
    PartialPass,    // Some metrics pass, minor issues
    XFail,          // Expected failure (e.g., C++ crashes on edge case)
    Fail,           // Validation failed
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    // Test parameters
    pub num_prompts: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
    
    // Tolerances
    pub token_match_threshold: f64,  // Min % tokens that must match (default: 0.95)
    pub perplexity_tolerance: f64,   // Max relative difference (default: 0.05)
    pub performance_tolerance: f64,  // Min speedup factor (default: 0.8)
    pub memory_tolerance: f64,       // Max memory ratio (default: 1.2)
    
    // Feature flags
    pub test_determinism: bool,
    pub test_batch_inference: bool,
    pub test_edge_cases: bool,
    pub collect_profiling: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            num_prompts: 10,
            max_tokens: 100,
            temperature: 0.0,  // Deterministic
            top_k: None,
            top_p: None,
            seed: 42,
            
            token_match_threshold: 0.95,
            perplexity_tolerance: 0.05,
            performance_tolerance: 0.8,
            memory_tolerance: 1.2,
            
            test_determinism: true,
            test_batch_inference: true,
            test_edge_cases: true,
            collect_profiling: false,
        }
    }
}

/// Main validation runner
pub struct ValidationHarness {
    config: ValidationConfig,
    model_path: PathBuf,
    cpp_path: Option<PathBuf>,
}

impl ValidationHarness {
    pub fn new(model_path: PathBuf, config: ValidationConfig) -> Self {
        let cpp_path = std::env::var_os("BITNET_CPP_DIR")
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|h| h.join(".cache/bitnet_cpp")));
        
        Self {
            config,
            model_path,
            cpp_path,
        }
    }
    
    /// Run comprehensive validation
    pub async fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            timestamp: chrono::Utc::now(),
            model_path: self.model_path.clone(),
            model_metadata: self.extract_metadata()?,
            rust_load: LoadResult::default(),
            cpp_load: LoadResult::default(),
            accuracy: AccuracyMetrics::default(),
            performance: PerformanceMetrics::default(),
            memory: MemoryMetrics::default(),
            compatibility: CompatibilityReport::default(),
            status: ValidationStatus::Fail,
            notes: Vec::new(),
        };
        
        // Phase 1: Model loading
        println!("🔍 Phase 1: Model Loading Validation");
        self.validate_loading(&mut report).await?;
        
        // Phase 2: Accuracy validation
        println!("🎯 Phase 2: Accuracy Validation");
        self.validate_accuracy(&mut report).await?;
        
        // Phase 3: Performance benchmarking
        println!("⚡ Phase 3: Performance Benchmarking");
        self.validate_performance(&mut report).await?;
        
        // Phase 4: Memory efficiency
        println!("💾 Phase 4: Memory Efficiency Analysis");
        self.validate_memory(&mut report).await?;
        
        // Phase 5: Compatibility testing
        println!("🔧 Phase 5: Compatibility Testing");
        self.validate_compatibility(&mut report).await?;
        
        // Determine overall status
        report.status = self.determine_status(&report);
        
        Ok(report)
    }
    
    fn extract_metadata(&self) -> Result<ModelMetadata> {
        use bitnet_models::formats::gguf::GgufReader;
        use bitnet_models::loader::MmapFile;
        
        let mmap = MmapFile::open(&self.model_path)?;
        let reader = GgufReader::new(mmap.as_slice())?;
        
        let file_metadata = std::fs::metadata(&self.model_path)?;
        
        Ok(ModelMetadata {
            file_size: file_metadata.len(),
            gguf_version: reader.version(),
            tensor_count: reader.tensor_count(),
            kv_count: reader.metadata_kv_count(),
            data_offset: reader.data_offset(),
            architecture: reader.get_string("general.architecture").ok(),
            quantization: reader.get_string("general.quantization_version").ok(),
            context_length: reader.get_u32("llama.context_length").ok(),
            vocab_size: reader.get_u32("llama.vocab_size").ok(),
        })
    }
    
    async fn validate_loading(&self, report: &mut ValidationReport) -> Result<()> {
        // Test Rust loading
        let start = Instant::now();
        match self.load_rust_model().await {
            Ok(mem_used) => {
                report.rust_load = LoadResult {
                    success: true,
                    load_time_ms: start.elapsed().as_millis() as f64,
                    memory_used_mb: mem_used,
                    error: None,
                };
            }
            Err(e) => {
                report.rust_load = LoadResult {
                    success: false,
                    load_time_ms: start.elapsed().as_millis() as f64,
                    memory_used_mb: 0.0,
                    error: Some(e.to_string()),
                };
            }
        }
        
        // Test C++ loading if available
        if self.cpp_path.is_some() {
            let start = Instant::now();
            match self.load_cpp_model().await {
                Ok(mem_used) => {
                    report.cpp_load = LoadResult {
                        success: true,
                        load_time_ms: start.elapsed().as_millis() as f64,
                        memory_used_mb: mem_used,
                        error: None,
                    };
                }
                Err(e) => {
                    report.cpp_load = LoadResult {
                        success: false,
                        load_time_ms: start.elapsed().as_millis() as f64,
                        memory_used_mb: 0.0,
                        error: Some(e.to_string()),
                    };
                }
            }
        }
        
        Ok(())
    }
    
    async fn validate_accuracy(&self, report: &mut ValidationReport) -> Result<()> {
        // Generate test prompts
        let prompts = self.generate_test_prompts();
        
        let mut token_matches = 0;
        let mut total_tokens = 0;
        let mut edit_distances = Vec::new();
        
        for prompt in &prompts {
            // Generate with both implementations
            let rust_tokens = self.generate_rust(prompt).await?;
            let cpp_tokens = self.generate_cpp(prompt).await?;
            
            // Compare tokens
            let matches = rust_tokens.iter()
                .zip(cpp_tokens.iter())
                .filter(|(a, b)| a == b)
                .count();
            
            token_matches += matches;
            total_tokens += rust_tokens.len().max(cpp_tokens.len());
            
            // Calculate edit distance
            let distance = self.edit_distance(&rust_tokens, &cpp_tokens);
            edit_distances.push(distance as f64);
        }
        
        report.accuracy.token_match_rate = token_matches as f64 / total_tokens as f64;
        report.accuracy.edit_distance = edit_distances.iter().sum::<f64>() / edit_distances.len() as f64;
        
        // Test determinism
        if self.config.test_determinism {
            report.accuracy.deterministic = self.test_determinism().await?;
        }
        
        Ok(())
    }
    
    async fn validate_performance(&self, report: &mut ValidationReport) -> Result<()> {
        // Single token generation benchmark
        let rust_perf = self.benchmark_rust().await?;
        let cpp_perf = self.benchmark_cpp().await?;
        
        report.performance.rust_tokens_per_sec = rust_perf.0;
        report.performance.cpp_tokens_per_sec = cpp_perf.0;
        report.performance.speedup_factor = rust_perf.0 / cpp_perf.0;
        
        report.performance.rust_first_token_ms = rust_perf.1;
        report.performance.cpp_first_token_ms = cpp_perf.1;
        
        // Batch inference if configured
        if self.config.test_batch_inference {
            for batch_size in &[1, 4, 8, 16] {
                let rust_batch = self.benchmark_batch_rust(*batch_size).await?;
                let cpp_batch = self.benchmark_batch_cpp(*batch_size).await?;
                report.performance.batch_throughput.insert(*batch_size, (rust_batch, cpp_batch));
            }
            report.performance.batch_sizes_tested = vec![1, 4, 8, 16];
        }
        
        Ok(())
    }
    
    async fn validate_memory(&self, report: &mut ValidationReport) -> Result<()> {
        // Memory tracking during inference
        let rust_mem = self.track_memory_rust().await?;
        let cpp_mem = self.track_memory_cpp().await?;
        
        report.memory.rust_peak_mb = rust_mem.0;
        report.memory.rust_model_size_mb = rust_mem.1;
        report.memory.rust_inference_overhead_mb = rust_mem.0 - rust_mem.1;
        
        report.memory.cpp_peak_mb = cpp_mem.0;
        report.memory.cpp_model_size_mb = cpp_mem.1;
        report.memory.cpp_inference_overhead_mb = cpp_mem.0 - cpp_mem.1;
        
        report.memory.memory_efficiency_ratio = cpp_mem.0 / rust_mem.0;
        
        Ok(())
    }
    
    async fn validate_compatibility(&self, report: &mut ValidationReport) -> Result<()> {
        report.compatibility.model_format_compatible = 
            report.rust_load.success && report.cpp_load.success;
        
        // Test edge cases if configured
        if self.config.test_edge_cases {
            let edge_cases = vec![
                ("empty_prompt", ""),
                ("unicode", "你好世界 🦀"),
                ("long_prompt", &"a".repeat(1000)),
                ("special_tokens", "<s> </s> <unk>"),
            ];
            
            for (name, prompt) in edge_cases {
                let rust_result = self.generate_rust(prompt).await
                    .map(|t| format!("{} tokens", t.len()))
                    .unwrap_or_else(|e| e.to_string());
                    
                let cpp_result = self.generate_cpp(prompt).await
                    .map(|t| format!("{} tokens", t.len()))
                    .unwrap_or_else(|e| e.to_string());
                
                report.compatibility.edge_cases_handled.push(EdgeCaseResult {
                    case_name: name.to_string(),
                    rust_result: rust_result.clone(),
                    cpp_result: cpp_result.clone(),
                    compatible: rust_result == cpp_result,
                });
            }
        }
        
        report.compatibility.output_compatible = 
            report.accuracy.token_match_rate >= self.config.token_match_threshold;
        
        Ok(())
    }
    
    fn determine_status(&self, report: &ValidationReport) -> ValidationStatus {
        // Check if Rust loads but C++ fails (edge case superiority)
        if report.rust_load.success && !report.cpp_load.success {
            return ValidationStatus::XFail;
        }
        
        // Both must load for full validation
        if !report.rust_load.success {
            return ValidationStatus::Fail;
        }
        
        // Check accuracy thresholds
        let accuracy_pass = report.accuracy.token_match_rate >= self.config.token_match_threshold;
        
        // Check performance thresholds
        let performance_pass = report.performance.speedup_factor >= self.config.performance_tolerance;
        
        // Check memory thresholds
        let memory_pass = report.memory.memory_efficiency_ratio <= self.config.memory_tolerance;
        
        if accuracy_pass && performance_pass && memory_pass {
            ValidationStatus::Pass
        } else if accuracy_pass {
            ValidationStatus::PartialPass
        } else {
            ValidationStatus::Fail
        }
    }
    
    // Stub implementations for actual model interaction
    async fn load_rust_model(&self) -> Result<f64> {
        // TODO: Implement actual Rust model loading
        Ok(1024.0) // Mock memory usage in MB
    }
    
    async fn load_cpp_model(&self) -> Result<f64> {
        // TODO: Implement actual C++ model loading
        Ok(1100.0) // Mock memory usage in MB
    }
    
    async fn generate_rust(&self, prompt: &str) -> Result<Vec<u32>> {
        // TODO: Implement actual Rust generation
        Ok(vec![1, 2, 3, 4, 5])
    }
    
    async fn generate_cpp(&self, prompt: &str) -> Result<Vec<u32>> {
        // TODO: Implement actual C++ generation
        Ok(vec![1, 2, 3, 4, 6]) // Slightly different for testing
    }
    
    async fn benchmark_rust(&self) -> Result<(f64, f64)> {
        // TODO: Implement actual benchmarking
        Ok((150.0, 25.0)) // (tokens/sec, first_token_ms)
    }
    
    async fn benchmark_cpp(&self) -> Result<(f64, f64)> {
        // TODO: Implement actual benchmarking
        Ok((120.0, 30.0)) // (tokens/sec, first_token_ms)
    }
    
    async fn benchmark_batch_rust(&self, batch_size: usize) -> Result<f64> {
        // TODO: Implement batch benchmarking
        Ok(150.0 * batch_size as f64 * 0.8) // Mock throughput
    }
    
    async fn benchmark_batch_cpp(&self, batch_size: usize) -> Result<f64> {
        // TODO: Implement batch benchmarking
        Ok(120.0 * batch_size as f64 * 0.75) // Mock throughput
    }
    
    async fn track_memory_rust(&self) -> Result<(f64, f64)> {
        // TODO: Implement memory tracking
        Ok((1200.0, 1024.0)) // (peak_mb, model_mb)
    }
    
    async fn track_memory_cpp(&self) -> Result<(f64, f64)> {
        // TODO: Implement memory tracking
        Ok((1400.0, 1100.0)) // (peak_mb, model_mb)
    }
    
    async fn test_determinism(&self) -> Result<bool> {
        // TODO: Test with same seed multiple times
        Ok(true)
    }
    
    fn generate_test_prompts(&self) -> Vec<String> {
        vec![
            "The capital of France is".to_string(),
            "In a hole in the ground there lived".to_string(),
            "The quick brown fox".to_string(),
            "def fibonacci(n):".to_string(),
            "Once upon a time".to_string(),
        ]
    }
    
    fn edit_distance(&self, a: &[u32], b: &[u32]) -> usize {
        let len_a = a.len();
        let len_b = b.len();
        let mut matrix = vec![vec![0; len_b + 1]; len_a + 1];
        
        for i in 0..=len_a {
            matrix[i][0] = i;
        }
        for j in 0..=len_b {
            matrix[0][j] = j;
        }
        
        for i in 1..=len_a {
            for j in 1..=len_b {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    matrix[i - 1][j] + 1,
                    std::cmp::min(
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + cost,
                    ),
                );
            }
        }
        
        matrix[len_a][len_b]
    }
}

// Default trait implementations
impl Default for LoadResult {
    fn default() -> Self {
        Self {
            success: false,
            load_time_ms: 0.0,
            memory_used_mb: 0.0,
            error: None,
        }
    }
}

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self {
            token_match_rate: 0.0,
            edit_distance: 0.0,
            logit_mse: None,
            logit_cosine_sim: None,
            top_k_accuracy: HashMap::new(),
            rust_perplexity: None,
            cpp_perplexity: None,
            perplexity_delta: None,
            ks_test_pvalue: None,
            deterministic: false,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            rust_tokens_per_sec: 0.0,
            cpp_tokens_per_sec: 0.0,
            speedup_factor: 0.0,
            rust_first_token_ms: 0.0,
            cpp_first_token_ms: 0.0,
            rust_p50_ms: 0.0,
            cpp_p50_ms: 0.0,
            rust_p99_ms: 0.0,
            cpp_p99_ms: 0.0,
            batch_sizes_tested: Vec::new(),
            batch_throughput: HashMap::new(),
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            rust_peak_mb: 0.0,
            cpp_peak_mb: 0.0,
            rust_model_size_mb: 0.0,
            cpp_model_size_mb: 0.0,
            rust_inference_overhead_mb: 0.0,
            cpp_inference_overhead_mb: 0.0,
            memory_efficiency_ratio: 0.0,
        }
    }
}

impl Default for CompatibilityReport {
    fn default() -> Self {
        Self {
            model_format_compatible: false,
            api_compatible: false,
            output_compatible: false,
            edge_cases_handled: Vec::new(),
        }
    }
}

impl ValidationReport {
    /// Save report to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load report from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
    
    /// Generate markdown summary
    pub fn to_markdown(&self) -> String {
        let status_emoji = match self.status {
            ValidationStatus::Pass => "✅",
            ValidationStatus::PartialPass => "⚠️",
            ValidationStatus::XFail => "🔧",
            ValidationStatus::Fail => "❌",
        };
        
        format!(
            r#"# Validation Report {}

## Model Information
- **Path**: {}
- **Size**: {:.2} MB
- **GGUF Version**: v{}
- **Architecture**: {}
- **Quantization**: {}

## Loading Results
| Implementation | Success | Time (ms) | Memory (MB) |
|---------------|---------|-----------|-------------|
| Rust | {} | {:.1} | {:.1} |
| C++ | {} | {:.1} | {:.1} |

## Accuracy Metrics
- **Token Match Rate**: {:.2}%
- **Edit Distance**: {:.2}
- **Deterministic**: {}

## Performance Metrics
- **Rust**: {:.1} tokens/sec
- **C++**: {:.1} tokens/sec
- **Speedup**: {:.2}x

## Memory Efficiency
- **Rust Peak**: {:.1} MB
- **C++ Peak**: {:.1} MB
- **Efficiency Ratio**: {:.2}x

## Notes
{}
"#,
            status_emoji,
            self.model_path.display(),
            self.model_metadata.file_size as f64 / 1_048_576.0,
            self.model_metadata.gguf_version,
            self.model_metadata.architecture.as_deref().unwrap_or("unknown"),
            self.model_metadata.quantization.as_deref().unwrap_or("unknown"),
            if self.rust_load.success { "✅" } else { "❌" },
            self.rust_load.load_time_ms,
            self.rust_load.memory_used_mb,
            if self.cpp_load.success { "✅" } else { "❌" },
            self.cpp_load.load_time_ms,
            self.cpp_load.memory_used_mb,
            self.accuracy.token_match_rate * 100.0,
            self.accuracy.edit_distance,
            if self.accuracy.deterministic { "✅" } else { "❌" },
            self.performance.rust_tokens_per_sec,
            self.performance.cpp_tokens_per_sec,
            self.performance.speedup_factor,
            self.memory.rust_peak_mb,
            self.memory.cpp_peak_mb,
            self.memory.memory_efficiency_ratio,
            self.notes.join("\n- ")
        )
    }
}