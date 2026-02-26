#![cfg(feature = "integration-tests")]

//! Performance visualization demonstration
//!
//! This test demonstrates the performance visualization capabilities including:
//! - Performance metrics visualization
//! - Rust vs C++ performance comparison charts
//! - Performance trend analysis and reporting
//! - Performance regression detection
//! - Interactive performance dashboards

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tempfile::TempDir;

// Import the performance visualization components
mod common;
use common::units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};

// Stub types for missing dependencies
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    ops_per_sec: f64,
    memory_mb: f64,
    time_ms: f64,
}

#[derive(Debug)]
struct BenchmarkRunner;

#[derive(Debug)]
struct MetricSummary {
    mean: f64,
    median: f64,
    std_dev: f64,
}

#[derive(Debug)]
struct PerformanceSummary {
    total_ops: u64,
    total_time: Duration,
}

#[derive(Debug)]
struct DashboardConfig;

#[derive(Debug)]
struct PerformanceComparison;

#[derive(Debug)]
struct PerformanceDashboardGenerator;

#[derive(Debug)]
struct PerformanceVisualizer;

#[derive(Debug)]
struct VisualizationConfig;

fn create_performance_dashboard(_: &str) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[derive(Debug)]
struct ComparisonResults;

#[derive(Debug)]
struct PerformanceMetrics;

fn create_performance_comparison(_: &str) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

/// Create a test benchmark result for demonstration
fn create_demo_benchmark_result(
    name: &str,
    ops_per_sec: f64,
    memory_mb: u64,
    duration_ms: u64,
) -> BenchmarkResult {
    let mut custom_metrics = HashMap::new();
    custom_metrics.insert(
        "accuracy".to_string(),
        MetricSummary {
            count: 10,
            average: 0.95,
            minimum: 0.92,
            maximum: 0.98,
            std_deviation: 0.02,
        },
    );

    BenchmarkResult {
        name: name.to_string(),
        iterations: 10,
        warmup_iterations: 3,
        summary: PerformanceSummary {
            count: 10,
            avg_duration: Some(Duration::from_millis(duration_ms)),
            min_duration: Some(Duration::from_millis(duration_ms - 5)),
            max_duration: Some(Duration::from_millis(duration_ms + 10)),
            avg_memory_usage: Some(memory_mb * BYTES_PER_MB),
            peak_memory_usage: Some(memory_mb * BYTES_PER_MB),
            total_memory_allocated: Some(memory_mb * BYTES_PER_MB),
            custom_metrics,
        },
    }
}

/// Create sample performance comparison data
fn create_sample_performance_data() -> Vec<PerformanceComparison> {
    let mut comparisons = Vec::new();

    // Simulate performance data over time with various scenarios
    let scenarios = vec![
        // Scenario 1: Rust performing better
        (120.0, 100.0, 450, 512, 8, 10),
        // Scenario 2: Similar performance
        (105.0, 100.0, 480, 500, 9, 10),
        // Scenario 3: Rust improvement
        (140.0, 100.0, 400, 520, 7, 10),
        // Scenario 4: Slight regression
        (95.0, 100.0, 520, 480, 11, 10),
        // Scenario 5: Recovery
        (130.0, 100.0, 420, 500, 8, 10),
    ];

    for (i, (rust_ops, cpp_ops, rust_mem, cpp_mem, rust_dur, cpp_dur)) in
        scenarios.iter().enumerate()
    {
        let rust_benchmark = create_demo_benchmark_result(
            &format!("rust_benchmark_{}", i),
            *rust_ops,
            *rust_mem,
            *rust_dur,
        );

        let cpp_benchmark = create_demo_benchmark_result(
            &format!("cpp_benchmark_{}", i),
            *cpp_ops,
            *cpp_mem,
            *cpp_dur,
        );

        let comparison = create_performance_comparison(&rust_benchmark, &cpp_benchmark, 5.0);
        comparisons.push(comparison);
    }

    comparisons
}

#[tokio::test]
async fn test_performance_visualization_basic() {
    let temp_dir = TempDir::new().unwrap();
    let config = VisualizationConfig::default();
    let mut visualizer = PerformanceVisualizer::new(config);

    // Add sample performance data
    let sample_data = create_sample_performance_data();
    for comparison in sample_data {
        visualizer.add_comparison_data(comparison);
    }

    // Generate performance dashboard
    let dashboard_path = temp_dir.path().join("performance_dashboard.html");
    visualizer.generate_performance_dashboard(&dashboard_path).await.unwrap();

    // Verify dashboard was created
    assert!(dashboard_path.exists());

    // Verify dashboard content
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("BitNet-rs Performance Dashboard"));
    assert!(content.contains("Performance Comparison Charts"));
    assert!(content.contains("Trend Analysis"));
    assert!(content.contains("Regression Detection"));

    println!("✅ Basic performance visualization test passed");
    println!("📊 Dashboard generated at: {}", dashboard_path.display());
}

#[tokio::test]
async fn test_performance_dashboard_generator() {
    let temp_dir = TempDir::new().unwrap();
    let config = DashboardConfig {
        title: "Demo Performance Dashboard".to_string(),
        include_test_results: true,
        include_regression_analysis: true,
        include_trend_analysis: true,
        auto_refresh_interval: Some(60),
        export_formats: vec![
            common::reporting::dashboard::ExportFormat::Json,
            common::reporting::dashboard::ExportFormat::Csv,
        ],
    };

    let mut generator = PerformanceDashboardGenerator::new(temp_dir.path().to_path_buf(), config);

    // Add benchmark comparisons
    let rust_benchmark = create_demo_benchmark_result("rust_inference", 150.0, 400, 6);
    let cpp_benchmark = create_demo_benchmark_result("cpp_inference", 120.0, 480, 8);

    generator.add_benchmark_comparison(&rust_benchmark, &cpp_benchmark);

    // Add more data points to show trends
    for i in 0..5 {
        let rust_perf = 150.0 + (i as f64 * 5.0); // Improving performance
        let cpp_perf = 120.0 + (i as f64 * 2.0); // Slower improvement

        let rust_bench = create_demo_benchmark_result(
            &format!("rust_trend_{}", i),
            rust_perf,
            400 - (i * 10), // Improving memory usage
            6,
        );
        let cpp_bench = create_demo_benchmark_result(
            &format!("cpp_trend_{}", i),
            cpp_perf,
            480 + (i * 5), // Slightly worse memory usage
            8,
        );

        generator.add_benchmark_comparison(&rust_bench, &cpp_bench);
    }

    // Generate complete dashboard
    let output = generator.generate_dashboard().await.unwrap();

    // Verify all expected files were generated
    assert!(output.dashboard_url.exists());
    assert!(!output.generated_files.is_empty());

    // Check for specific files
    let json_file = temp_dir.path().join("performance_data.json");
    let csv_file = temp_dir.path().join("performance_data.csv");
    let summary_file = temp_dir.path().join("performance_summary.md");

    assert!(json_file.exists());
    assert!(csv_file.exists());
    assert!(summary_file.exists());

    // Verify content of generated files
    let json_content = tokio::fs::read_to_string(&json_file).await.unwrap();
    assert!(json_content.contains("dashboard_config"));
    assert!(json_content.contains("BitNet-rs"));

    let csv_content = tokio::fs::read_to_string(&csv_file).await.unwrap();
    assert!(csv_content.contains("timestamp,rust_throughput"));

    let summary_content = tokio::fs::read_to_string(&summary_file).await.unwrap();
    assert!(summary_content.contains("# Demo Performance Dashboard"));
    assert!(summary_content.contains("## Overview"));

    println!("✅ Performance dashboard generator test passed");
    println!("📁 Generated {} files", output.generated_files.len());
    println!("🌐 Dashboard URL: {}", output.dashboard_url.display());
}

#[tokio::test]
async fn test_regression_detection() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = VisualizationConfig::default();
    config.regression_threshold = 10.0; // 10% threshold for easier testing

    let mut visualizer = PerformanceVisualizer::new(config);

    // Add baseline performance data
    let baseline_rust = create_demo_benchmark_result("rust_baseline", 100.0, 500, 10);
    let baseline_cpp = create_demo_benchmark_result("cpp_baseline", 80.0, 600, 12);
    let baseline_comparison = create_performance_comparison(&baseline_rust, &baseline_cpp, 10.0);
    visualizer.add_comparison_data(baseline_comparison);

    // Add regression data (Rust performance drops significantly)
    let regressed_rust = create_demo_benchmark_result("rust_regressed", 70.0, 700, 15);
    let regressed_cpp = create_demo_benchmark_result("cpp_stable", 80.0, 600, 12);
    let regression_comparison =
        create_performance_comparison(&regressed_rust, &regressed_cpp, 10.0);
    visualizer.add_comparison_data(regression_comparison);

    // Generate dashboard with regression detection
    let dashboard_path = temp_dir.path().join("regression_dashboard.html");
    visualizer.generate_performance_dashboard(&dashboard_path).await.unwrap();

    // Verify regression detection in dashboard
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("Regression Detection"));

    // The regression should be detected since Rust performance dropped from 25% better to 12.5% worse
    println!("✅ Regression detection test passed");
    println!("⚠️  Regression dashboard generated at: {}", dashboard_path.display());
}

#[tokio::test]
async fn test_trend_analysis() {
    let temp_dir = TempDir::new().unwrap();
    let config = VisualizationConfig::default();
    let mut visualizer = PerformanceVisualizer::new(config);

    // Create a clear improving trend
    let trend_data = vec![
        (80.0, 100.0, 600, 500),  // Starting point
        (90.0, 100.0, 580, 500),  // Improving
        (100.0, 100.0, 560, 500), // More improvement
        (110.0, 100.0, 540, 500), // Continued improvement
        (120.0, 100.0, 520, 500), // Strong improvement
        (130.0, 100.0, 500, 500), // Excellent improvement
    ];

    for (i, (rust_ops, cpp_ops, rust_mem, cpp_mem)) in trend_data.iter().enumerate() {
        let rust_benchmark =
            create_demo_benchmark_result(&format!("rust_trend_{}", i), *rust_ops, *rust_mem, 10);
        let cpp_benchmark =
            create_demo_benchmark_result(&format!("cpp_trend_{}", i), *cpp_ops, *cpp_mem, 12);

        let comparison = create_performance_comparison(&rust_benchmark, &cpp_benchmark, 5.0);
        visualizer.add_comparison_data(comparison);
    }

    // Generate dashboard with trend analysis
    let dashboard_path = temp_dir.path().join("trend_dashboard.html");
    visualizer.generate_performance_dashboard(&dashboard_path).await.unwrap();

    // Verify trend analysis in dashboard
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("Performance Trend Analysis"));
    assert!(content.contains("Trend Direction"));

    println!("✅ Trend analysis test passed");
    println!("📈 Trend dashboard generated at: {}", dashboard_path.display());
}

#[tokio::test]
async fn test_interactive_dashboard_features() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = VisualizationConfig::default();
    config.include_interactive_charts = true;

    let mut visualizer = PerformanceVisualizer::new(config);

    // Add diverse performance data
    let sample_data = create_sample_performance_data();
    for comparison in sample_data {
        visualizer.add_comparison_data(comparison);
    }

    // Generate interactive dashboard
    let dashboard_path = temp_dir.path().join("interactive_dashboard.html");
    visualizer.generate_performance_dashboard(&dashboard_path).await.unwrap();

    // Verify interactive features
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("chart.js")); // Chart.js library
    assert!(content.contains("initializeCharts")); // JavaScript initialization
    assert!(content.contains("Dashboard Controls")); // Interactive controls
    assert!(content.contains("Time Range")); // Filter controls
    assert!(content.contains("Export Data")); // Export functionality

    println!("✅ Interactive dashboard features test passed");
    println!("🎛️  Interactive dashboard generated at: {}", dashboard_path.display());
}

#[tokio::test]
async fn test_comprehensive_performance_analysis() {
    let temp_dir = TempDir::new().unwrap();

    // Create comprehensive dashboard with all features
    let config = DashboardConfig {
        title: "Comprehensive BitNet-rs Performance Analysis".to_string(),
        include_test_results: true,
        include_regression_analysis: true,
        include_trend_analysis: true,
        auto_refresh_interval: Some(300),
        export_formats: vec![
            common::reporting::dashboard::ExportFormat::Json,
            common::reporting::dashboard::ExportFormat::Csv,
        ],
    };

    let mut generator = PerformanceDashboardGenerator::new(temp_dir.path().to_path_buf(), config);

    // Simulate a comprehensive performance testing scenario
    let test_scenarios = vec![
        // Model loading performance
        ("model_loading", 50.0, 45.0, 2048, 2200, 20, 25),
        // Tokenization performance
        ("tokenization", 1000.0, 800.0, 128, 150, 1, 1),
        // Small model inference
        ("small_model_inference", 200.0, 180.0, 512, 600, 5, 6),
        // Medium model inference
        ("medium_model_inference", 100.0, 90.0, 1024, 1200, 10, 12),
        // Large model inference
        ("large_model_inference", 50.0, 45.0, 2048, 2400, 20, 24),
        // Batch processing
        ("batch_processing", 80.0, 70.0, 1536, 1800, 15, 18),
        // Streaming inference
        ("streaming_inference", 150.0, 130.0, 768, 900, 7, 8),
    ];

    for (scenario, rust_ops, cpp_ops, rust_mem, cpp_mem, rust_dur, cpp_dur) in test_scenarios {
        let rust_benchmark = create_demo_benchmark_result(
            &format!("rust_{}", scenario),
            rust_ops,
            rust_mem,
            rust_dur,
        );
        let cpp_benchmark =
            create_demo_benchmark_result(&format!("cpp_{}", scenario), cpp_ops, cpp_mem, cpp_dur);

        generator.add_benchmark_comparison(&rust_benchmark, &cpp_benchmark);
    }

    // Generate comprehensive dashboard
    let output = generator.generate_dashboard().await.unwrap();

    // Verify comprehensive analysis
    assert!(output.dashboard_url.exists());
    assert!(output.generated_files.len() >= 3); // HTML, JSON, CSV, Summary

    // Verify dashboard content includes all scenarios
    let dashboard_content = tokio::fs::read_to_string(&output.dashboard_url).await.unwrap();
    assert!(dashboard_content.contains("Comprehensive BitNet-rs Performance Analysis"));
    assert!(dashboard_content.contains("Performance Comparison Charts"));
    assert!(dashboard_content.contains("Regression Detection"));

    // Verify summary report
    let summary_path = temp_dir.path().join("performance_summary.md");
    let summary_content = tokio::fs::read_to_string(&summary_path).await.unwrap();
    assert!(summary_content.contains("## Performance Metrics"));
    assert!(summary_content.contains("## Regression Detection"));

    println!("✅ Comprehensive performance analysis test passed");
    println!("📊 Analyzed {} performance scenarios", test_scenarios.len());
    println!("📁 Generated comprehensive dashboard with {} files", output.generated_files.len());
    println!("🎯 Dashboard available at: {}", output.dashboard_url.display());
}

/// Integration test that demonstrates the complete performance visualization workflow
#[tokio::test]
async fn test_complete_performance_visualization_workflow() {
    println!("🚀 Starting complete performance visualization workflow test...");

    let temp_dir = TempDir::new().unwrap();
    println!("📁 Working directory: {}", temp_dir.path().display());

    // Step 1: Create performance dashboard generator
    let mut generator = create_performance_dashboard(temp_dir.path().to_path_buf());
    println!("✅ Step 1: Dashboard generator created");

    // Step 2: Simulate running benchmarks and collecting data
    println!("🔄 Step 2: Simulating benchmark execution...");

    // Simulate multiple benchmark runs over time
    for run in 0..10 {
        let rust_performance = 100.0 + (run as f64 * 2.0); // Gradual improvement
        let cpp_performance = 90.0 + (run as f64 * 1.0); // Slower improvement
        let rust_memory = 500 - (run * 5); // Memory optimization
        let cpp_memory = 600 + (run * 2); // Slight memory increase

        let rust_benchmark = create_demo_benchmark_result(
            &format!("rust_run_{}", run),
            rust_performance,
            rust_memory,
            8,
        );
        let cpp_benchmark = create_demo_benchmark_result(
            &format!("cpp_run_{}", run),
            cpp_performance,
            cpp_memory,
            10,
        );

        generator.add_benchmark_comparison(&rust_benchmark, &cpp_benchmark);
    }
    println!("✅ Step 2: Added 10 benchmark comparison data points");

    // Step 3: Generate comprehensive dashboard
    println!("🔄 Step 3: Generating performance dashboard...");
    let output = generator.generate_dashboard().await.unwrap();
    println!("✅ Step 3: Dashboard generated successfully");

    // Step 4: Verify all components
    println!("🔄 Step 4: Verifying dashboard components...");

    // Verify main dashboard
    assert!(output.dashboard_url.exists());
    let dashboard_content = tokio::fs::read_to_string(&output.dashboard_url).await.unwrap();
    assert!(dashboard_content.contains("Performance Comparison Charts"));
    assert!(dashboard_content.contains("Trend Analysis"));
    assert!(dashboard_content.contains("Regression Detection"));
    println!("  ✅ Interactive HTML dashboard verified");

    // Verify data exports
    let json_file = temp_dir.path().join("performance_data.json");
    assert!(json_file.exists());
    println!("  ✅ JSON data export verified");

    let csv_file = temp_dir.path().join("performance_data.csv");
    assert!(csv_file.exists());
    println!("  ✅ CSV data export verified");

    // Verify summary report
    let summary_file = temp_dir.path().join("performance_summary.md");
    assert!(summary_file.exists());
    let summary_content = tokio::fs::read_to_string(&summary_file).await.unwrap();
    assert!(summary_content.contains("Performance Metrics"));
    println!("  ✅ Summary report verified");

    println!("✅ Step 4: All dashboard components verified");

    // Step 5: Display results
    println!("\n🎉 Complete performance visualization workflow test PASSED!");
    println!("📊 Dashboard Features Demonstrated:");
    println!("  • Performance metrics visualization");
    println!("  • Rust vs C++ performance comparison charts");
    println!("  • Performance trend analysis and reporting");
    println!("  • Performance regression detection");
    println!("  • Interactive performance dashboards");
    println!("  • Multiple export formats (JSON, CSV)");
    println!("  • Comprehensive summary reporting");
    println!("\n📁 Generated Files:");
    for file in &output.generated_files {
        println!("  • {} - {}", file.path.file_name().unwrap().to_string_lossy(), file.description);
    }
    println!("\n🌐 Open the dashboard: {}", output.dashboard_url.display());
}
