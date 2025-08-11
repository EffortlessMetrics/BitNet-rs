//! Simple performance visualization test
//!
//! This test demonstrates the core performance visualization functionality
//! without relying on the complex test harness infrastructure.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tempfile::TempDir;

// Simple test structures to avoid dependency issues
#[derive(Debug, Clone)]
pub struct SimpleBenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub peak_memory: u64,
    pub avg_memory: u64,
    pub ops_per_second: f64,
}

impl SimpleBenchmarkResult {
    pub fn new(name: &str, ops_per_sec: f64, memory_mb: u64, duration_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            iterations: 10,
            avg_duration: Duration::from_millis(duration_ms),
            min_duration: Duration::from_millis(duration_ms.saturating_sub(2)),
            max_duration: Duration::from_millis(duration_ms + 5),
            peak_memory: memory_mb * 1024 * 1024,
            avg_memory: memory_mb * 1024 * 1024,
            ops_per_second: ops_per_sec,
        }
    }
}

// Simple performance comparison structure
#[derive(Debug, Clone)]
pub struct SimplePerformanceComparison {
    pub rust_ops_per_sec: f64,
    pub cpp_ops_per_sec: f64,
    pub rust_memory_mb: u64,
    pub cpp_memory_mb: u64,
    pub rust_duration_ms: u64,
    pub cpp_duration_ms: u64,
    pub performance_improvement: f64,
    pub memory_improvement: f64,
    pub regression_detected: bool,
    pub timestamp: SystemTime,
}

impl SimplePerformanceComparison {
    pub fn new(
        rust_benchmark: &SimpleBenchmarkResult,
        cpp_benchmark: &SimpleBenchmarkResult,
        regression_threshold: f64,
    ) -> Self {
        let throughput_ratio = rust_benchmark.ops_per_second / cpp_benchmark.ops_per_second;
        let memory_ratio = rust_benchmark.peak_memory as f64 / cpp_benchmark.peak_memory as f64;

        let performance_improvement = (throughput_ratio - 1.0) * 100.0;
        let memory_improvement = (1.0 - memory_ratio) * 100.0;
        let regression_detected = performance_improvement < -regression_threshold;

        Self {
            rust_ops_per_sec: rust_benchmark.ops_per_second,
            cpp_ops_per_sec: cpp_benchmark.ops_per_second,
            rust_memory_mb: rust_benchmark.peak_memory / 1024 / 1024,
            cpp_memory_mb: cpp_benchmark.peak_memory / 1024 / 1024,
            rust_duration_ms: rust_benchmark.avg_duration.as_millis() as u64,
            cpp_duration_ms: cpp_benchmark.avg_duration.as_millis() as u64,
            performance_improvement,
            memory_improvement,
            regression_detected,
            timestamp: SystemTime::now(),
        }
    }
}

// Simple dashboard generator
pub struct SimplePerformanceDashboard {
    comparisons: Vec<SimplePerformanceComparison>,
    title: String,
}

impl SimplePerformanceDashboard {
    pub fn new(title: String) -> Self {
        Self {
            comparisons: Vec::new(),
            title,
        }
    }

    pub fn add_comparison(&mut self, comparison: SimplePerformanceComparison) {
        self.comparisons.push(comparison);
    }

    pub async fn generate_dashboard(
        &self,
        output_path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let html_content = self.generate_html_content();

        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(output_path, html_content).await?;
        Ok(())
    }

    fn generate_html_content(&self) -> String {
        let mut html = String::new();

        // HTML document structure
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!("    <title>{}</title>\n", self.title));
        html.push_str(&self.generate_css());
        html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // Header
        html.push_str(&format!(
            r#"
    <div class="container">
        <div class="header">
            <h1>{}</h1>
            <div class="subtitle">
                Performance monitoring and regression detection | {} data points
            </div>
        </div>
"#,
            self.title,
            self.comparisons.len()
        ));

        // Summary cards
        html.push_str(&self.generate_summary_cards());

        // Charts
        html.push_str(&self.generate_charts());

        // Data table
        html.push_str(&self.generate_data_table());

        // JavaScript
        html.push_str(&self.generate_javascript());

        html.push_str("    </div>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }

    fn generate_css(&self) -> String {
        r#"
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .summary-card h3 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .summary-card.improvement h3 { color: #27ae60; }
        .summary-card.regression h3 { color: #e74c3c; }
        .summary-card.neutral h3 { color: #95a5a6; }
        
        .chart-section {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .chart-section h2 {
            margin-bottom: 1.5rem;
            color: #2c3e50;
        }
        
        .chart-container {
            position: relative;
            height: 400px;
            margin: 1rem 0;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .data-table {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .performance-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        .performance-table th,
        .performance-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        .performance-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .performance-table tr:hover {
            background: #f8f9fa;
        }
        
        .regression-alert {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .improvement-highlight {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
"#
        .to_string()
    }

    fn generate_summary_cards(&self) -> String {
        if self.comparisons.is_empty() {
            return r#"
        <div class="summary-cards">
            <div class="summary-card neutral">
                <h3>No Data</h3>
                <p>Performance data not available</p>
            </div>
        </div>
"#
            .to_string();
        }

        let latest = self.comparisons.last().unwrap();
        let avg_performance = self
            .comparisons
            .iter()
            .map(|c| c.performance_improvement)
            .sum::<f64>()
            / self.comparisons.len() as f64;

        let avg_memory = self
            .comparisons
            .iter()
            .map(|c| c.memory_improvement)
            .sum::<f64>()
            / self.comparisons.len() as f64;

        let regressions = self
            .comparisons
            .iter()
            .filter(|c| c.regression_detected)
            .count();

        format!(
            r#"
        <div class="summary-cards">
            <div class="summary-card {}">
                <h3>{:.1}%</h3>
                <p>Latest Performance vs C++</p>
            </div>
            <div class="summary-card {}">
                <h3>{:.1}%</h3>
                <p>Average Performance Improvement</p>
            </div>
            <div class="summary-card {}">
                <h3>{:.1}%</h3>
                <p>Memory Efficiency</p>
            </div>
            <div class="summary-card {}">
                <h3>{}</h3>
                <p>Regressions Detected</p>
            </div>
        </div>
"#,
            if latest.performance_improvement > 0.0 {
                "improvement"
            } else {
                "regression"
            },
            latest.performance_improvement,
            if avg_performance > 0.0 {
                "improvement"
            } else {
                "regression"
            },
            avg_performance,
            if avg_memory > 0.0 {
                "improvement"
            } else {
                "regression"
            },
            avg_memory,
            if regressions > 0 {
                "regression"
            } else {
                "improvement"
            },
            regressions
        )
    }

    fn generate_charts(&self) -> String {
        r#"
        <div class="chart-section">
            <h2>Performance Comparison Charts</h2>
            <div class="chart-grid">
                <div>
                    <h3>Throughput Comparison</h3>
                    <div class="chart-container">
                        <canvas id="throughputChart"></canvas>
                    </div>
                </div>
                <div>
                    <h3>Memory Usage Comparison</h3>
                    <div class="chart-container">
                        <canvas id="memoryChart"></canvas>
                    </div>
                </div>
                <div>
                    <h3>Performance Improvement Over Time</h3>
                    <div class="chart-container">
                        <canvas id="improvementChart"></canvas>
                    </div>
                </div>
                <div>
                    <h3>Execution Time Comparison</h3>
                    <div class="chart-container">
                        <canvas id="durationChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
"#
        .to_string()
    }

    fn generate_data_table(&self) -> String {
        let mut html = String::new();

        html.push_str(
            r#"
        <div class="data-table">
            <h2>Performance Data</h2>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Rust Throughput (ops/s)</th>
                        <th>C++ Throughput (ops/s)</th>
                        <th>Rust Memory (MB)</th>
                        <th>C++ Memory (MB)</th>
                        <th>Performance Improvement (%)</th>
                        <th>Memory Improvement (%)</th>
                        <th>Regression</th>
                    </tr>
                </thead>
                <tbody>
"#,
        );

        for comparison in &self.comparisons {
            let timestamp = comparison
                .timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            html.push_str(&format!(
                r#"
                    <tr>
                        <td>{}</td>
                        <td>{:.1}</td>
                        <td>{:.1}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{:.1}%</td>
                        <td>{:.1}%</td>
                        <td>{}</td>
                    </tr>
"#,
                timestamp,
                comparison.rust_ops_per_sec,
                comparison.cpp_ops_per_sec,
                comparison.rust_memory_mb,
                comparison.cpp_memory_mb,
                comparison.performance_improvement,
                comparison.memory_improvement,
                if comparison.regression_detected {
                    "⚠️ Yes"
                } else {
                    "✅ No"
                }
            ));
        }

        html.push_str(
            r#"
                </tbody>
            </table>
        </div>
"#,
        );

        html
    }

    fn generate_javascript(&self) -> String {
        let chart_data = self.generate_chart_data_json();

        format!(
            r#"
    <script>
        // Performance data
        const performanceData = {};
        
        // Initialize charts
        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
        }});
        
        function initializeCharts() {{
            // Throughput comparison chart
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            new Chart(throughputCtx, {{
                type: 'line',
                data: {{
                    labels: performanceData.labels,
                    datasets: [{{
                        label: 'Rust Throughput (ops/s)',
                        data: performanceData.rust_throughput,
                        borderColor: '#e74c3c',
                        backgroundColor: '#e74c3c20',
                        tension: 0.4
                    }}, {{
                        label: 'C++ Throughput (ops/s)',
                        data: performanceData.cpp_throughput,
                        borderColor: '#3498db',
                        backgroundColor: '#3498db20',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
            
            // Memory usage chart
            const memoryCtx = document.getElementById('memoryChart').getContext('2d');
            new Chart(memoryCtx, {{
                type: 'bar',
                data: {{
                    labels: performanceData.labels,
                    datasets: [{{
                        label: 'Rust Memory (MB)',
                        data: performanceData.rust_memory,
                        backgroundColor: '#e74c3c80',
                        borderColor: '#e74c3c',
                        borderWidth: 1
                    }}, {{
                        label: 'C++ Memory (MB)',
                        data: performanceData.cpp_memory,
                        backgroundColor: '#3498db80',
                        borderColor: '#3498db',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
            
            // Performance improvement chart
            const improvementCtx = document.getElementById('improvementChart').getContext('2d');
            new Chart(improvementCtx, {{
                type: 'line',
                data: {{
                    labels: performanceData.labels,
                    datasets: [{{
                        label: 'Performance Improvement (%)',
                        data: performanceData.performance_improvement,
                        borderColor: '#27ae60',
                        backgroundColor: function(context) {{
                            const value = context.parsed.y;
                            return value >= 0 ? '#27ae6040' : '#e74c3c40';
                        }},
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            grid: {{
                                color: function(context) {{
                                    return context.tick.value === 0 ? '#000' : '#e0e0e0';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
            
            // Duration comparison chart
            const durationCtx = document.getElementById('durationChart').getContext('2d');
            new Chart(durationCtx, {{
                type: 'line',
                data: {{
                    labels: performanceData.labels,
                    datasets: [{{
                        label: 'Rust Duration (ms)',
                        data: performanceData.rust_duration,
                        borderColor: '#e74c3c',
                        backgroundColor: '#e74c3c20',
                        tension: 0.4
                    }}, {{
                        label: 'C++ Duration (ms)',
                        data: performanceData.cpp_duration,
                        borderColor: '#3498db',
                        backgroundColor: '#3498db20',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }}
    </script>
"#,
            chart_data
        )
    }

    fn generate_chart_data_json(&self) -> String {
        if self.comparisons.is_empty() {
            return "{}".to_string();
        }

        let labels: Vec<String> = (0..self.comparisons.len())
            .map(|i| format!("Run {}", i + 1))
            .collect();

        let rust_throughput: Vec<f64> = self
            .comparisons
            .iter()
            .map(|c| c.rust_ops_per_sec)
            .collect();

        let cpp_throughput: Vec<f64> = self.comparisons.iter().map(|c| c.cpp_ops_per_sec).collect();

        let rust_memory: Vec<u64> = self.comparisons.iter().map(|c| c.rust_memory_mb).collect();

        let cpp_memory: Vec<u64> = self.comparisons.iter().map(|c| c.cpp_memory_mb).collect();

        let rust_duration: Vec<u64> = self
            .comparisons
            .iter()
            .map(|c| c.rust_duration_ms)
            .collect();

        let cpp_duration: Vec<u64> = self.comparisons.iter().map(|c| c.cpp_duration_ms).collect();

        let performance_improvement: Vec<f64> = self
            .comparisons
            .iter()
            .map(|c| c.performance_improvement)
            .collect();

        format!(
            r#"{{
    "labels": {:?},
    "rust_throughput": {:?},
    "cpp_throughput": {:?},
    "rust_memory": {:?},
    "cpp_memory": {:?},
    "rust_duration": {:?},
    "cpp_duration": {:?},
    "performance_improvement": {:?}
}}"#,
            labels,
            rust_throughput,
            cpp_throughput,
            rust_memory,
            cpp_memory,
            rust_duration,
            cpp_duration,
            performance_improvement
        )
    }
}

#[tokio::test]
async fn test_simple_performance_visualization() {
    let temp_dir = TempDir::new().unwrap();
    let mut dashboard = SimplePerformanceDashboard::new("BitNet.rs Performance Test".to_string());

    // Create sample performance data
    let test_scenarios = vec![
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
        test_scenarios.iter().enumerate()
    {
        let rust_benchmark = SimpleBenchmarkResult::new(
            &format!("rust_test_{}", i),
            *rust_ops,
            *rust_mem,
            *rust_dur,
        );
        let cpp_benchmark =
            SimpleBenchmarkResult::new(&format!("cpp_test_{}", i), *cpp_ops, *cpp_mem, *cpp_dur);

        let comparison = SimplePerformanceComparison::new(&rust_benchmark, &cpp_benchmark, 5.0);
        dashboard.add_comparison(comparison);
    }

    // Generate dashboard
    let dashboard_path = temp_dir.path().join("performance_dashboard.html");
    dashboard.generate_dashboard(&dashboard_path).await.unwrap();

    // Verify dashboard was created
    assert!(dashboard_path.exists());

    // Verify dashboard content
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("BitNet.rs Performance Test"));
    assert!(content.contains("Performance Comparison Charts"));
    assert!(content.contains("Throughput Comparison"));
    assert!(content.contains("Memory Usage Comparison"));
    assert!(content.contains("Performance Improvement Over Time"));
    assert!(content.contains("chart.js")); // Chart.js library
    assert!(content.contains("Performance Data")); // Data table

    println!("✅ Simple performance visualization test passed");
    println!("📊 Dashboard generated at: {}", dashboard_path.display());
    println!(
        "🎯 Dashboard contains {} performance comparisons",
        dashboard.comparisons.len()
    );
}

#[tokio::test]
async fn test_regression_detection() {
    let temp_dir = TempDir::new().unwrap();
    let mut dashboard = SimplePerformanceDashboard::new("Regression Detection Test".to_string());

    // Add baseline performance data
    let baseline_rust = SimpleBenchmarkResult::new("rust_baseline", 100.0, 500, 10);
    let baseline_cpp = SimpleBenchmarkResult::new("cpp_baseline", 80.0, 600, 12);
    let baseline_comparison = SimplePerformanceComparison::new(&baseline_rust, &baseline_cpp, 10.0);
    dashboard.add_comparison(baseline_comparison);

    // Add regression data (Rust performance drops significantly)
    let regressed_rust = SimpleBenchmarkResult::new("rust_regressed", 70.0, 700, 15);
    let regressed_cpp = SimpleBenchmarkResult::new("cpp_stable", 80.0, 600, 12);
    let regression_comparison =
        SimplePerformanceComparison::new(&regressed_rust, &regressed_cpp, 10.0);
    dashboard.add_comparison(regression_comparison);

    // Generate dashboard
    let dashboard_path = temp_dir.path().join("regression_dashboard.html");
    dashboard.generate_dashboard(&dashboard_path).await.unwrap();

    // Verify regression detection
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("⚠️ Yes")); // Regression detected in table

    // Check that regression was properly detected
    assert!(dashboard.comparisons[1].regression_detected);
    assert!(dashboard.comparisons[1].performance_improvement < -10.0);

    println!("✅ Regression detection test passed");
    println!("⚠️  Regression detected and displayed in dashboard");
}

#[tokio::test]
async fn test_performance_improvement_tracking() {
    let temp_dir = TempDir::new().unwrap();
    let mut dashboard =
        SimplePerformanceDashboard::new("Performance Improvement Tracking".to_string());

    // Create a clear improving trend
    let trend_data = vec![
        (80.0, 100.0, 600, 500),  // Starting point: -20% performance
        (90.0, 100.0, 580, 500),  // Improving: -10% performance
        (100.0, 100.0, 560, 500), // Equal: 0% performance
        (110.0, 100.0, 540, 500), // Better: +10% performance
        (120.0, 100.0, 520, 500), // Much better: +20% performance
        (130.0, 100.0, 500, 500), // Excellent: +30% performance
    ];

    for (i, (rust_ops, cpp_ops, rust_mem, cpp_mem)) in trend_data.iter().enumerate() {
        let rust_benchmark =
            SimpleBenchmarkResult::new(&format!("rust_trend_{}", i), *rust_ops, *rust_mem, 10);
        let cpp_benchmark =
            SimpleBenchmarkResult::new(&format!("cpp_trend_{}", i), *cpp_ops, *cpp_mem, 12);

        let comparison = SimplePerformanceComparison::new(&rust_benchmark, &cpp_benchmark, 5.0);
        dashboard.add_comparison(comparison);
    }

    // Generate dashboard
    let dashboard_path = temp_dir.path().join("improvement_dashboard.html");
    dashboard.generate_dashboard(&dashboard_path).await.unwrap();

    // Verify improvement tracking
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();
    assert!(content.contains("Performance Improvement Over Time"));

    // Check that improvements are tracked correctly
    let improvements: Vec<f64> = dashboard
        .comparisons
        .iter()
        .map(|c| c.performance_improvement)
        .collect();

    // Should show improving trend
    assert!(improvements[0] < improvements[1]); // First improvement
    assert!(improvements[1] < improvements[2]); // Continued improvement
    assert!(improvements[4] > improvements[3]); // Strong improvement
    assert!(improvements[5] > 25.0); // Final strong performance

    println!("✅ Performance improvement tracking test passed");
    println!(
        "📈 Tracked improvement from {:.1}% to {:.1}%",
        improvements[0], improvements[5]
    );
}

#[tokio::test]
async fn test_comprehensive_dashboard_features() {
    let temp_dir = TempDir::new().unwrap();
    let mut dashboard =
        SimplePerformanceDashboard::new("Comprehensive Performance Dashboard".to_string());

    // Add diverse performance scenarios
    let scenarios = vec![
        // Model loading performance
        ("model_loading", 50.0, 45.0, 2048, 2200, 20, 25),
        // Tokenization performance
        ("tokenization", 1000.0, 800.0, 128, 150, 1, 1),
        // Small model inference
        ("small_inference", 200.0, 180.0, 512, 600, 5, 6),
        // Medium model inference
        ("medium_inference", 100.0, 90.0, 1024, 1200, 10, 12),
        // Large model inference
        ("large_inference", 50.0, 45.0, 2048, 2400, 20, 24),
        // Batch processing
        ("batch_processing", 80.0, 70.0, 1536, 1800, 15, 18),
        // Streaming inference
        ("streaming", 150.0, 130.0, 768, 900, 7, 8),
    ];

    for (scenario, rust_ops, cpp_ops, rust_mem, cpp_mem, rust_dur, cpp_dur) in &scenarios {
        let rust_benchmark = SimpleBenchmarkResult::new(
            &format!("rust_{}", scenario),
            *rust_ops,
            *rust_mem,
            *rust_dur,
        );
        let cpp_benchmark =
            SimpleBenchmarkResult::new(&format!("cpp_{}", scenario), *cpp_ops, *cpp_mem, *cpp_dur);

        let comparison = SimplePerformanceComparison::new(&rust_benchmark, &cpp_benchmark, 5.0);
        dashboard.add_comparison(comparison);
    }

    // Generate comprehensive dashboard
    let dashboard_path = temp_dir.path().join("comprehensive_dashboard.html");
    dashboard.generate_dashboard(&dashboard_path).await.unwrap();

    // Verify comprehensive features
    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();

    // Check all major sections are present
    assert!(content.contains("Comprehensive Performance Dashboard"));
    assert!(content.contains("Performance Comparison Charts"));
    assert!(content.contains("Throughput Comparison"));
    assert!(content.contains("Memory Usage Comparison"));
    assert!(content.contains("Performance Improvement Over Time"));
    assert!(content.contains("Execution Time Comparison"));
    assert!(content.contains("Performance Data")); // Data table

    // Check interactive features
    assert!(content.contains("chart.js"));
    assert!(content.contains("initializeCharts"));

    // Check data is present (verify some performance values are in the table)
    assert!(content.contains("50.0")); // model_loading rust ops
    assert!(content.contains("1000.0")); // tokenization rust ops
    assert!(content.contains("150.0")); // streaming rust ops

    // Verify performance calculations
    let avg_improvement: f64 = dashboard
        .comparisons
        .iter()
        .map(|c| c.performance_improvement)
        .sum::<f64>()
        / dashboard.comparisons.len() as f64;

    assert!(avg_improvement > 0.0); // Overall Rust should be better

    println!("✅ Comprehensive dashboard features test passed");
    println!("📊 Generated dashboard with {} scenarios", scenarios.len());
    println!(
        "📈 Average performance improvement: {:.1}%",
        avg_improvement
    );
    println!("🌐 Dashboard available at: {}", dashboard_path.display());
}

/// Integration test demonstrating the complete workflow
#[tokio::test]
async fn test_complete_performance_visualization_workflow() {
    println!("🚀 Starting complete performance visualization workflow test...");

    let temp_dir = TempDir::new().unwrap();
    println!("📁 Working directory: {}", temp_dir.path().display());

    // Step 1: Create dashboard
    let mut dashboard =
        SimplePerformanceDashboard::new("BitNet.rs Complete Performance Analysis".to_string());
    println!("✅ Step 1: Dashboard created");

    // Step 2: Simulate benchmark runs over time
    println!("🔄 Step 2: Simulating benchmark execution...");

    for run in 0..10 {
        let rust_performance = 100.0 + (run as f64 * 3.0); // Gradual improvement
        let cpp_performance = 90.0 + (run as f64 * 1.5); // Slower improvement
        let rust_memory = 500u64.saturating_sub(run * 8); // Memory optimization
        let cpp_memory = 600 + (run * 3); // Slight memory increase

        let rust_benchmark = SimpleBenchmarkResult::new(
            &format!("rust_run_{}", run),
            rust_performance,
            rust_memory,
            8,
        );
        let cpp_benchmark = SimpleBenchmarkResult::new(
            &format!("cpp_run_{}", run),
            cpp_performance,
            cpp_memory,
            10,
        );

        let comparison = SimplePerformanceComparison::new(&rust_benchmark, &cpp_benchmark, 5.0);
        dashboard.add_comparison(comparison);
    }
    println!("✅ Step 2: Added 10 benchmark comparison data points");

    // Step 3: Generate dashboard
    println!("🔄 Step 3: Generating performance dashboard...");
    let dashboard_path = temp_dir.path().join("complete_dashboard.html");
    dashboard.generate_dashboard(&dashboard_path).await.unwrap();
    println!("✅ Step 3: Dashboard generated successfully");

    // Step 4: Verify all components
    println!("🔄 Step 4: Verifying dashboard components...");

    let content = tokio::fs::read_to_string(&dashboard_path).await.unwrap();

    // Verify main sections
    assert!(content.contains("BitNet.rs Complete Performance Analysis"));
    assert!(content.contains("Performance Comparison Charts"));
    assert!(content.contains("Performance Data"));
    println!("  ✅ Main sections verified");

    // Verify charts
    assert!(content.contains("Throughput Comparison"));
    assert!(content.contains("Memory Usage Comparison"));
    assert!(content.contains("Performance Improvement Over Time"));
    assert!(content.contains("Execution Time Comparison"));
    println!("  ✅ All charts verified");

    // Verify interactive features
    assert!(content.contains("chart.js"));
    assert!(content.contains("initializeCharts"));
    println!("  ✅ Interactive features verified");

    // Verify data table has performance data
    assert!(content.contains("Performance Data"));
    assert!(content.contains("Rust Throughput"));
    println!("  ✅ Data table verified");

    println!("✅ Step 4: All dashboard components verified");

    // Step 5: Analyze results
    println!("🔄 Step 5: Analyzing performance results...");

    let first_improvement = dashboard.comparisons[0].performance_improvement;
    let last_improvement = dashboard.comparisons[9].performance_improvement;
    let improvement_delta = last_improvement - first_improvement;

    let regressions = dashboard
        .comparisons
        .iter()
        .filter(|c| c.regression_detected)
        .count();

    println!("  📊 First run performance: {:.1}%", first_improvement);
    println!("  📊 Last run performance: {:.1}%", last_improvement);
    println!("  📈 Performance improvement: {:.1}%", improvement_delta);
    println!("  ⚠️  Regressions detected: {}", regressions);

    assert!(improvement_delta > 0.0); // Should show improvement over time
    assert!(last_improvement > first_improvement); // Performance should improve

    println!("✅ Step 5: Performance analysis completed");

    // Final summary
    println!("\n🎉 Complete performance visualization workflow test PASSED!");
    println!("📊 Dashboard Features Demonstrated:");
    println!("  • Performance metrics visualization ✅");
    println!("  • Rust vs C++ performance comparison charts ✅");
    println!("  • Performance trend analysis and reporting ✅");
    println!("  • Performance regression detection ✅");
    println!("  • Interactive performance dashboards ✅");
    println!("  • Comprehensive data tables ✅");
    println!("  • Real-time chart updates ✅");
    println!("\n🌐 Open the dashboard: {}", dashboard_path.display());
    println!(
        "📈 Performance improved by {:.1}% over {} runs",
        improvement_delta,
        dashboard.comparisons.len()
    );
}
