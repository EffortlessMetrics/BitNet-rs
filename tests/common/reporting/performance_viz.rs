//! Performance visualization and analysis components
//!
//! This module provides comprehensive performance visualization capabilities including:
//! - Performance metrics visualization
//! - Rust vs C++ performance comparison charts
//! - Performance trend analysis and reporting
//! - Performance regression detection
//! - Interactive performance dashboards

use super::dashboard::BenchmarkResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime};
use tokio::fs;

/// Performance measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: SystemTime,
}

/// Performance comparison data between implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub rust_metrics: PerformanceMetrics,
    pub cpp_metrics: PerformanceMetrics,
    pub comparison_results: ComparisonResults,
    pub timestamp: SystemTime,
}

/// Performance metrics for a single implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub implementation_name: String,
    pub average_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub memory_peak: u64,
    pub memory_average: u64,
    pub throughput_ops_per_sec: f64,
    pub custom_metrics: HashMap<String, f64>,
}

/// Results of performance comparison analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    pub throughput_ratio: f64,        // rust_throughput / cpp_throughput
    pub memory_ratio: f64,            // rust_memory / cpp_memory
    pub duration_ratio: f64,          // rust_duration / cpp_duration
    pub performance_improvement: f64, // percentage improvement (negative = regression)
    pub memory_improvement: f64,      // percentage improvement (negative = regression)
    pub regression_detected: bool,
    pub regression_threshold: f64,
}

/// Performance trend data over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub data_points: Vec<TrendDataPoint>,
    pub trend_direction: TrendDirection,
    pub regression_points: Vec<usize>, // indices of regression points
}

/// Single data point in a performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendDataPoint {
    pub timestamp: SystemTime,
    pub value: f64,
    pub implementation: String,
    pub test_name: String,
}

/// Direction of performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}
/// Performance visualization generator
pub struct PerformanceVisualizer {
    config: VisualizationConfig,
    historical_data: Vec<PerformanceComparison>,
}

/// Configuration for performance visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub regression_threshold: f64, // percentage threshold for regression detection
    pub trend_window_size: usize,  // number of data points for trend analysis
    pub include_interactive_charts: bool,
    pub chart_width: u32,
    pub chart_height: u32,
    pub color_scheme: ColorScheme,
}

/// Color scheme for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    pub rust_color: String,
    pub cpp_color: String,
    pub improvement_color: String,
    pub regression_color: String,
    pub neutral_color: String,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            regression_threshold: 5.0, // 5% regression threshold
            trend_window_size: 20,
            include_interactive_charts: true,
            chart_width: 800,
            chart_height: 400,
            color_scheme: ColorScheme {
                rust_color: "#e74c3c".to_string(),
                cpp_color: "#3498db".to_string(),
                improvement_color: "#27ae60".to_string(),
                regression_color: "#e74c3c".to_string(),
                neutral_color: "#95a5a6".to_string(),
            },
        }
    }
}

impl PerformanceVisualizer {
    /// Create a new performance visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            historical_data: Vec::new(),
        }
    }

    /// Add performance comparison data
    pub fn add_comparison_data(&mut self, comparison: PerformanceComparison) {
        self.historical_data.push(comparison);

        // Keep only the last N data points based on trend window size
        if self.historical_data.len() > self.config.trend_window_size * 2 {
            self.historical_data.drain(0..self.config.trend_window_size);
        }
    }

    /// Generate comprehensive performance visualization HTML
    pub async fn generate_performance_dashboard(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let html_content = self.generate_dashboard_html().await?;

        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(output_path, html_content).await?;
        Ok(())
    }

    /// Generate the complete dashboard HTML
    async fn generate_dashboard_html(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut html = String::new();

        // HTML document structure
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str("    <title>BitNet.rs Performance Dashboard</title>\n");
        html.push_str(&self.generate_dashboard_css());

        if self.config.include_interactive_charts {
            html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n");
            html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js\"></script>\n");
        }

        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // Dashboard header
        html.push_str(&self.generate_dashboard_header());

        // Performance summary cards
        html.push_str(&self.generate_summary_cards());

        // Performance comparison charts
        html.push_str(&self.generate_comparison_charts());

        // Trend analysis section
        html.push_str(&self.generate_trend_analysis());

        // Regression detection section
        html.push_str(&self.generate_regression_analysis());

        // Interactive controls (if enabled)
        if self.config.include_interactive_charts {
            html.push_str(&self.generate_interactive_controls());
            html.push_str(&self.generate_dashboard_javascript());
        }

        html.push_str("</body>\n");
        html.push_str("</html>\n");

        Ok(html)
    }

    /// Generate CSS styles for the dashboard
    fn generate_dashboard_css(&self) -> String {
        format!(
            r#"
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .dashboard-header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .summary-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }}
        
        .summary-card h3 {{
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}
        
        .summary-card.improvement h3 {{ color: {improvement_color}; }}
        .summary-card.regression h3 {{ color: {regression_color}; }}
        .summary-card.neutral h3 {{ color: {neutral_color}; }}
        
        .summary-card .metric-label {{
            font-size: 0.9rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .summary-card .metric-change {{
            font-size: 0.8rem;
            margin-top: 0.5rem;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-weight: 500;
        }}
        
        .metric-change.positive {{
            background: #d4edda;
            color: #155724;
        }}
        
        .metric-change.negative {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .chart-section {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        .chart-section h2 {{
            margin-bottom: 1.5rem;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 0.5rem;
        }}
        
        .chart-container {{
            position: relative;
            height: {chart_height}px;
            margin: 1rem 0;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}
        
        .regression-alert {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        .regression-alert h4 {{
            margin-bottom: 0.5rem;
        }}
        
        .trend-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        .trend-indicator.improving {{
            background: #d4edda;
            color: #155724;
        }}
        
        .trend-indicator.degrading {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .trend-indicator.stable {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .trend-indicator.volatile {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .controls-panel {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        .controls-panel h3 {{
            margin-bottom: 1rem;
            color: #2c3e50;
        }}
        
        .control-group {{
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .control-group label {{
            font-weight: 500;
            min-width: 120px;
        }}
        
        .control-group select,
        .control-group input {{
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        
        .btn {{
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: background-color 0.2s;
        }}
        
        .btn-primary {{
            background: #007bff;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #0056b3;
        }}
        
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        .performance-table th,
        .performance-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .performance-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        
        .performance-table tr:hover {{
            background: #f8f9fa;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .dashboard-header h1 {{
                font-size: 2rem;
            }}
            
            .summary-cards {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            
            .control-group {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
"#,
            improvement_color = self.config.color_scheme.improvement_color,
            regression_color = self.config.color_scheme.regression_color,
            neutral_color = self.config.color_scheme.neutral_color,
            chart_height = self.config.chart_height
        )
    }

    /// Generate dashboard header
    fn generate_dashboard_header(&self) -> String {
        let latest_comparison = self.historical_data.last();
        let data_points = self.historical_data.len();

        format!(
            r#"
    <div class="container">
        <div class="dashboard-header">
            <h1>BitNet.rs Performance Dashboard</h1>
            <div class="subtitle">
                Real-time performance monitoring and regression detection | {} data points
                {}
            </div>
        </div>
"#,
            data_points,
            if let Some(comparison) = latest_comparison {
                format!(
                    "| Last updated: {}",
                    comparison
                        .timestamp
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                )
            } else {
                "| No data available".to_string()
            }
        )
    }

    /// Generate summary cards showing key performance metrics
    fn generate_summary_cards(&self) -> String {
        if self.historical_data.is_empty() {
            return r#"
        <div class="summary-cards">
            <div class="summary-card neutral">
                <h3>No Data</h3>
                <p class="metric-label">Performance data not available</p>
            </div>
        </div>
"#
            .to_string();
        }

        let latest = self.historical_data.last().unwrap();
        let performance_change = if self.historical_data.len() > 1 {
            let previous = &self.historical_data[self.historical_data.len() - 2];
            latest.comparison_results.performance_improvement
                - previous.comparison_results.performance_improvement
        } else {
            0.0
        };

        let memory_change = if self.historical_data.len() > 1 {
            let previous = &self.historical_data[self.historical_data.len() - 2];
            latest.comparison_results.memory_improvement
                - previous.comparison_results.memory_improvement
        } else {
            0.0
        };

        format!(
            r#"
        <div class="summary-cards">
            <div class="summary-card {}">
                <h3>{:.1}%</h3>
                <p class="metric-label">Performance vs C++</p>
                <div class="metric-change {}">
                    {} {:.1}% from last run
                </div>
            </div>
            <div class="summary-card {}">
                <h3>{:.1}%</h3>
                <p class="metric-label">Memory Efficiency</p>
                <div class="metric-change {}">
                    {} {:.1}% from last run
                </div>
            </div>
            <div class="summary-card {}">
                <h3>{:.2}x</h3>
                <p class="metric-label">Throughput Ratio</p>
                <div class="metric-change neutral">
                    Rust vs C++ throughput
                </div>
            </div>
            <div class="summary-card {}">
                <h3>{}</h3>
                <p class="metric-label">Regression Status</p>
                <div class="metric-change {}">
                    Threshold: {:.1}%
                </div>
            </div>
        </div>
"#,
            if latest.comparison_results.performance_improvement > 0.0 {
                "improvement"
            } else {
                "regression"
            },
            latest.comparison_results.performance_improvement,
            if performance_change >= 0.0 {
                "positive"
            } else {
                "negative"
            },
            if performance_change >= 0.0 {
                "↑"
            } else {
                "↓"
            },
            performance_change.abs(),
            if latest.comparison_results.memory_improvement > 0.0 {
                "improvement"
            } else {
                "regression"
            },
            latest.comparison_results.memory_improvement,
            if memory_change >= 0.0 {
                "positive"
            } else {
                "negative"
            },
            if memory_change >= 0.0 { "↑" } else { "↓" },
            memory_change.abs(),
            "neutral",
            latest.comparison_results.throughput_ratio,
            if latest.comparison_results.regression_detected {
                "regression"
            } else {
                "improvement"
            },
            if latest.comparison_results.regression_detected {
                "DETECTED"
            } else {
                "CLEAR"
            },
            if latest.comparison_results.regression_detected {
                "negative"
            } else {
                "positive"
            },
            self.config.regression_threshold
        )
    }
    /// Generate performance comparison charts
    fn generate_comparison_charts(&self) -> String {
        let mut html = String::new();

        html.push_str(
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
                    <h3>Execution Time Comparison</h3>
                    <div class="chart-container">
                        <canvas id="durationChart"></canvas>
                    </div>
                </div>
                <div>
                    <h3>Performance Improvement Over Time</h3>
                    <div class="chart-container">
                        <canvas id="improvementChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
"#,
        );

        html
    }

    /// Generate trend analysis section
    fn generate_trend_analysis(&self) -> String {
        let trends = self.analyze_performance_trends();

        let mut html = String::new();
        html.push_str(
            r#"
        <div class="chart-section">
            <h2>Performance Trend Analysis</h2>
"#,
        );

        if trends.is_empty() {
            html.push_str(
                r#"
            <p>Insufficient data for trend analysis. Need at least 5 data points.</p>
"#,
            );
        } else {
            html.push_str(
                r#"
            <div class="performance-table-container">
                <table class="performance-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Trend Direction</th>
                            <th>Data Points</th>
                            <th>Regression Points</th>
                        </tr>
                    </thead>
                    <tbody>
"#,
            );

            for trend in &trends {
                let trend_class = match trend.trend_direction {
                    TrendDirection::Improving => "improving",
                    TrendDirection::Degrading => "degrading",
                    TrendDirection::Stable => "stable",
                    TrendDirection::Volatile => "volatile",
                };

                html.push_str(&format!(
                    r#"
                        <tr>
                            <td>{}</td>
                            <td>
                                <span class="trend-indicator {}">
                                    {} {:?}
                                </span>
                            </td>
                            <td>{}</td>
                            <td>{}</td>
                        </tr>
"#,
                    trend.metric_name,
                    trend_class,
                    match trend.trend_direction {
                        TrendDirection::Improving => "↗",
                        TrendDirection::Degrading => "↘",
                        TrendDirection::Stable => "→",
                        TrendDirection::Volatile => "↕",
                    },
                    trend.trend_direction,
                    trend.data_points.len(),
                    trend.regression_points.len()
                ));
            }

            html.push_str(
                r#"
                    </tbody>
                </table>
            </div>
"#,
            );
        }

        html.push_str("        </div>\n");
        html
    }

    /// Generate regression analysis section
    fn generate_regression_analysis(&self) -> String {
        let regressions = self.detect_regressions();

        let mut html = String::new();
        html.push_str(
            r#"
        <div class="chart-section">
            <h2>Regression Detection</h2>
"#,
        );

        if regressions.is_empty() {
            html.push_str(
                r#"
            <div class="alert alert-success">
                <h4>✅ No Performance Regressions Detected</h4>
                <p>All performance metrics are within acceptable thresholds.</p>
            </div>
"#,
            );
        } else {
            for regression in &regressions {
                html.push_str(&format!(
                    r#"
            <div class="regression-alert">
                <h4>⚠️ Performance Regression Detected</h4>
                <p><strong>Metric:</strong> {}</p>
                <p><strong>Regression:</strong> {:.1}% (threshold: {:.1}%)</p>
                <p><strong>Implementation:</strong> {}</p>
            </div>
"#,
                    regression.metric_name,
                    regression.regression_percentage,
                    self.config.regression_threshold,
                    regression.implementation
                ));
            }
        }

        html.push_str("        </div>\n");
        html
    }

    /// Generate interactive controls
    fn generate_interactive_controls(&self) -> String {
        r#"
        <div class="controls-panel">
            <h3>Dashboard Controls</h3>
            <div class="control-group">
                <label for="timeRange">Time Range:</label>
                <select id="timeRange">
                    <option value="24h">Last 24 Hours</option>
                    <option value="7d">Last 7 Days</option>
                    <option value="30d" selected>Last 30 Days</option>
                    <option value="all">All Time</option>
                </select>
            </div>
            <div class="control-group">
                <label for="metricFilter">Metric Filter:</label>
                <select id="metricFilter">
                    <option value="all" selected>All Metrics</option>
                    <option value="throughput">Throughput</option>
                    <option value="memory">Memory Usage</option>
                    <option value="duration">Execution Time</option>
                </select>
            </div>
            <div class="control-group">
                <label for="regressionThreshold">Regression Threshold (%):</label>
                <input type="number" id="regressionThreshold" value="5.0" min="0.1" max="50.0" step="0.1">
                <button class="btn btn-primary" onclick="updateRegressionThreshold()">Update</button>
            </div>
            <div class="control-group">
                <button class="btn btn-primary" onclick="refreshDashboard()">Refresh Data</button>
                <button class="btn btn-primary" onclick="exportData()">Export Data</button>
            </div>
        </div>
"#.to_string()
    }

    /// Generate JavaScript for interactive features
    fn generate_dashboard_javascript(&self) -> String {
        let chart_data = self.generate_chart_data_json();

        format!(
            r#"
    <script>
        // Performance data
        const performanceData = {};
        
        // Chart instances
        let charts = {{}};
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
            setupEventListeners();
        }});
        
        function initializeCharts() {{
            // Throughput comparison chart
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            charts.throughput = new Chart(throughputCtx, {{
                type: 'line',
                data: {{
                    labels: performanceData.timestamps,
                    datasets: [{{
                        label: 'Rust Throughput',
                        data: performanceData.rust_throughput,
                        borderColor: '{}',
                        backgroundColor: '{}20',
                        tension: 0.4
                    }}, {{
                        label: 'C++ Throughput',
                        data: performanceData.cpp_throughput,
                        borderColor: '{}',
                        backgroundColor: '{}20',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Operations per Second'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Throughput Comparison Over Time'
                        }}
                    }}
                }}
            }});
            
            // Memory usage chart
            const memoryCtx = document.getElementById('memoryChart').getContext('2d');
            charts.memory = new Chart(memoryCtx, {{
                type: 'bar',
                data: {{
                    labels: performanceData.timestamps,
                    datasets: [{{
                        label: 'Rust Memory (MB)',
                        data: performanceData.rust_memory,
                        backgroundColor: '{}80',
                        borderColor: '{}',
                        borderWidth: 1
                    }}, {{
                        label: 'C++ Memory (MB)',
                        data: performanceData.cpp_memory,
                        backgroundColor: '{}80',
                        borderColor: '{}',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Memory Usage (MB)'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Memory Usage Comparison'
                        }}
                    }}
                }}
            }});
            
            // Duration comparison chart
            const durationCtx = document.getElementById('durationChart').getContext('2d');
            charts.duration = new Chart(durationCtx, {{
                type: 'line',
                data: {{
                    labels: performanceData.timestamps,
                    datasets: [{{
                        label: 'Rust Duration (ms)',
                        data: performanceData.rust_duration,
                        borderColor: '{}',
                        backgroundColor: '{}20',
                        tension: 0.4
                    }}, {{
                        label: 'C++ Duration (ms)',
                        data: performanceData.cpp_duration,
                        borderColor: '{}',
                        backgroundColor: '{}20',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Execution Time (ms)'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Execution Time Comparison'
                        }}
                    }}
                }}
            }});
            
            // Performance improvement chart
            const improvementCtx = document.getElementById('improvementChart').getContext('2d');
            charts.improvement = new Chart(improvementCtx, {{
                type: 'line',
                data: {{
                    labels: performanceData.timestamps,
                    datasets: [{{
                        label: 'Performance Improvement (%)',
                        data: performanceData.performance_improvement,
                        borderColor: '{}',
                        backgroundColor: function(context) {{
                            const value = context.parsed.y;
                            return value >= 0 ? '{}40' : '{}40';
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
                            title: {{
                                display: true,
                                text: 'Improvement (%)'
                            }},
                            grid: {{
                                color: function(context) {{
                                    return context.tick.value === 0 ? '#000' : '#e0e0e0';
                                }}
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Performance Improvement Trend'
                        }}
                    }}
                }}
            }});
        }}
        
        function setupEventListeners() {{
            document.getElementById('timeRange').addEventListener('change', updateTimeRange);
            document.getElementById('metricFilter').addEventListener('change', updateMetricFilter);
        }}
        
        function updateTimeRange() {{
            const range = document.getElementById('timeRange').value;
            // Filter data based on time range and update charts
            console.log('Updating time range to:', range);
        }}
        
        function updateMetricFilter() {{
            const filter = document.getElementById('metricFilter').value;
            // Filter metrics and update charts
            console.log('Updating metric filter to:', filter);
        }}
        
        function updateRegressionThreshold() {{
            const threshold = document.getElementById('regressionThreshold').value;
            // Update regression detection threshold
            console.log('Updating regression threshold to:', threshold);
        }}
        
        function refreshDashboard() {{
            // Refresh dashboard data
            location.reload();
        }}
        
        function exportData() {{
            // Export performance data as JSON
            const dataStr = JSON.stringify(performanceData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'performance_data.json';
            link.click();
            URL.revokeObjectURL(url);
        }}
    </script>
"#,
            chart_data,
            self.config.color_scheme.rust_color,
            self.config.color_scheme.rust_color,
            self.config.color_scheme.cpp_color,
            self.config.color_scheme.cpp_color,
            self.config.color_scheme.rust_color,
            self.config.color_scheme.rust_color,
            self.config.color_scheme.cpp_color,
            self.config.color_scheme.cpp_color,
            self.config.color_scheme.rust_color,
            self.config.color_scheme.rust_color,
            self.config.color_scheme.cpp_color,
            self.config.color_scheme.cpp_color,
            self.config.color_scheme.neutral_color,
            self.config.color_scheme.improvement_color,
            self.config.color_scheme.regression_color
        )
    }

    /// Generate chart data as JSON
    fn generate_chart_data_json(&self) -> String {
        if self.historical_data.is_empty() {
            return "{}".to_string();
        }

        let timestamps: Vec<String> = self
            .historical_data
            .iter()
            .map(|d| {
                d.timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    .to_string()
            })
            .collect();

        let rust_throughput: Vec<f64> = self
            .historical_data
            .iter()
            .map(|d| d.rust_metrics.throughput_ops_per_sec)
            .collect();

        let cpp_throughput: Vec<f64> = self
            .historical_data
            .iter()
            .map(|d| d.cpp_metrics.throughput_ops_per_sec)
            .collect();

        let rust_memory: Vec<f64> = self.historical_data
            .iter()
            .map(|d| d.rust_metrics.memory_peak as f64 / 1024.0 / 1024.0) // Convert to MB
            .collect();

        let cpp_memory: Vec<f64> = self.historical_data
            .iter()
            .map(|d| d.cpp_metrics.memory_peak as f64 / 1024.0 / 1024.0) // Convert to MB
            .collect();

        let rust_duration: Vec<f64> = self
            .historical_data
            .iter()
            .map(|d| d.rust_metrics.average_duration.as_millis() as f64)
            .collect();

        let cpp_duration: Vec<f64> = self
            .historical_data
            .iter()
            .map(|d| d.cpp_metrics.average_duration.as_millis() as f64)
            .collect();

        let performance_improvement: Vec<f64> = self
            .historical_data
            .iter()
            .map(|d| d.comparison_results.performance_improvement)
            .collect();

        format!(
            r#"{{
    "timestamps": {:?},
    "rust_throughput": {:?},
    "cpp_throughput": {:?},
    "rust_memory": {:?},
    "cpp_memory": {:?},
    "rust_duration": {:?},
    "cpp_duration": {:?},
    "performance_improvement": {:?}
}}"#,
            timestamps,
            rust_throughput,
            cpp_throughput,
            rust_memory,
            cpp_memory,
            rust_duration,
            cpp_duration,
            performance_improvement
        )
    }

    /// Analyze performance trends
    fn analyze_performance_trends(&self) -> Vec<PerformanceTrend> {
        if self.historical_data.len() < 5 {
            return Vec::new();
        }

        let mut trends = Vec::new();

        // Analyze throughput trend
        let throughput_data: Vec<TrendDataPoint> = self
            .historical_data
            .iter()
            .enumerate()
            .flat_map(|(i, d)| {
                vec![
                    TrendDataPoint {
                        timestamp: d.timestamp,
                        value: d.rust_metrics.throughput_ops_per_sec,
                        implementation: "Rust".to_string(),
                        test_name: format!("throughput_{}", i),
                    },
                    TrendDataPoint {
                        timestamp: d.timestamp,
                        value: d.cpp_metrics.throughput_ops_per_sec,
                        implementation: "C++".to_string(),
                        test_name: format!("throughput_{}", i),
                    },
                ]
            })
            .collect();

        trends.push(PerformanceTrend {
            metric_name: "Throughput".to_string(),
            data_points: throughput_data.clone(),
            trend_direction: self.calculate_trend_direction(&throughput_data),
            regression_points: self.find_regression_points(&throughput_data),
        });

        // Analyze memory trend
        let memory_data: Vec<TrendDataPoint> = self
            .historical_data
            .iter()
            .enumerate()
            .flat_map(|(i, d)| {
                vec![
                    TrendDataPoint {
                        timestamp: d.timestamp,
                        value: d.rust_metrics.memory_peak as f64,
                        implementation: "Rust".to_string(),
                        test_name: format!("memory_{}", i),
                    },
                    TrendDataPoint {
                        timestamp: d.timestamp,
                        value: d.cpp_metrics.memory_peak as f64,
                        implementation: "C++".to_string(),
                        test_name: format!("memory_{}", i),
                    },
                ]
            })
            .collect();

        trends.push(PerformanceTrend {
            metric_name: "Memory Usage".to_string(),
            data_points: memory_data.clone(),
            trend_direction: self.calculate_trend_direction(&memory_data),
            regression_points: self.find_regression_points(&memory_data),
        });

        trends
    }

    /// Calculate trend direction from data points
    fn calculate_trend_direction(&self, data_points: &[TrendDataPoint]) -> TrendDirection {
        if data_points.len() < 3 {
            return TrendDirection::Stable;
        }

        let values: Vec<f64> = data_points.iter().map(|p| p.value).collect();
        let n = values.len() as f64;

        // Calculate linear regression slope
        let x_mean = (0..values.len()).map(|i| i as f64).sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = (0..values.len())
            .map(|i| (i as f64 - x_mean) * (values[i] - y_mean))
            .sum();

        let denominator: f64 = (0..values.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

        if denominator == 0.0 {
            return TrendDirection::Stable;
        }

        let slope = numerator / denominator;

        // Calculate variance to determine volatility
        let variance = values.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / y_mean.abs();

        // Determine trend direction
        if coefficient_of_variation > 0.3 {
            TrendDirection::Volatile
        } else if slope.abs() < 0.01 {
            TrendDirection::Stable
        } else if slope > 0.0 {
            TrendDirection::Improving
        } else {
            TrendDirection::Degrading
        }
    }

    /// Find regression points in the data
    fn find_regression_points(&self, data_points: &[TrendDataPoint]) -> Vec<usize> {
        let mut regression_points = Vec::new();

        if data_points.len() < 3 {
            return regression_points;
        }

        for i in 1..data_points.len() - 1 {
            let prev_value = data_points[i - 1].value;
            let curr_value = data_points[i].value;
            let next_value = data_points[i + 1].value;

            // Check for significant drop in performance
            let drop_percentage = ((prev_value - curr_value) / prev_value) * 100.0;
            let recovery_percentage = ((next_value - curr_value) / curr_value) * 100.0;

            if drop_percentage > self.config.regression_threshold && recovery_percentage < 5.0 {
                regression_points.push(i);
            }
        }

        regression_points
    }

    /// Detect performance regressions
    fn detect_regressions(&self) -> Vec<RegressionAlert> {
        let mut regressions = Vec::new();

        if self.historical_data.len() < 2 {
            return regressions;
        }

        let latest = self.historical_data.last().unwrap();

        if latest.comparison_results.regression_detected {
            regressions.push(RegressionAlert {
                metric_name: "Overall Performance".to_string(),
                regression_percentage: -latest.comparison_results.performance_improvement,
                implementation: "Rust".to_string(),
                timestamp: latest.timestamp,
            });
        }

        if latest.comparison_results.memory_improvement < -self.config.regression_threshold {
            regressions.push(RegressionAlert {
                metric_name: "Memory Efficiency".to_string(),
                regression_percentage: -latest.comparison_results.memory_improvement,
                implementation: "Rust".to_string(),
                timestamp: latest.timestamp,
            });
        }

        regressions
    }
}

/// Alert for detected performance regression
#[derive(Debug, Clone)]
pub struct RegressionAlert {
    pub metric_name: String,
    pub regression_percentage: f64,
    pub implementation: String,
    pub timestamp: SystemTime,
}

/// Create performance comparison from benchmark results
pub fn create_performance_comparison(
    rust_benchmark: &BenchmarkResult,
    cpp_benchmark: &BenchmarkResult,
    regression_threshold: f64,
) -> PerformanceComparison {
    let rust_metrics = PerformanceMetrics {
        implementation_name: "BitNet.rs".to_string(),
        average_duration: rust_benchmark.summary.avg_duration,
        min_duration: rust_benchmark.summary.min_duration,
        max_duration: rust_benchmark.summary.max_duration,
        memory_peak: rust_benchmark.summary.peak_memory_usage as u64,
        memory_average: rust_benchmark.summary.avg_memory_usage as u64,
        throughput_ops_per_sec: rust_benchmark.ops_per_second(),
        custom_metrics: rust_benchmark
            .summary
            .custom_metrics
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
    };

    let cpp_metrics = PerformanceMetrics {
        implementation_name: "BitNet.cpp".to_string(),
        average_duration: cpp_benchmark.summary.avg_duration,
        min_duration: cpp_benchmark.summary.min_duration,
        max_duration: cpp_benchmark.summary.max_duration,
        memory_peak: cpp_benchmark.summary.peak_memory_usage as u64,
        memory_average: cpp_benchmark.summary.avg_memory_usage as u64,
        throughput_ops_per_sec: cpp_benchmark.ops_per_second(),
        custom_metrics: cpp_benchmark
            .summary
            .custom_metrics
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
    };

    // Calculate comparison results
    let throughput_ratio = if cpp_metrics.throughput_ops_per_sec > 0.0 {
        rust_metrics.throughput_ops_per_sec / cpp_metrics.throughput_ops_per_sec
    } else {
        1.0
    };

    let memory_ratio = if cpp_metrics.memory_peak > 0 {
        rust_metrics.memory_peak as f64 / cpp_metrics.memory_peak as f64
    } else {
        1.0
    };

    let duration_ratio = if cpp_metrics.average_duration.as_secs_f64() > 0.0 {
        rust_metrics.average_duration.as_secs_f64() / cpp_metrics.average_duration.as_secs_f64()
    } else {
        1.0
    };

    let performance_improvement = (throughput_ratio - 1.0) * 100.0;
    let memory_improvement = (1.0 - memory_ratio) * 100.0;
    let regression_detected = performance_improvement < -regression_threshold;

    let comparison_results = ComparisonResults {
        throughput_ratio,
        memory_ratio,
        duration_ratio,
        performance_improvement,
        memory_improvement,
        regression_detected,
        regression_threshold,
    };

    PerformanceComparison {
        rust_metrics,
        cpp_metrics,
        comparison_results,
        timestamp: SystemTime::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_benchmark_result(
        name: &str,
        ops_per_sec: f64,
        memory_mb: u64,
    ) -> BenchmarkResult {
        use crate::data::performance::{MetricSummary, PerformanceSummary};

        BenchmarkResult {
            name: name.to_string(),
            iterations: 10,
            warmup_iterations: 3,
            summary: PerformanceSummary {
                count: 10,
                avg_duration: Some(Duration::from_millis((1000.0 / ops_per_sec) as u64)),
                min_duration: Some(Duration::from_millis((900.0 / ops_per_sec) as u64)),
                max_duration: Some(Duration::from_millis((1100.0 / ops_per_sec) as u64)),
                avg_memory_usage: Some(memory_mb * 1024 * 1024),
                peak_memory_usage: Some(memory_mb * 1024 * 1024),
                total_memory_allocated: Some(memory_mb * 1024 * 1024),
                custom_metrics: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_performance_comparison_creation() {
        let rust_benchmark = create_test_benchmark_result("rust_test", 100.0, 512);
        let cpp_benchmark = create_test_benchmark_result("cpp_test", 80.0, 600);

        let comparison = create_performance_comparison(&rust_benchmark, &cpp_benchmark, 5.0);

        assert_eq!(comparison.rust_metrics.implementation_name, "BitNet.rs");
        assert_eq!(comparison.cpp_metrics.implementation_name, "BitNet.cpp");
        assert!(comparison.comparison_results.throughput_ratio > 1.0);
        assert!(comparison.comparison_results.performance_improvement > 0.0);
        assert!(!comparison.comparison_results.regression_detected);
    }

    #[test]
    fn test_regression_detection() {
        let rust_benchmark = create_test_benchmark_result("rust_test", 70.0, 700);
        let cpp_benchmark = create_test_benchmark_result("cpp_test", 100.0, 500);

        let comparison = create_performance_comparison(&rust_benchmark, &cpp_benchmark, 5.0);

        assert!(comparison.comparison_results.throughput_ratio < 1.0);
        assert!(comparison.comparison_results.performance_improvement < 0.0);
        assert!(comparison.comparison_results.regression_detected);
    }

    #[tokio::test]
    async fn test_visualizer_creation() {
        let config = VisualizationConfig::default();
        let visualizer = PerformanceVisualizer::new(config);

        assert_eq!(visualizer.historical_data.len(), 0);
        assert_eq!(visualizer.config.regression_threshold, 5.0);
    }

    #[test]
    fn test_trend_direction_calculation() {
        let config = VisualizationConfig::default();
        let visualizer = PerformanceVisualizer::new(config);

        // Test improving trend
        let improving_data = vec![
            TrendDataPoint {
                timestamp: SystemTime::now(),
                value: 10.0,
                implementation: "test".to_string(),
                test_name: "test".to_string(),
            },
            TrendDataPoint {
                timestamp: SystemTime::now(),
                value: 15.0,
                implementation: "test".to_string(),
                test_name: "test".to_string(),
            },
            TrendDataPoint {
                timestamp: SystemTime::now(),
                value: 20.0,
                implementation: "test".to_string(),
                test_name: "test".to_string(),
            },
        ];

        let trend = visualizer.calculate_trend_direction(&improving_data);
        assert!(matches!(trend, TrendDirection::Improving));
    }
}
