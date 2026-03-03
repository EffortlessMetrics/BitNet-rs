//! Layer executor for per-layer forward pass orchestration.
//!
//! Manages execution of transformer layers with timing and diagnostics.

use std::time::{Duration, Instant};

/// Layer type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    Embedding,
    Attention,
    FeedForward,
    Normalization,
    Output,
    Custom,
}

impl LayerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Attention => "attention",
            Self::FeedForward => "feed_forward",
            Self::Normalization => "normalization",
            Self::Output => "output",
            Self::Custom => "custom",
        }
    }
}

/// Execution timing for one layer.
#[derive(Debug, Clone)]
pub struct LayerTiming {
    pub layer_index: usize,
    pub layer_type: LayerType,
    pub duration: Duration,
    pub input_elements: usize,
    pub output_elements: usize,
}

impl LayerTiming {
    pub fn throughput(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.output_elements as f64 / secs
    }
}

/// Execution plan for a model.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub layers: Vec<LayerSpec>,
    pub total_params: u64,
}

/// Specification of a single layer.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub index: usize,
    pub layer_type: LayerType,
    pub param_count: u64,
    pub name: String,
}

impl ExecutionPlan {
    pub fn new() -> Self {
        Self { layers: Vec::new(), total_params: 0 }
    }

    pub fn add_layer(&mut self, name: &str, layer_type: LayerType, params: u64) {
        let index = self.layers.len();
        self.layers.push(LayerSpec {
            index,
            layer_type,
            param_count: params,
            name: name.to_string(),
        });
        self.total_params += params;
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn params_by_type(&self) -> Vec<(LayerType, u64)> {
        let mut map = std::collections::HashMap::new();
        for l in &self.layers {
            *map.entry(l.layer_type).or_insert(0u64) += l.param_count;
        }
        let mut result: Vec<_> = map.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution trace from running a forward pass.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub timings: Vec<LayerTiming>,
    pub total_duration: Duration,
}

impl ExecutionTrace {
    pub fn new() -> Self {
        Self { timings: Vec::new(), total_duration: Duration::ZERO }
    }

    pub fn record(&mut self, timing: LayerTiming) {
        self.total_duration += timing.duration;
        self.timings.push(timing);
    }

    pub fn slowest(&self, n: usize) -> Vec<&LayerTiming> {
        let mut sorted: Vec<_> = self.timings.iter().collect();
        sorted.sort_by(|a, b| b.duration.cmp(&a.duration));
        sorted.truncate(n);
        sorted
    }

    pub fn by_type(&self, lt: LayerType) -> Vec<&LayerTiming> {
        self.timings.iter().filter(|t| t.layer_type == lt).collect()
    }

    pub fn avg_duration(&self) -> Duration {
        if self.timings.is_empty() {
            return Duration::ZERO;
        }
        self.total_duration / self.timings.len() as u32
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer executor that runs layers and collects timing.
#[derive(Debug)]
pub struct LayerExecutor {
    plan: ExecutionPlan,
    trace: ExecutionTrace,
}

impl LayerExecutor {
    pub fn new(plan: ExecutionPlan) -> Self {
        Self { plan, trace: ExecutionTrace::new() }
    }

    /// Simulate executing a layer and record timing.
    pub fn execute_layer<F>(
        &mut self,
        layer_index: usize,
        input_elements: usize,
        output_elements: usize,
        f: F,
    ) where
        F: FnOnce(),
    {
        let layer_type =
            self.plan.layers.get(layer_index).map(|l| l.layer_type).unwrap_or(LayerType::Custom);
        let start = Instant::now();
        f();
        let duration = start.elapsed();
        self.trace.record(LayerTiming {
            layer_index,
            layer_type,
            duration,
            input_elements,
            output_elements,
        });
    }

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    pub fn trace(&self) -> &ExecutionTrace {
        &self.trace
    }

    pub fn into_trace(self) -> ExecutionTrace {
        self.trace
    }
}

/// Build a standard transformer execution plan.
pub fn build_transformer_plan(
    num_layers: usize,
    hidden_size: u64,
    intermediate_size: u64,
    vocab_size: u64,
) -> ExecutionPlan {
    let mut plan = ExecutionPlan::new();
    plan.add_layer("embedding", LayerType::Embedding, vocab_size * hidden_size);
    for i in 0..num_layers {
        plan.add_layer(&format!("layer_{i}_norm"), LayerType::Normalization, hidden_size);
        plan.add_layer(
            &format!("layer_{i}_attn"),
            LayerType::Attention,
            4 * hidden_size * hidden_size,
        );
        plan.add_layer(
            &format!("layer_{i}_ffn"),
            LayerType::FeedForward,
            2 * hidden_size * intermediate_size,
        );
    }
    plan.add_layer("output", LayerType::Output, vocab_size * hidden_size);
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_str() {
        assert_eq!(LayerType::Attention.as_str(), "attention");
        assert_eq!(LayerType::Embedding.as_str(), "embedding");
    }

    #[test]
    fn test_execution_plan_add() {
        let mut plan = ExecutionPlan::new();
        plan.add_layer("test", LayerType::Attention, 1000);
        assert_eq!(plan.layer_count(), 1);
        assert_eq!(plan.total_params, 1000);
    }

    #[test]
    fn test_params_by_type() {
        let mut plan = ExecutionPlan::new();
        plan.add_layer("a1", LayerType::Attention, 100);
        plan.add_layer("a2", LayerType::Attention, 200);
        plan.add_layer("f1", LayerType::FeedForward, 50);
        let by_type = plan.params_by_type();
        assert_eq!(by_type[0].1, 300); // attention total
    }

    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new();
        trace.record(LayerTiming {
            layer_index: 0,
            layer_type: LayerType::Attention,
            duration: Duration::from_millis(10),
            input_elements: 100,
            output_elements: 100,
        });
        assert_eq!(trace.timings.len(), 1);
        assert_eq!(trace.total_duration, Duration::from_millis(10));
    }

    #[test]
    fn test_slowest() {
        let mut trace = ExecutionTrace::new();
        trace.record(LayerTiming {
            layer_index: 0,
            layer_type: LayerType::Attention,
            duration: Duration::from_millis(5),
            input_elements: 10,
            output_elements: 10,
        });
        trace.record(LayerTiming {
            layer_index: 1,
            layer_type: LayerType::FeedForward,
            duration: Duration::from_millis(20),
            input_elements: 10,
            output_elements: 10,
        });
        let s = trace.slowest(1);
        assert_eq!(s[0].layer_index, 1);
    }

    #[test]
    fn test_by_type() {
        let mut trace = ExecutionTrace::new();
        trace.record(LayerTiming {
            layer_index: 0,
            layer_type: LayerType::Attention,
            duration: Duration::from_millis(1),
            input_elements: 10,
            output_elements: 10,
        });
        trace.record(LayerTiming {
            layer_index: 1,
            layer_type: LayerType::FeedForward,
            duration: Duration::from_millis(1),
            input_elements: 10,
            output_elements: 10,
        });
        assert_eq!(trace.by_type(LayerType::Attention).len(), 1);
    }

    #[test]
    fn test_layer_executor() {
        let mut plan = ExecutionPlan::new();
        plan.add_layer("attn", LayerType::Attention, 100);
        let mut exec = LayerExecutor::new(plan);
        exec.execute_layer(0, 64, 64, || { /* simulate */ });
        assert_eq!(exec.trace().timings.len(), 1);
    }

    #[test]
    fn test_build_transformer_plan() {
        let plan = build_transformer_plan(2, 512, 2048, 32000);
        // embedding + 2*(norm + attn + ffn) + output = 1 + 6 + 1 = 8
        assert_eq!(plan.layer_count(), 8);
        assert!(plan.total_params > 0);
    }

    #[test]
    fn test_throughput() {
        let timing = LayerTiming {
            layer_index: 0,
            layer_type: LayerType::Attention,
            duration: Duration::from_secs(1),
            input_elements: 100,
            output_elements: 1000,
        };
        assert!((timing.throughput() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_avg_duration() {
        let mut trace = ExecutionTrace::new();
        trace.record(LayerTiming {
            layer_index: 0,
            layer_type: LayerType::Attention,
            duration: Duration::from_millis(10),
            input_elements: 10,
            output_elements: 10,
        });
        trace.record(LayerTiming {
            layer_index: 1,
            layer_type: LayerType::Attention,
            duration: Duration::from_millis(20),
            input_elements: 10,
            output_elements: 10,
        });
        assert_eq!(trace.avg_duration(), Duration::from_millis(15));
    }

    #[test]
    fn test_into_trace() {
        let plan = ExecutionPlan::new();
        let exec = LayerExecutor::new(plan);
        let trace = exec.into_trace();
        assert!(trace.timings.is_empty());
    }

    #[test]
    fn test_empty_trace_defaults() {
        let trace = ExecutionTrace::new();
        assert_eq!(trace.avg_duration(), Duration::ZERO);
        assert!(trace.slowest(5).is_empty());
    }
}
