use super::WorkgroupConfig;

/// The result of running a single workgroup configuration.
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub config: WorkgroupConfig,
    pub elapsed_us: u64,
    pub throughput: f64,
}
