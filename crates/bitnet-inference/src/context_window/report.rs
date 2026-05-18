use super::ContextWindow;

/// Context usage report.
#[derive(Debug, Clone)]
pub struct ContextReport {
    pub max_length: usize,
    pub used: usize,
    pub remaining: usize,
    pub utilization: f64,
}

impl ContextWindow {
    pub fn report(&self) -> ContextReport {
        ContextReport {
            max_length: self.max_length(),
            used: self.current_length(),
            remaining: self.remaining(),
            utilization: self.utilization(),
        }
    }
}
