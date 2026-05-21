/// A candidate workgroup configuration for kernel dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkgroupConfig {
    pub size: [u32; 3],
    pub label: String,
}

impl WorkgroupConfig {
    pub fn new(size: [u32; 3], label: impl Into<String>) -> Self {
        Self { size, label: label.into() }
    }

    /// Total number of invocations per workgroup.
    pub fn total_invocations(&self) -> u32 {
        self.size[0] * self.size[1] * self.size[2]
    }
}
