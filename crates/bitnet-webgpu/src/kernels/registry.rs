//! Runtime kernel registry for WGSL shaders.

/// Metadata for a single registered kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelEntry {
    /// Human-readable kernel name (e.g. `"matmul"`).
    pub name: &'static str,
    /// WGSL shader source.
    pub source: &'static str,
    /// Entry-point function name inside the WGSL source.
    pub entry_point: &'static str,
    /// Workgroup size hint `(x, y, z)`.
    pub workgroup_size: (u32, u32, u32),
}

/// The static kernel registry.
pub static REGISTRY: KernelRegistry = KernelRegistry::new();

/// A registry mapping kernel names to their WGSL source and metadata.
pub struct KernelRegistry {
    entries: &'static [KernelEntry],
}

impl KernelRegistry {
    const fn new() -> Self {
        Self {
            entries: &[
                KernelEntry {
                    name: "matmul",
                    source: super::MATMUL_WGSL,
                    entry_point: "main",
                    workgroup_size: (16, 16, 1),
                },
                KernelEntry {
                    name: "softmax",
                    source: super::SOFTMAX_WGSL,
                    entry_point: "main",
                    workgroup_size: (256, 1, 1),
                },
                KernelEntry {
                    name: "attention",
                    source: super::ATTENTION_WGSL,
                    entry_point: "main",
                    workgroup_size: (256, 1, 1),
                },
                KernelEntry {
                    name: "rmsnorm",
                    source: super::RMSNORM_WGSL,
                    entry_point: "main",
                    workgroup_size: (256, 1, 1),
                },
            ],
        }
    }

    /// Look up a kernel by name.
    pub fn get(&self, name: &str) -> Option<&KernelEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Return all registered kernel entries.
    pub fn all(&self) -> &[KernelEntry] {
        self.entries
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all kernel names.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.iter().map(|e| e.name)
    }
}

/// Helper to create a [`wgpu::ComputePipeline`] from a [`KernelEntry`].
///
/// The caller is responsible for providing a device. The bind-group layout is
/// automatically inferred from the shader by wgpu.
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    entry: &KernelEntry,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry.name),
        source: wgpu::ShaderSource::Wgsl(entry.source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry.name),
        layout: None, // auto-infer from shader
        module: &shader,
        entry_point: Some(entry.entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_all_kernels() {
        assert_eq!(REGISTRY.len(), 4);
        assert!(!REGISTRY.is_empty());
    }

    #[test]
    fn registry_lookup_matmul() {
        let entry = REGISTRY.get("matmul").expect("matmul not found");
        assert_eq!(entry.name, "matmul");
        assert_eq!(entry.entry_point, "main");
        assert_eq!(entry.workgroup_size, (16, 16, 1));
    }

    #[test]
    fn registry_lookup_softmax() {
        let entry = REGISTRY.get("softmax").expect("softmax not found");
        assert_eq!(entry.name, "softmax");
        assert_eq!(entry.workgroup_size, (256, 1, 1));
    }

    #[test]
    fn registry_lookup_attention() {
        let entry = REGISTRY.get("attention").expect("attention not found");
        assert_eq!(entry.name, "attention");
        assert!(!entry.source.is_empty());
    }

    #[test]
    fn registry_lookup_rmsnorm() {
        let entry = REGISTRY.get("rmsnorm").expect("rmsnorm not found");
        assert_eq!(entry.name, "rmsnorm");
        assert!(!entry.source.is_empty());
    }

    #[test]
    fn registry_lookup_missing_returns_none() {
        assert!(REGISTRY.get("nonexistent").is_none());
    }

    #[test]
    fn registry_names_complete() {
        let names: Vec<&str> = REGISTRY.names().collect();
        assert!(names.contains(&"matmul"));
        assert!(names.contains(&"softmax"));
        assert!(names.contains(&"attention"));
        assert!(names.contains(&"rmsnorm"));
    }

    #[test]
    fn all_kernels_have_entry_point_main() {
        for entry in REGISTRY.all() {
            assert_eq!(
                entry.entry_point, "main",
                "kernel {} has unexpected entry point: {}",
                entry.name, entry.entry_point
            );
        }
    }

    #[test]
    fn all_kernel_sources_contain_fn_main() {
        for entry in REGISTRY.all() {
            assert!(
                entry.source.contains("fn main"),
                "kernel {} source missing 'fn main'",
                entry.name
            );
        }
    }

    #[test]
    fn matmul_source_has_compute_attribute() {
        let entry = REGISTRY.get("matmul").unwrap();
        assert!(entry.source.contains("@compute"));
    }

    #[test]
    fn softmax_source_uses_workgroup_barrier() {
        let entry = REGISTRY.get("softmax").unwrap();
        assert!(entry.source.contains("workgroupBarrier"));
    }

    #[test]
    fn attention_source_has_scale() {
        let entry = REGISTRY.get("attention").unwrap();
        assert!(
            entry.source.contains("sqrt"),
            "attention kernel should scale by 1/sqrt(d_k)"
        );
    }

    #[test]
    fn rmsnorm_source_has_eps() {
        let entry = REGISTRY.get("rmsnorm").unwrap();
        assert!(
            entry.source.contains("eps"),
            "rmsnorm kernel should use epsilon parameter"
        );
    }
}
