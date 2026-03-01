//! Compute pipeline creation and shader module caching.

use std::collections::HashMap;

use crate::device::WgpuDevice;
use crate::error::WgpuError;

/// A compiled compute pipeline with metadata.
pub struct ComputePipeline {
    inner: wgpu::ComputePipeline,
    entry_point: String,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    /// Compile a WGSL shader and create a compute pipeline.
    ///
    /// `bind_group_entries` describes the bind group layout entries for bind
    /// group 0.  The caller is responsible for matching entries to the shader.
    pub fn new(
        device: &WgpuDevice,
        shader_src: &str,
        entry_point: &str,
        bind_group_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<Self, WgpuError> {
        let module = device.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bitnet-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout =
            device.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bitnet-bgl"),
                entries: bind_group_entries,
            });

        let pipeline_layout =
            device.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bitnet-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let inner = device.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bitnet-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self { inner, entry_point: entry_point.to_string(), bind_group_layout })
    }

    /// Access the underlying `wgpu::ComputePipeline`.
    pub fn inner(&self) -> &wgpu::ComputePipeline {
        &self.inner
    }

    /// The entry point function name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// The bind group layout for bind group 0.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipeline").field("entry_point", &self.entry_point).finish()
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache lookups.
    pub lookups: u64,
    /// Number of cache hits.
    pub hits: u64,
    /// Number of shader compilations (cache misses).
    pub compilations: u64,
}

impl CacheStats {
    /// Cache hit rate (0.0–1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 { 0.0 } else { self.hits as f64 / self.lookups as f64 }
    }
}

/// Caches compiled compute pipelines keyed by a hash of the shader source
/// and entry point.
pub struct PipelineCache {
    entries: HashMap<u64, ComputePipeline>,
    stats: CacheStats,
}

impl PipelineCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self { entries: HashMap::new(), stats: CacheStats::default() }
    }

    /// Compute a simple hash key for a (shader_src, entry_point) pair.
    fn cache_key(shader_src: &str, entry_point: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        shader_src.hash(&mut hasher);
        entry_point.hash(&mut hasher);
        hasher.finish()
    }

    /// Look up a cached pipeline or compile a new one.
    ///
    /// Returns a reference to the cached pipeline.  The `bind_group_entries`
    /// are only used on a cache miss (first compilation).
    pub fn get_or_create(
        &mut self,
        device: &WgpuDevice,
        shader_src: &str,
        entry_point: &str,
        bind_group_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<&ComputePipeline, WgpuError> {
        let key = Self::cache_key(shader_src, entry_point);
        self.stats.lookups += 1;

        if self.entries.contains_key(&key) {
            self.stats.hits += 1;
            return Ok(self.entries.get(&key).unwrap());
        }

        let pipeline = ComputePipeline::new(device, shader_src, entry_point, bind_group_entries)?;
        self.stats.compilations += 1;
        self.entries.insert(key, pipeline);
        Ok(self.entries.get(&key).unwrap())
    }

    /// Current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Number of cached pipelines.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached pipelines.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("entries", &self.entries.len())
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GPU-gated tests ──────────────────────────────────────────────

    /// Minimal WGSL shader for pipeline creation tests.
    #[cfg(test)]
    const TRIVIAL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] + 1.0;
}
"#;

    #[cfg(test)]
    fn trivial_bind_group_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }]
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn pipeline_creation_trivial_shader() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let entries = trivial_bind_group_entries();
        let pipe = ComputePipeline::new(&dev, TRIVIAL_SHADER, "main", &entries);
        assert!(pipe.is_ok());
        assert_eq!(pipe.unwrap().entry_point(), "main");
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn pipeline_debug_output() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let entries = trivial_bind_group_entries();
        let pipe = ComputePipeline::new(&dev, TRIVIAL_SHADER, "main", &entries).unwrap();
        let dbg = format!("{pipe:?}");
        assert!(dbg.contains("main"));
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn cache_miss_then_hit() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let entries = trivial_bind_group_entries();
        let mut cache = PipelineCache::new();

        // First lookup — miss.
        let _p1 = cache.get_or_create(&dev, TRIVIAL_SHADER, "main", &entries).unwrap();
        assert_eq!(cache.stats().compilations, 1);
        assert_eq!(cache.stats().hits, 0);

        // Second lookup — hit.
        let _p2 = cache.get_or_create(&dev, TRIVIAL_SHADER, "main", &entries).unwrap();
        assert_eq!(cache.stats().compilations, 1);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn cache_different_entry_points() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let entries = trivial_bind_group_entries();
        let mut cache = PipelineCache::new();

        let _p1 = cache.get_or_create(&dev, TRIVIAL_SHADER, "main", &entries).unwrap();

        // Different shader source with same entry point name compiles separately.
        let other_shader = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2.0;
}
"#;
        let _p2 = cache.get_or_create(&dev, other_shader, "main", &entries).unwrap();
        assert_eq!(cache.stats().compilations, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn cache_clear_resets() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let entries = trivial_bind_group_entries();
        let mut cache = PipelineCache::new();

        let _ = cache.get_or_create(&dev, TRIVIAL_SHADER, "main", &entries).unwrap();
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }

    // ── Non-GPU tests ────────────────────────────────────────────────

    #[test]
    fn cache_stats_initial() {
        let cache = PipelineCache::new();
        let stats = cache.stats();
        assert_eq!(stats.lookups, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.compilations, 0);
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn cache_stats_hit_rate_computed() {
        let stats = CacheStats { lookups: 10, hits: 7, compilations: 3 };
        assert!((stats.hit_rate() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn cache_empty_initial() {
        let cache = PipelineCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_debug_output() {
        let cache = PipelineCache::new();
        let dbg = format!("{cache:?}");
        assert!(dbg.contains("PipelineCache"));
    }

    #[test]
    fn cache_key_deterministic() {
        let k1 = PipelineCache::cache_key("shader_a", "main");
        let k2 = PipelineCache::cache_key("shader_a", "main");
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_varies_on_source() {
        let k1 = PipelineCache::cache_key("shader_a", "main");
        let k2 = PipelineCache::cache_key("shader_b", "main");
        assert_ne!(k1, k2);
    }
}
