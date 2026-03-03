//! Metal GPU resource lifecycle validation tests.
//!
//! Validates buffer/texture lifecycle state machines, heap management,
//! resource pooling, leak detection, and cross-resource dependencies.
//!
//! All types are pure-Rust mocks — no actual Metal framework dependency.

#![cfg(feature = "cpu")]

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ═══════════════════════════════════════════════════════════════════════
// Mock Metal resource types
// ═══════════════════════════════════════════════════════════════════════

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Buffer lifecycle state machine: Allocated → Mapped → InUse → Released.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BufferState {
    Allocated,
    Mapped,
    InUse,
    Released,
}

/// Texture pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PixelFormat {
    RGBA8Unorm,
    RGBA16Float,
    BGRA8Unorm,
    R32Float,
    Depth32Float,
    RG16Float,
}

impl PixelFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::RGBA8Unorm | PixelFormat::BGRA8Unorm => 4,
            PixelFormat::RGBA16Float => 8,
            PixelFormat::R32Float | PixelFormat::Depth32Float => 4,
            PixelFormat::RG16Float => 4,
        }
    }
}

/// Texture usage flags (bitflag-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TextureUsage(u32);

impl TextureUsage {
    const SHADER_READ: Self = Self(1);
    const SHADER_WRITE: Self = Self(2);
    const RENDER_TARGET: Self = Self(4);
    const PIXEL_VIEW: Self = Self(8);

    fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Texture lifecycle state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureState {
    Created,
    Uploading,
    Ready,
    Sampling,
    RenderTarget,
    Released,
}

/// Metal buffer with lifecycle state tracking.
#[derive(Debug, Clone)]
struct MetalBuffer {
    id: u64,
    size: usize,
    state: BufferState,
    label: String,
    map_count: u32,
    access_count: u64,
}

impl MetalBuffer {
    fn new(size: usize, label: &str) -> Self {
        Self {
            id: next_id(),
            size,
            state: BufferState::Allocated,
            label: label.to_string(),
            map_count: 0,
            access_count: 0,
        }
    }

    fn map(&mut self) -> Result<(), String> {
        match self.state {
            BufferState::Allocated | BufferState::InUse => {
                self.state = BufferState::Mapped;
                self.map_count += 1;
                Ok(())
            }
            BufferState::Mapped => Err("buffer already mapped".into()),
            BufferState::Released => Err("use-after-free: buffer released".into()),
        }
    }

    fn unmap(&mut self) -> Result<(), String> {
        match self.state {
            BufferState::Mapped => {
                self.state = BufferState::Allocated;
                Ok(())
            }
            _ => Err(format!("cannot unmap buffer in state {:?}", self.state)),
        }
    }

    fn begin_use(&mut self) -> Result<(), String> {
        match self.state {
            BufferState::Allocated => {
                self.state = BufferState::InUse;
                self.access_count += 1;
                Ok(())
            }
            BufferState::Released => Err("use-after-free: buffer released".into()),
            BufferState::Mapped => Err("cannot use mapped buffer on GPU".into()),
            _ => Err(format!("cannot begin use in state {:?}", self.state)),
        }
    }

    fn end_use(&mut self) -> Result<(), String> {
        match self.state {
            BufferState::InUse => {
                self.state = BufferState::Allocated;
                Ok(())
            }
            _ => Err(format!("cannot end use in state {:?}", self.state)),
        }
    }

    fn release(&mut self) -> Result<(), String> {
        match self.state {
            BufferState::Released => Err("double-free detected".into()),
            BufferState::InUse => Err("cannot release buffer in use".into()),
            BufferState::Mapped => Err("cannot release mapped buffer".into()),
            _ => {
                self.state = BufferState::Released;
                Ok(())
            }
        }
    }

    fn is_released(&self) -> bool {
        self.state == BufferState::Released
    }
}

/// Metal texture with format, dimensions, and mip levels.
#[derive(Debug, Clone)]
struct MetalTexture {
    id: u64,
    width: u32,
    height: u32,
    format: PixelFormat,
    usage: TextureUsage,
    mip_levels: u32,
    state: TextureState,
    label: String,
}

impl MetalTexture {
    fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        usage: TextureUsage,
        mip_levels: u32,
        label: &str,
    ) -> Self {
        Self {
            id: next_id(),
            width,
            height,
            format,
            usage,
            mip_levels: mip_levels.max(1),
            state: TextureState::Created,
            label: label.to_string(),
        }
    }

    fn max_mip_levels(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height);
        (max_dim as f64).log2().floor() as u32 + 1
    }

    fn mip_size(&self, level: u32) -> Option<(u32, u32)> {
        if level >= self.mip_levels {
            return None;
        }
        let w = (self.width >> level).max(1);
        let h = (self.height >> level).max(1);
        Some((w, h))
    }

    fn total_bytes(&self) -> usize {
        let bpp = self.format.bytes_per_pixel();
        let mut total = 0;
        for level in 0..self.mip_levels {
            let (w, h) = self.mip_size(level).unwrap();
            total += w as usize * h as usize * bpp;
        }
        total
    }

    fn upload(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::Created | TextureState::Ready => {
                self.state = TextureState::Uploading;
                Ok(())
            }
            TextureState::Released => Err("use-after-free: texture released".into()),
            _ => Err(format!("cannot upload in state {:?}", self.state)),
        }
    }

    fn finish_upload(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::Uploading => {
                self.state = TextureState::Ready;
                Ok(())
            }
            _ => Err(format!("cannot finish upload in state {:?}", self.state)),
        }
    }

    fn begin_sampling(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::Ready => {
                if !self.usage.contains(TextureUsage::SHADER_READ) {
                    return Err("texture lacks SHADER_READ usage".into());
                }
                self.state = TextureState::Sampling;
                Ok(())
            }
            TextureState::Released => Err("use-after-free: texture released".into()),
            _ => Err(format!("cannot sample in state {:?}", self.state)),
        }
    }

    fn end_sampling(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::Sampling => {
                self.state = TextureState::Ready;
                Ok(())
            }
            _ => Err(format!("cannot end sampling in state {:?}", self.state)),
        }
    }

    fn begin_render_target(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::Ready => {
                if !self.usage.contains(TextureUsage::RENDER_TARGET) {
                    return Err("texture lacks RENDER_TARGET usage".into());
                }
                self.state = TextureState::RenderTarget;
                Ok(())
            }
            _ => Err(format!("cannot use as render target in state {:?}", self.state)),
        }
    }

    fn end_render_target(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::RenderTarget => {
                self.state = TextureState::Ready;
                Ok(())
            }
            _ => Err(format!("cannot end render target in state {:?}", self.state)),
        }
    }

    fn release(&mut self) -> Result<(), String> {
        match self.state {
            TextureState::Released => Err("double-free detected".into()),
            TextureState::Sampling | TextureState::RenderTarget => {
                Err("cannot release texture in active use".into())
            }
            TextureState::Uploading => Err("cannot release texture during upload".into()),
            _ => {
                self.state = TextureState::Released;
                Ok(())
            }
        }
    }
}

/// Allocation record within a heap.
#[derive(Debug, Clone)]
struct HeapAllocation {
    offset: usize,
    size: usize,
    resource_id: u64,
    freed: bool,
}

/// Heap growth policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrowthPolicy {
    Fixed,
    Double,
    Linear(usize),
}

/// Metal memory heap with sub-allocation tracking.
#[derive(Debug)]
struct MetalHeap {
    id: u64,
    capacity: usize,
    used: usize,
    allocations: Vec<HeapAllocation>,
    growth_policy: GrowthPolicy,
    peak_usage: usize,
}

impl MetalHeap {
    fn new(capacity: usize, policy: GrowthPolicy) -> Self {
        Self {
            id: next_id(),
            capacity,
            used: 0,
            allocations: Vec::new(),
            growth_policy: policy,
            peak_usage: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> Result<u64, String> {
        let aligned = (size + 255) & !255; // 256-byte alignment
        if self.used + aligned > self.capacity {
            match self.growth_policy {
                GrowthPolicy::Fixed => {
                    return Err("heap OOM: fixed policy".into());
                }
                GrowthPolicy::Double => {
                    self.capacity = (self.capacity * 2).max(self.used + aligned);
                }
                GrowthPolicy::Linear(step) => {
                    while self.used + aligned > self.capacity {
                        self.capacity += step;
                    }
                }
            }
        }
        let offset = self.used;
        let rid = next_id();
        self.allocations.push(HeapAllocation {
            offset,
            size: aligned,
            resource_id: rid,
            freed: false,
        });
        self.used += aligned;
        self.peak_usage = self.peak_usage.max(self.used);
        Ok(rid)
    }

    fn free(&mut self, resource_id: u64) -> Result<(), String> {
        let alloc = self.allocations.iter_mut().find(|a| a.resource_id == resource_id);
        match alloc {
            Some(a) if a.freed => Err("double-free in heap".into()),
            Some(a) => {
                a.freed = true;
                Ok(())
            }
            None => Err("resource not found in heap".into()),
        }
    }

    fn fragmentation_ratio(&self) -> f64 {
        if self.allocations.is_empty() {
            return 0.0;
        }
        let freed_bytes: usize = self.allocations.iter().filter(|a| a.freed).map(|a| a.size).sum();
        if self.used == 0 {
            return 0.0;
        }
        freed_bytes as f64 / self.used as f64
    }

    fn defragment(&mut self) -> usize {
        let active: Vec<HeapAllocation> =
            self.allocations.iter().filter(|a| !a.freed).cloned().collect();
        let freed_count = self.allocations.iter().filter(|a| a.freed).count();

        let mut offset = 0;
        let mut compacted = Vec::new();
        for mut a in active {
            a.offset = offset;
            offset += a.size;
            compacted.push(a);
        }
        self.allocations = compacted;
        self.used = offset;
        freed_count
    }

    fn live_count(&self) -> usize {
        self.allocations.iter().filter(|a| !a.freed).count()
    }

    fn total_allocated(&self) -> usize {
        self.allocations.iter().filter(|a| !a.freed).map(|a| a.size).sum()
    }
}

/// Size class for pool bucketing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SizeClass(usize);

impl SizeClass {
    fn from_size(size: usize) -> Self {
        // Round up to next power of 2.
        let class = size.next_power_of_two();
        Self(class)
    }
}

/// Pool entry wrapping a recyclable buffer.
#[derive(Debug, Clone)]
struct PoolEntry {
    buffer: MetalBuffer,
    last_used_frame: u64,
}

/// Resource pool for recycling buffers and textures.
#[derive(Debug)]
struct MetalResourcePool {
    buffer_pools: BTreeMap<SizeClass, VecDeque<PoolEntry>>,
    texture_cache: HashMap<(u32, u32, u32), VecDeque<MetalTexture>>,
    max_entries_per_class: usize,
    current_frame: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl MetalResourcePool {
    fn new(max_entries: usize) -> Self {
        Self {
            buffer_pools: BTreeMap::new(),
            texture_cache: HashMap::new(),
            max_entries_per_class: max_entries,
            current_frame: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn warmup_buffers(&mut self, size: usize, count: usize) {
        let class = SizeClass::from_size(size);
        let pool = self.buffer_pools.entry(class).or_default();
        for i in 0..count {
            let buf = MetalBuffer::new(class.0, &format!("warmup_{}", i));
            pool.push_back(PoolEntry { buffer: buf, last_used_frame: self.current_frame });
        }
    }

    fn acquire_buffer(&mut self, size: usize) -> MetalBuffer {
        let class = SizeClass::from_size(size);
        if let Some(pool) = self.buffer_pools.get_mut(&class) {
            if let Some(entry) = pool.pop_front() {
                self.hits += 1;
                let mut buf = entry.buffer;
                buf.state = BufferState::Allocated;
                return buf;
            }
        }
        self.misses += 1;
        MetalBuffer::new(class.0, "pool_alloc")
    }

    fn release_buffer(&mut self, mut buffer: MetalBuffer) {
        let class = SizeClass::from_size(buffer.size);
        buffer.state = BufferState::Allocated;
        buffer.map_count = 0;
        let pool = self.buffer_pools.entry(class).or_default();

        if pool.len() >= self.max_entries_per_class {
            pool.pop_front(); // evict oldest
            self.evictions += 1;
        }

        pool.push_back(PoolEntry { buffer, last_used_frame: self.current_frame });
    }

    fn acquire_texture(
        &mut self,
        w: u32,
        h: u32,
        format: PixelFormat,
        usage: TextureUsage,
        mips: u32,
    ) -> MetalTexture {
        let key = (w, h, format as u32);
        if let Some(pool) = self.texture_cache.get_mut(&key) {
            if let Some(mut tex) = pool.pop_front() {
                self.hits += 1;
                tex.state = TextureState::Created;
                return tex;
            }
        }
        self.misses += 1;
        MetalTexture::new(w, h, format, usage, mips, "pool_tex")
    }

    fn release_texture(&mut self, mut texture: MetalTexture) {
        let key = (texture.width, texture.height, texture.format as u32);
        texture.state = TextureState::Created;
        let pool = self.texture_cache.entry(key).or_default();
        if pool.len() >= self.max_entries_per_class {
            pool.pop_front();
            self.evictions += 1;
        }
        pool.push_back(texture);
    }

    fn evict_lru(&mut self, max_age: u64) -> usize {
        let mut evicted = 0;
        let frame = self.current_frame;
        for pool in self.buffer_pools.values_mut() {
            let before = pool.len();
            pool.retain(|e| frame - e.last_used_frame < max_age);
            evicted += before - pool.len();
        }
        self.evictions += evicted as u64;
        evicted
    }

    fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    fn total_cached(&self) -> usize {
        let bufs: usize = self.buffer_pools.values().map(|p| p.len()).sum();
        let texs: usize = self.texture_cache.values().map(|p| p.len()).sum();
        bufs + texs
    }

    fn size_classes(&self) -> Vec<SizeClass> {
        self.buffer_pools.keys().copied().collect()
    }
}

/// Resource kind for tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceKind {
    Buffer,
    Texture,
}

/// Tracked resource entry.
#[derive(Debug, Clone)]
struct TrackedResource {
    id: u64,
    kind: ResourceKind,
    label: String,
    scope_depth: usize,
    released: bool,
    reported: bool,
}

/// Tracks all resources for leak detection.
#[derive(Debug)]
struct MetalResourceTracker {
    resources: HashMap<u64, TrackedResource>,
    scope_stack: Vec<String>,
    leak_reports: Vec<String>,
}

impl MetalResourceTracker {
    fn new() -> Self {
        Self { resources: HashMap::new(), scope_stack: Vec::new(), leak_reports: Vec::new() }
    }

    fn track_buffer(&mut self, buffer: &MetalBuffer) {
        self.resources.insert(
            buffer.id,
            TrackedResource {
                id: buffer.id,
                kind: ResourceKind::Buffer,
                label: buffer.label.clone(),
                scope_depth: self.scope_stack.len(),
                released: false,
                reported: false,
            },
        );
    }

    fn track_texture(&mut self, texture: &MetalTexture) {
        self.resources.insert(
            texture.id,
            TrackedResource {
                id: texture.id,
                kind: ResourceKind::Texture,
                label: texture.label.clone(),
                scope_depth: self.scope_stack.len(),
                released: false,
                reported: false,
            },
        );
    }

    fn mark_released(&mut self, id: u64) -> Result<(), String> {
        match self.resources.get_mut(&id) {
            Some(r) if r.released => Err(format!("double-free: resource {} already released", id)),
            Some(r) => {
                r.released = true;
                Ok(())
            }
            None => Err(format!("untracked resource {}", id)),
        }
    }

    fn push_scope(&mut self, name: &str) {
        self.scope_stack.push(name.to_string());
    }

    fn pop_scope(&mut self) -> Vec<String> {
        let depth = self.scope_stack.len();
        self.scope_stack.pop();

        let mut leaks = Vec::new();
        let mut leak_ids = Vec::new();
        for r in self.resources.values() {
            if r.scope_depth == depth && !r.released && !r.reported {
                let msg = format!(
                    "LEAK: {:?} id={} label='{}' in scope depth {}",
                    r.kind, r.id, r.label, depth
                );
                leaks.push(msg.clone());
                self.leak_reports.push(msg);
                leak_ids.push(r.id);
            }
        }
        for id in leak_ids {
            if let Some(r) = self.resources.get_mut(&id) {
                r.reported = true;
            }
        }
        leaks
    }

    fn active_count(&self) -> usize {
        self.resources.values().filter(|r| !r.released).count()
    }

    fn leaked_resources(&self) -> Vec<&TrackedResource> {
        self.resources.values().filter(|r| !r.released).collect()
    }

    fn total_tracked(&self) -> usize {
        self.resources.len()
    }

    fn scope_depth(&self) -> usize {
        self.scope_stack.len()
    }

    fn cleanup_scope(&mut self) {
        let depth = self.scope_stack.len();
        let ids: Vec<u64> = self
            .resources
            .iter()
            .filter(|(_, r)| r.scope_depth == depth && !r.released)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(r) = self.resources.get_mut(&id) {
                r.released = true;
            }
        }
    }
}

/// Dependency edge between resources.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResourceDependency {
    source_id: u64,
    target_id: u64,
    kind: DependencyKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DependencyKind {
    Copy,
    Alias,
    ReadAfterWrite,
    WriteAfterRead,
}

/// Barrier type for synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BarrierType {
    BufferBarrier,
    TextureBarrier,
    MemoryBarrier,
}

/// Tracks cross-resource dependencies and barriers.
#[derive(Debug)]
struct DependencyGraph {
    edges: Vec<ResourceDependency>,
    barriers: Vec<(BarrierType, u64, u64)>,
}

impl DependencyGraph {
    fn new() -> Self {
        Self { edges: Vec::new(), barriers: Vec::new() }
    }

    fn add_dependency(&mut self, source: u64, target: u64, kind: DependencyKind) {
        self.edges.push(ResourceDependency { source_id: source, target_id: target, kind });
    }

    fn insert_barrier(&mut self, btype: BarrierType, after: u64, before: u64) {
        self.barriers.push((btype, after, before));
    }

    fn dependencies_of(&self, id: u64) -> Vec<&ResourceDependency> {
        self.edges.iter().filter(|e| e.target_id == id).collect()
    }

    fn dependents_of(&self, id: u64) -> Vec<&ResourceDependency> {
        self.edges.iter().filter(|e| e.source_id == id).collect()
    }

    fn has_cycle(&self) -> bool {
        let mut ids: HashSet<u64> = HashSet::new();
        for e in &self.edges {
            ids.insert(e.source_id);
            ids.insert(e.target_id);
        }
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for &id in &ids {
            if self.dfs_cycle(id, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        &self,
        node: u64,
        visited: &mut HashSet<u64>,
        in_stack: &mut HashSet<u64>,
    ) -> bool {
        if in_stack.contains(&node) {
            return true;
        }
        if visited.contains(&node) {
            return false;
        }
        visited.insert(node);
        in_stack.insert(node);
        for e in &self.edges {
            if e.source_id == node && self.dfs_cycle(e.target_id, visited, in_stack) {
                return true;
            }
        }
        in_stack.remove(&node);
        false
    }

    fn topological_order(&self) -> Option<Vec<u64>> {
        if self.has_cycle() {
            return None;
        }
        let mut ids: HashSet<u64> = HashSet::new();
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for e in &self.edges {
            ids.insert(e.source_id);
            ids.insert(e.target_id);
            *in_degree.entry(e.target_id).or_default() += 1;
            in_degree.entry(e.source_id).or_default();
        }
        let mut queue: VecDeque<u64> =
            in_degree.iter().filter(|&(_, &d)| d == 0).map(|(id, _)| *id).collect();
        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for e in &self.edges {
                if e.source_id == node {
                    if let Some(d) = in_degree.get_mut(&e.target_id) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(e.target_id);
                        }
                    }
                }
            }
        }
        if order.len() == ids.len() { Some(order) } else { None }
    }

    fn barrier_count(&self) -> usize {
        self.barriers.len()
    }

    fn required_barriers(&self) -> Vec<(BarrierType, u64, u64)> {
        let mut required = Vec::new();
        for e in &self.edges {
            let btype = match e.kind {
                DependencyKind::ReadAfterWrite | DependencyKind::WriteAfterRead => {
                    BarrierType::MemoryBarrier
                }
                DependencyKind::Copy => BarrierType::BufferBarrier,
                DependencyKind::Alias => BarrierType::TextureBarrier,
            };
            required.push((btype, e.source_id, e.target_id));
        }
        required
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test modules
// ═══════════════════════════════════════════════════════════════════════

mod buffer_lifecycle {
    use super::*;

    #[test]
    fn create_buffer_initial_state() {
        let buf = MetalBuffer::new(1024, "test");
        assert_eq!(buf.state, BufferState::Allocated);
        assert_eq!(buf.size, 1024);
        assert_eq!(buf.map_count, 0);
    }

    #[test]
    fn map_allocated_buffer() {
        let mut buf = MetalBuffer::new(256, "map_test");
        assert!(buf.map().is_ok());
        assert_eq!(buf.state, BufferState::Mapped);
        assert_eq!(buf.map_count, 1);
    }

    #[test]
    fn unmap_mapped_buffer() {
        let mut buf = MetalBuffer::new(256, "unmap_test");
        buf.map().unwrap();
        assert!(buf.unmap().is_ok());
        assert_eq!(buf.state, BufferState::Allocated);
    }

    #[test]
    fn full_map_unmap_cycle() {
        let mut buf = MetalBuffer::new(512, "cycle");
        buf.map().unwrap();
        buf.unmap().unwrap();
        assert_eq!(buf.state, BufferState::Allocated);
        assert_eq!(buf.map_count, 1);
    }

    #[test]
    fn begin_end_use_cycle() {
        let mut buf = MetalBuffer::new(1024, "use_cycle");
        buf.begin_use().unwrap();
        assert_eq!(buf.state, BufferState::InUse);
        buf.end_use().unwrap();
        assert_eq!(buf.state, BufferState::Allocated);
    }

    #[test]
    fn release_allocated_buffer() {
        let mut buf = MetalBuffer::new(128, "release");
        assert!(buf.release().is_ok());
        assert!(buf.is_released());
    }

    #[test]
    fn double_free_detected() {
        let mut buf = MetalBuffer::new(64, "double_free");
        buf.release().unwrap();
        let err = buf.release().unwrap_err();
        assert!(err.contains("double-free"));
    }

    #[test]
    fn use_after_free_on_map() {
        let mut buf = MetalBuffer::new(64, "uaf_map");
        buf.release().unwrap();
        let err = buf.map().unwrap_err();
        assert!(err.contains("use-after-free"));
    }

    #[test]
    fn use_after_free_on_begin_use() {
        let mut buf = MetalBuffer::new(64, "uaf_use");
        buf.release().unwrap();
        let err = buf.begin_use().unwrap_err();
        assert!(err.contains("use-after-free"));
    }

    #[test]
    fn cannot_use_mapped_buffer() {
        let mut buf = MetalBuffer::new(256, "mapped_use");
        buf.map().unwrap();
        let err = buf.begin_use().unwrap_err();
        assert!(err.contains("cannot use mapped buffer"));
    }

    #[test]
    fn cannot_release_in_use_buffer() {
        let mut buf = MetalBuffer::new(256, "in_use_release");
        buf.begin_use().unwrap();
        let err = buf.release().unwrap_err();
        assert!(err.contains("cannot release buffer in use"));
    }

    #[test]
    fn cannot_release_mapped_buffer() {
        let mut buf = MetalBuffer::new(256, "mapped_release");
        buf.map().unwrap();
        let err = buf.release().unwrap_err();
        assert!(err.contains("cannot release mapped buffer"));
    }

    #[test]
    fn multiple_map_unmap_cycles() {
        let mut buf = MetalBuffer::new(256, "multi_cycle");
        for _ in 0..5 {
            buf.map().unwrap();
            buf.unmap().unwrap();
        }
        assert_eq!(buf.map_count, 5);
        assert_eq!(buf.state, BufferState::Allocated);
    }

    #[test]
    fn double_map_fails() {
        let mut buf = MetalBuffer::new(256, "double_map");
        buf.map().unwrap();
        let err = buf.map().unwrap_err();
        assert!(err.contains("already mapped"));
    }

    #[test]
    fn map_after_use_cycle() {
        let mut buf = MetalBuffer::new(512, "use_then_map");
        buf.begin_use().unwrap();
        buf.end_use().unwrap();
        assert!(buf.map().is_ok());
        assert_eq!(buf.state, BufferState::Mapped);
    }

    #[test]
    fn access_count_tracks_uses() {
        let mut buf = MetalBuffer::new(1024, "access_count");
        for _ in 0..3 {
            buf.begin_use().unwrap();
            buf.end_use().unwrap();
        }
        assert_eq!(buf.access_count, 3);
    }

    #[test]
    fn buffer_labels_preserved() {
        let buf = MetalBuffer::new(64, "my_label");
        assert_eq!(buf.label, "my_label");
    }

    #[test]
    fn buffer_ids_unique() {
        let b1 = MetalBuffer::new(64, "a");
        let b2 = MetalBuffer::new(64, "b");
        assert_ne!(b1.id, b2.id);
    }

    #[test]
    fn concurrent_buffer_access_pattern() {
        let buf = Arc::new(Mutex::new(MetalBuffer::new(4096, "shared")));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let b = Arc::clone(&buf);
                std::thread::spawn(move || {
                    let mut guard = b.lock().unwrap();
                    if guard.state == BufferState::Allocated {
                        let _ = guard.begin_use();
                        let _ = guard.end_use();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let guard = buf.lock().unwrap();
        assert_eq!(guard.state, BufferState::Allocated);
    }
}

mod texture_lifecycle {
    use super::*;

    #[test]
    fn create_texture_initial_state() {
        let tex = MetalTexture::new(
            512,
            512,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "test_tex",
        );
        assert_eq!(tex.state, TextureState::Created);
        assert_eq!(tex.width, 512);
        assert_eq!(tex.height, 512);
    }

    #[test]
    fn upload_and_finish() {
        let mut tex = MetalTexture::new(
            256,
            256,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "upload",
        );
        tex.upload().unwrap();
        assert_eq!(tex.state, TextureState::Uploading);
        tex.finish_upload().unwrap();
        assert_eq!(tex.state, TextureState::Ready);
    }

    #[test]
    fn sample_ready_texture() {
        let mut tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "sample",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        assert!(tex.begin_sampling().is_ok());
        assert_eq!(tex.state, TextureState::Sampling);
    }

    #[test]
    fn end_sampling_returns_to_ready() {
        let mut tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "end_sample",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        tex.begin_sampling().unwrap();
        tex.end_sampling().unwrap();
        assert_eq!(tex.state, TextureState::Ready);
    }

    #[test]
    fn release_ready_texture() {
        let mut tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "release",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        assert!(tex.release().is_ok());
        assert_eq!(tex.state, TextureState::Released);
    }

    #[test]
    fn double_free_texture_detected() {
        let mut tex =
            MetalTexture::new(32, 32, PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ, 1, "df");
        tex.release().unwrap();
        let err = tex.release().unwrap_err();
        assert!(err.contains("double-free"));
    }

    #[test]
    fn cannot_release_during_sampling() {
        let mut tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "busy",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        tex.begin_sampling().unwrap();
        let err = tex.release().unwrap_err();
        assert!(err.contains("cannot release texture in active use"));
    }

    #[test]
    fn mip_chain_sizes() {
        let tex = MetalTexture::new(
            256,
            256,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            4,
            "mip",
        );
        assert_eq!(tex.mip_size(0), Some((256, 256)));
        assert_eq!(tex.mip_size(1), Some((128, 128)));
        assert_eq!(tex.mip_size(2), Some((64, 64)));
        assert_eq!(tex.mip_size(3), Some((32, 32)));
        assert_eq!(tex.mip_size(4), None);
    }

    #[test]
    fn max_mip_levels_calculation() {
        assert_eq!(MetalTexture::max_mip_levels(256, 256), 9);
        assert_eq!(MetalTexture::max_mip_levels(1, 1), 1);
        assert_eq!(MetalTexture::max_mip_levels(1024, 512), 11);
    }

    #[test]
    fn total_bytes_with_mips() {
        let tex =
            MetalTexture::new(4, 4, PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ, 3, "bytes");
        // level 0: 4*4*4=64, level 1: 2*2*4=16, level 2: 1*1*4=4
        assert_eq!(tex.total_bytes(), 84);
    }

    #[test]
    fn format_conversion_lifecycle() {
        let mut tex_src = MetalTexture::new(
            128,
            128,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "src",
        );
        let mut tex_dst = MetalTexture::new(
            128,
            128,
            PixelFormat::RGBA16Float,
            TextureUsage::SHADER_WRITE,
            1,
            "dst",
        );
        tex_src.upload().unwrap();
        tex_src.finish_upload().unwrap();
        tex_dst.upload().unwrap();
        tex_dst.finish_upload().unwrap();
        assert_eq!(tex_src.format, PixelFormat::RGBA8Unorm);
        assert_eq!(tex_dst.format, PixelFormat::RGBA16Float);
        tex_src.release().unwrap();
        tex_dst.release().unwrap();
    }

    #[test]
    fn render_target_usage() {
        let mut tex = MetalTexture::new(
            1920,
            1080,
            PixelFormat::BGRA8Unorm,
            TextureUsage::RENDER_TARGET,
            1,
            "rt",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        tex.begin_render_target().unwrap();
        assert_eq!(tex.state, TextureState::RenderTarget);
        tex.end_render_target().unwrap();
        assert_eq!(tex.state, TextureState::Ready);
    }

    #[test]
    fn cannot_sample_without_shader_read() {
        let mut tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_WRITE,
            1,
            "no_read",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        let err = tex.begin_sampling().unwrap_err();
        assert!(err.contains("SHADER_READ"));
    }

    #[test]
    fn cannot_render_without_render_target_flag() {
        let mut tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "no_rt",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        let err = tex.begin_render_target().unwrap_err();
        assert!(err.contains("RENDER_TARGET"));
    }

    #[test]
    fn render_target_reuse_cycle() {
        let mut tex = MetalTexture::new(
            800,
            600,
            PixelFormat::BGRA8Unorm,
            TextureUsage::RENDER_TARGET,
            1,
            "reuse_rt",
        );
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        for _ in 0..3 {
            tex.begin_render_target().unwrap();
            tex.end_render_target().unwrap();
        }
        assert_eq!(tex.state, TextureState::Ready);
    }

    #[test]
    fn depth_texture_lifecycle() {
        let mut tex = MetalTexture::new(
            1024,
            1024,
            PixelFormat::Depth32Float,
            TextureUsage::RENDER_TARGET,
            1,
            "depth",
        );
        assert_eq!(tex.format.bytes_per_pixel(), 4);
        tex.upload().unwrap();
        tex.finish_upload().unwrap();
        tex.begin_render_target().unwrap();
        tex.end_render_target().unwrap();
        tex.release().unwrap();
    }

    #[test]
    fn texture_ids_unique() {
        let t1 = MetalTexture::new(8, 8, PixelFormat::R32Float, TextureUsage::SHADER_READ, 1, "a");
        let t2 = MetalTexture::new(8, 8, PixelFormat::R32Float, TextureUsage::SHADER_READ, 1, "b");
        assert_ne!(t1.id, t2.id);
    }

    #[test]
    fn use_after_free_texture_upload() {
        let mut tex =
            MetalTexture::new(32, 32, PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ, 1, "uaf");
        tex.release().unwrap();
        let err = tex.upload().unwrap_err();
        assert!(err.contains("use-after-free"));
    }
}

mod heap_management {
    use super::*;

    #[test]
    fn create_heap_initial_state() {
        let heap = MetalHeap::new(1024 * 1024, GrowthPolicy::Fixed);
        assert_eq!(heap.capacity, 1024 * 1024);
        assert_eq!(heap.used, 0);
        assert_eq!(heap.live_count(), 0);
    }

    #[test]
    fn allocate_from_heap() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let rid = heap.allocate(128).unwrap();
        assert!(rid > 0);
        assert_eq!(heap.live_count(), 1);
        assert!(heap.used > 0);
    }

    #[test]
    fn allocate_aligned_to_256() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        heap.allocate(100).unwrap();
        // 100 rounds up to 256
        assert_eq!(heap.used, 256);
    }

    #[test]
    fn multiple_allocations() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        for _ in 0..4 {
            heap.allocate(256).unwrap();
        }
        assert_eq!(heap.live_count(), 4);
        assert_eq!(heap.used, 1024);
    }

    #[test]
    fn oom_with_fixed_policy() {
        let mut heap = MetalHeap::new(512, GrowthPolicy::Fixed);
        heap.allocate(256).unwrap();
        let err = heap.allocate(512).unwrap_err();
        assert!(err.contains("OOM"));
    }

    #[test]
    fn double_growth_policy() {
        let mut heap = MetalHeap::new(512, GrowthPolicy::Double);
        heap.allocate(256).unwrap();
        heap.allocate(512).unwrap(); // triggers growth
        assert!(heap.capacity >= 1024);
    }

    #[test]
    fn linear_growth_policy() {
        let mut heap = MetalHeap::new(256, GrowthPolicy::Linear(512));
        heap.allocate(256).unwrap();
        heap.allocate(256).unwrap(); // triggers linear growth
        assert!(heap.capacity >= 768);
    }

    #[test]
    fn free_allocation() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let rid = heap.allocate(256).unwrap();
        assert!(heap.free(rid).is_ok());
    }

    #[test]
    fn double_free_in_heap() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let rid = heap.allocate(256).unwrap();
        heap.free(rid).unwrap();
        let err = heap.free(rid).unwrap_err();
        assert!(err.contains("double-free"));
    }

    #[test]
    fn free_unknown_resource() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let err = heap.free(99999).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn fragmentation_ratio_zero_initially() {
        let heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        assert!((heap.fragmentation_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fragmentation_increases_after_frees() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let r1 = heap.allocate(256).unwrap();
        let _r2 = heap.allocate(256).unwrap();
        heap.free(r1).unwrap();
        assert!(heap.fragmentation_ratio() > 0.0);
    }

    #[test]
    fn defragment_compacts_heap() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let r1 = heap.allocate(256).unwrap();
        let _r2 = heap.allocate(256).unwrap();
        let r3 = heap.allocate(256).unwrap();
        heap.free(r1).unwrap();
        heap.free(r3).unwrap();
        let freed = heap.defragment();
        assert_eq!(freed, 2);
        assert_eq!(heap.live_count(), 1);
        assert_eq!(heap.used, 256);
    }

    #[test]
    fn peak_usage_tracking() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        heap.allocate(256).unwrap();
        heap.allocate(256).unwrap();
        let peak = heap.peak_usage;
        assert_eq!(peak, 512);
    }

    #[test]
    fn total_allocated_excludes_freed() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let r1 = heap.allocate(256).unwrap();
        heap.allocate(256).unwrap();
        heap.free(r1).unwrap();
        assert_eq!(heap.total_allocated(), 256);
    }

    #[test]
    fn many_small_allocations() {
        let mut heap = MetalHeap::new(256 * 100, GrowthPolicy::Fixed);
        for _ in 0..100 {
            heap.allocate(1).unwrap(); // each rounds to 256
        }
        assert_eq!(heap.live_count(), 100);
    }

    #[test]
    fn growth_preserves_existing_allocations() {
        let mut heap = MetalHeap::new(256, GrowthPolicy::Double);
        let r1 = heap.allocate(128).unwrap();
        let r2 = heap.allocate(256).unwrap();
        assert_eq!(heap.live_count(), 2);
        // Both allocations should still be valid
        assert!(heap.free(r1).is_ok());
        assert!(heap.free(r2).is_ok());
    }

    #[test]
    fn defragment_empty_heap_is_noop() {
        let mut heap = MetalHeap::new(4096, GrowthPolicy::Fixed);
        let freed = heap.defragment();
        assert_eq!(freed, 0);
        assert_eq!(heap.used, 0);
    }
}

mod resource_pooling {
    use super::*;

    #[test]
    fn create_empty_pool() {
        let pool = MetalResourcePool::new(16);
        assert_eq!(pool.total_cached(), 0);
        assert_eq!(pool.hits, 0);
        assert_eq!(pool.misses, 0);
    }

    #[test]
    fn warmup_populates_pool() {
        let mut pool = MetalResourcePool::new(16);
        pool.warmup_buffers(1024, 4);
        assert_eq!(pool.total_cached(), 4);
    }

    #[test]
    fn acquire_from_warmed_pool() {
        let mut pool = MetalResourcePool::new(16);
        pool.warmup_buffers(1024, 2);
        let buf = pool.acquire_buffer(1024);
        assert_eq!(buf.state, BufferState::Allocated);
        assert_eq!(pool.hits, 1);
    }

    #[test]
    fn acquire_miss_creates_new() {
        let mut pool = MetalResourcePool::new(16);
        let buf = pool.acquire_buffer(512);
        assert_eq!(buf.state, BufferState::Allocated);
        assert_eq!(pool.misses, 1);
    }

    #[test]
    fn release_returns_to_pool() {
        let mut pool = MetalResourcePool::new(16);
        let buf = pool.acquire_buffer(256);
        pool.release_buffer(buf);
        assert_eq!(pool.total_cached(), 1);
    }

    #[test]
    fn acquire_release_cycle() {
        let mut pool = MetalResourcePool::new(16);
        let buf = pool.acquire_buffer(512);
        pool.release_buffer(buf);
        let buf2 = pool.acquire_buffer(512);
        assert_eq!(buf2.state, BufferState::Allocated);
        assert_eq!(pool.hits, 1);
    }

    #[test]
    fn lru_eviction_by_age() {
        let mut pool = MetalResourcePool::new(100);
        pool.warmup_buffers(256, 5);
        for _ in 0..10 {
            pool.advance_frame();
        }
        let evicted = pool.evict_lru(5);
        assert_eq!(evicted, 5);
        assert_eq!(pool.total_cached(), 0);
    }

    #[test]
    fn recent_entries_survive_eviction() {
        let mut pool = MetalResourcePool::new(100);
        pool.warmup_buffers(256, 3);
        for _ in 0..10 {
            pool.advance_frame();
        }
        // Add fresh entry at current frame
        let buf = pool.acquire_buffer(512); // miss, different class
        pool.release_buffer(buf);
        let evicted = pool.evict_lru(5);
        // Old 256-class entries evicted, new 512-class one survives
        assert_eq!(evicted, 3);
        assert_eq!(pool.total_cached(), 1);
    }

    #[test]
    fn pool_statistics_tracking() {
        let mut pool = MetalResourcePool::new(16);
        pool.warmup_buffers(256, 2);
        pool.acquire_buffer(256); // hit
        pool.acquire_buffer(256); // hit
        pool.acquire_buffer(256); // miss
        assert_eq!(pool.hits, 2);
        assert_eq!(pool.misses, 1);
        assert!((pool.hit_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn size_class_bucketing() {
        let mut pool = MetalResourcePool::new(16);
        pool.warmup_buffers(100, 1); // rounds to 128
        pool.warmup_buffers(200, 1); // rounds to 256
        pool.warmup_buffers(500, 1); // rounds to 512
        let classes = pool.size_classes();
        assert_eq!(classes.len(), 3);
        assert!(classes.contains(&SizeClass(128)));
        assert!(classes.contains(&SizeClass(256)));
        assert!(classes.contains(&SizeClass(512)));
    }

    #[test]
    fn max_entries_per_class_enforced() {
        let mut pool = MetalResourcePool::new(2);
        pool.warmup_buffers(256, 2); // fill to max
        let buf = pool.acquire_buffer(512); // miss (different class)
        pool.release_buffer(buf);
        let extra = MetalBuffer::new(256, "extra");
        pool.release_buffer(extra); // evicts oldest in 256 class
        let class = SizeClass::from_size(256);
        assert!(pool.buffer_pools[&class].len() <= 2);
        assert!(pool.evictions > 0);
    }

    #[test]
    fn texture_pool_acquire_release() {
        let mut pool = MetalResourcePool::new(16);
        let tex =
            pool.acquire_texture(256, 256, PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ, 1);
        assert_eq!(pool.misses, 1);
        pool.release_texture(tex);
        let tex2 =
            pool.acquire_texture(256, 256, PixelFormat::RGBA8Unorm, TextureUsage::SHADER_READ, 1);
        assert_eq!(pool.hits, 1);
        assert_eq!(tex2.state, TextureState::Created);
    }

    #[test]
    fn different_sizes_use_different_buckets() {
        let mut pool = MetalResourcePool::new(16);
        let b1 = pool.acquire_buffer(100);
        let b2 = pool.acquire_buffer(300);
        pool.release_buffer(b1);
        pool.release_buffer(b2);
        assert_eq!(pool.size_classes().len(), 2);
    }

    #[test]
    fn hit_rate_zero_when_empty() {
        let pool = MetalResourcePool::new(16);
        assert!((pool.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn frame_advancement() {
        let mut pool = MetalResourcePool::new(16);
        assert_eq!(pool.current_frame, 0);
        pool.advance_frame();
        pool.advance_frame();
        assert_eq!(pool.current_frame, 2);
    }

    #[test]
    fn pool_handles_zero_size() {
        let mut pool = MetalResourcePool::new(16);
        let buf = pool.acquire_buffer(0);
        // next_power_of_two(0) = 1
        assert!(buf.size >= 1);
    }

    #[test]
    fn pool_eviction_increments_counter() {
        let mut pool = MetalResourcePool::new(1);
        // Fill pool with one entry via warmup
        pool.warmup_buffers(256, 1);
        // Release another buffer of same class — triggers eviction
        let buf = MetalBuffer::new(256, "extra");
        pool.release_buffer(buf);
        assert!(pool.evictions >= 1);
    }

    #[test]
    fn multiple_size_classes_independent() {
        let mut pool = MetalResourcePool::new(4);
        pool.warmup_buffers(128, 2);
        pool.warmup_buffers(1024, 3);
        assert_eq!(pool.size_classes().len(), 2);
        assert_eq!(pool.total_cached(), 5);
        // Acquiring from one class doesn't affect the other
        pool.acquire_buffer(128);
        assert_eq!(pool.total_cached(), 4);
    }
}

mod leak_detection {
    use super::*;

    #[test]
    fn tracker_initially_empty() {
        let tracker = MetalResourceTracker::new();
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.total_tracked(), 0);
    }

    #[test]
    fn track_buffer() {
        let mut tracker = MetalResourceTracker::new();
        let buf = MetalBuffer::new(256, "tracked");
        tracker.track_buffer(&buf);
        assert_eq!(tracker.active_count(), 1);
    }

    #[test]
    fn track_texture() {
        let mut tracker = MetalResourceTracker::new();
        let tex = MetalTexture::new(
            64,
            64,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "tracked_tex",
        );
        tracker.track_texture(&tex);
        assert_eq!(tracker.active_count(), 1);
    }

    #[test]
    fn mark_released_decreases_active() {
        let mut tracker = MetalResourceTracker::new();
        let buf = MetalBuffer::new(256, "release_me");
        tracker.track_buffer(&buf);
        tracker.mark_released(buf.id).unwrap();
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn double_release_detected() {
        let mut tracker = MetalResourceTracker::new();
        let buf = MetalBuffer::new(256, "double");
        tracker.track_buffer(&buf);
        tracker.mark_released(buf.id).unwrap();
        let err = tracker.mark_released(buf.id).unwrap_err();
        assert!(err.contains("double-free"));
    }

    #[test]
    fn untracked_resource_release_fails() {
        let mut tracker = MetalResourceTracker::new();
        let err = tracker.mark_released(99999).unwrap_err();
        assert!(err.contains("untracked"));
    }

    #[test]
    fn leak_report_on_scope_exit() {
        let mut tracker = MetalResourceTracker::new();
        tracker.push_scope("render_pass");
        let buf = MetalBuffer::new(512, "leaked_buf");
        tracker.track_buffer(&buf);
        let leaks = tracker.pop_scope();
        assert_eq!(leaks.len(), 1);
        assert!(leaks[0].contains("LEAK"));
        assert!(leaks[0].contains("leaked_buf"));
    }

    #[test]
    fn no_leak_when_released_before_scope_exit() {
        let mut tracker = MetalResourceTracker::new();
        tracker.push_scope("clean_scope");
        let buf = MetalBuffer::new(128, "clean");
        tracker.track_buffer(&buf);
        tracker.mark_released(buf.id).unwrap();
        let leaks = tracker.pop_scope();
        assert!(leaks.is_empty());
    }

    #[test]
    fn nested_scopes() {
        let mut tracker = MetalResourceTracker::new();
        tracker.push_scope("outer");
        let b1 = MetalBuffer::new(64, "outer_buf");
        tracker.track_buffer(&b1);

        tracker.push_scope("inner");
        let b2 = MetalBuffer::new(64, "inner_buf");
        tracker.track_buffer(&b2);
        tracker.mark_released(b2.id).unwrap();
        let inner_leaks = tracker.pop_scope();
        assert!(inner_leaks.is_empty());

        tracker.mark_released(b1.id).unwrap();
        let outer_leaks = tracker.pop_scope();
        assert!(outer_leaks.is_empty());
    }

    #[test]
    fn nested_scope_inner_leak() {
        let mut tracker = MetalResourceTracker::new();
        tracker.push_scope("outer");
        tracker.push_scope("inner");
        let buf = MetalBuffer::new(64, "inner_leak");
        tracker.track_buffer(&buf);
        let leaks = tracker.pop_scope();
        assert_eq!(leaks.len(), 1);
    }

    #[test]
    fn scope_depth_tracking() {
        let mut tracker = MetalResourceTracker::new();
        assert_eq!(tracker.scope_depth(), 0);
        tracker.push_scope("a");
        assert_eq!(tracker.scope_depth(), 1);
        tracker.push_scope("b");
        assert_eq!(tracker.scope_depth(), 2);
        tracker.pop_scope();
        assert_eq!(tracker.scope_depth(), 1);
    }

    #[test]
    fn leaked_resources_list() {
        let mut tracker = MetalResourceTracker::new();
        let b1 = MetalBuffer::new(64, "leak1");
        let b2 = MetalBuffer::new(64, "leak2");
        let b3 = MetalBuffer::new(64, "released");
        tracker.track_buffer(&b1);
        tracker.track_buffer(&b2);
        tracker.track_buffer(&b3);
        tracker.mark_released(b3.id).unwrap();
        let leaked = tracker.leaked_resources();
        assert_eq!(leaked.len(), 2);
    }

    #[test]
    fn total_tracked_includes_released() {
        let mut tracker = MetalResourceTracker::new();
        let buf = MetalBuffer::new(64, "t");
        tracker.track_buffer(&buf);
        tracker.mark_released(buf.id).unwrap();
        assert_eq!(tracker.total_tracked(), 1);
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn cleanup_scope_releases_all() {
        let mut tracker = MetalResourceTracker::new();
        tracker.push_scope("auto_cleanup");
        for i in 0..5 {
            let buf = MetalBuffer::new(64, &format!("buf_{}", i));
            tracker.track_buffer(&buf);
        }
        assert_eq!(tracker.active_count(), 5);
        tracker.cleanup_scope();
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn leak_reports_accumulate() {
        let mut tracker = MetalResourceTracker::new();
        tracker.push_scope("s1");
        let b1 = MetalBuffer::new(32, "l1");
        tracker.track_buffer(&b1);
        tracker.pop_scope();

        tracker.push_scope("s2");
        let b2 = MetalBuffer::new(32, "l2");
        tracker.track_buffer(&b2);
        tracker.pop_scope();

        assert_eq!(tracker.leak_reports.len(), 2);
    }

    #[test]
    fn mixed_buffer_texture_tracking() {
        let mut tracker = MetalResourceTracker::new();
        let buf = MetalBuffer::new(256, "b");
        let tex =
            MetalTexture::new(64, 64, PixelFormat::R32Float, TextureUsage::SHADER_READ, 1, "t");
        tracker.track_buffer(&buf);
        tracker.track_texture(&tex);
        assert_eq!(tracker.active_count(), 2);
        tracker.mark_released(buf.id).unwrap();
        tracker.mark_released(tex.id).unwrap();
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn resource_kind_preserved() {
        let mut tracker = MetalResourceTracker::new();
        let buf = MetalBuffer::new(64, "buf");
        let tex =
            MetalTexture::new(8, 8, PixelFormat::R32Float, TextureUsage::SHADER_READ, 1, "tex");
        tracker.track_buffer(&buf);
        tracker.track_texture(&tex);
        assert_eq!(tracker.resources[&buf.id].kind, ResourceKind::Buffer);
        assert_eq!(tracker.resources[&tex.id].kind, ResourceKind::Texture);
    }
}

mod cross_resource_dependencies {
    use super::*;

    #[test]
    fn empty_dependency_graph() {
        let graph = DependencyGraph::new();
        assert_eq!(graph.barrier_count(), 0);
        assert!(!graph.has_cycle());
    }

    #[test]
    fn add_copy_dependency() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].kind, DependencyKind::Copy);
    }

    #[test]
    fn buffer_to_texture_copy() {
        let buf = MetalBuffer::new(4096, "staging");
        let tex = MetalTexture::new(
            32,
            32,
            PixelFormat::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            1,
            "dest",
        );
        let mut graph = DependencyGraph::new();
        graph.add_dependency(buf.id, tex.id, DependencyKind::Copy);
        let deps = graph.dependencies_of(tex.id);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].source_id, buf.id);
    }

    #[test]
    fn resource_aliasing_dependency() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(10, 20, DependencyKind::Alias);
        let deps = graph.dependencies_of(20);
        assert_eq!(deps[0].kind, DependencyKind::Alias);
    }

    #[test]
    fn read_after_write_chain() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::ReadAfterWrite);
        graph.add_dependency(2, 3, DependencyKind::ReadAfterWrite);
        let deps = graph.dependencies_of(3);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].source_id, 2);
    }

    #[test]
    fn write_after_read_dependency() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::WriteAfterRead);
        let deps = graph.dependencies_of(2);
        assert_eq!(deps[0].kind, DependencyKind::WriteAfterRead);
    }

    #[test]
    fn no_cycle_in_dag() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(2, 3, DependencyKind::Copy);
        graph.add_dependency(1, 3, DependencyKind::Copy);
        assert!(!graph.has_cycle());
    }

    #[test]
    fn cycle_detected() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(2, 3, DependencyKind::Copy);
        graph.add_dependency(3, 1, DependencyKind::Copy);
        assert!(graph.has_cycle());
    }

    #[test]
    fn topological_order_for_dag() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(2, 3, DependencyKind::Copy);
        let order = graph.topological_order().unwrap();
        let pos1 = order.iter().position(|&x| x == 1).unwrap();
        let pos2 = order.iter().position(|&x| x == 2).unwrap();
        let pos3 = order.iter().position(|&x| x == 3).unwrap();
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn topological_order_none_for_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(2, 1, DependencyKind::Copy);
        assert!(graph.topological_order().is_none());
    }

    #[test]
    fn barrier_insertion() {
        let mut graph = DependencyGraph::new();
        graph.insert_barrier(BarrierType::BufferBarrier, 1, 2);
        assert_eq!(graph.barrier_count(), 1);
    }

    #[test]
    fn memory_barrier_for_raw() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::ReadAfterWrite);
        let barriers = graph.required_barriers();
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].0, BarrierType::MemoryBarrier);
    }

    #[test]
    fn buffer_barrier_for_copy() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        let barriers = graph.required_barriers();
        assert_eq!(barriers[0].0, BarrierType::BufferBarrier);
    }

    #[test]
    fn texture_barrier_for_alias() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Alias);
        let barriers = graph.required_barriers();
        assert_eq!(barriers[0].0, BarrierType::TextureBarrier);
    }

    #[test]
    fn dependents_of_source() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(1, 3, DependencyKind::Copy);
        let dependents = graph.dependents_of(1);
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn complex_dependency_chain() {
        let mut graph = DependencyGraph::new();
        // staging → compute → render
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(2, 3, DependencyKind::ReadAfterWrite);
        graph.add_dependency(3, 4, DependencyKind::ReadAfterWrite);
        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 4);
        let barriers = graph.required_barriers();
        assert_eq!(barriers.len(), 3);
    }

    #[test]
    fn diamond_dependency_no_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, DependencyKind::Copy);
        graph.add_dependency(1, 3, DependencyKind::Copy);
        graph.add_dependency(2, 4, DependencyKind::Copy);
        graph.add_dependency(3, 4, DependencyKind::Copy);
        assert!(!graph.has_cycle());
        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn self_dependency_is_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 1, DependencyKind::ReadAfterWrite);
        assert!(graph.has_cycle());
    }
}
