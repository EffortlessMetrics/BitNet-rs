#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal compute pipeline optimization tests for Apple Silicon.
//!
//! Validates pipeline state caching, function specialization, threadgroup
//! sizing, indirect dispatch, async compilation, resource binding, shader
//! variant management, workload balancing, pipeline barriers, argument
//! buffers, Apple Silicon capabilities, and regression detection.
//!
//! All types are self-contained mocks — no GPU hardware or Metal/wgpu
//! crates required.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "cpu"))]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

// ===========================================================================
// Mock infrastructure
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineId(u64);

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_pipeline_id() -> PipelineId {
    PipelineId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    shader_name: String,
    constants: Vec<(String, u32)>,
}

#[derive(Debug, Clone)]
struct MockPipelineState {
    id: PipelineId,
    key: PipelineKey,
    compiled: bool,
    compilation_time_us: u64,
}

// ---------------------------------------------------------------------------
// 1. Pipeline State Caching
// ---------------------------------------------------------------------------

struct PsoCache {
    entries: HashMap<PipelineKey, PipelineId>,
    order: VecDeque<PipelineKey>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl PsoCache {
    fn new(capacity: usize) -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new(), capacity, hits: 0, misses: 0 }
    }

    fn lookup(&mut self, key: &PipelineKey) -> Option<PipelineId> {
        if let Some(&id) = self.entries.get(key) {
            self.hits += 1;
            // Move to back (most-recently used)
            self.order.retain(|k| k != key);
            self.order.push_back(key.clone());
            Some(id)
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: PipelineKey, id: PipelineId) {
        if self.entries.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(key.clone(), id);
        self.order.push_back(key);
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn make_key(name: &str) -> PipelineKey {
    PipelineKey { shader_name: name.to_string(), constants: vec![] }
}

fn make_key_with_constants(name: &str, constants: &[(&str, u32)]) -> PipelineKey {
    PipelineKey {
        shader_name: name.to_string(),
        constants: constants.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    }
}

#[test]
fn pso_cache_create_and_lookup() {
    let mut cache = PsoCache::new(16);
    let key = make_key("matmul");
    let id = next_pipeline_id();
    cache.insert(key.clone(), id);
    assert_eq!(cache.lookup(&key), Some(id));
}

#[test]
fn pso_cache_miss_returns_none() {
    let mut cache = PsoCache::new(16);
    assert_eq!(cache.lookup(&make_key("nonexistent")), None);
}

#[test]
fn pso_cache_lru_eviction_at_capacity() {
    let mut cache = PsoCache::new(2);
    let k1 = make_key("a");
    let k2 = make_key("b");
    let k3 = make_key("c");
    cache.insert(k1.clone(), next_pipeline_id());
    cache.insert(k2.clone(), next_pipeline_id());
    // Cache full — inserting k3 evicts k1 (LRU)
    cache.insert(k3.clone(), next_pipeline_id());
    assert_eq!(cache.lookup(&k1), None);
    assert!(cache.lookup(&k2).is_some());
    assert!(cache.lookup(&k3).is_some());
}

#[test]
fn pso_cache_lru_touch_prevents_eviction() {
    let mut cache = PsoCache::new(2);
    let k1 = make_key("a");
    let k2 = make_key("b");
    let k3 = make_key("c");
    cache.insert(k1.clone(), next_pipeline_id());
    cache.insert(k2.clone(), next_pipeline_id());
    // Touch k1 so k2 is now LRU
    let _ = cache.lookup(&k1);
    cache.insert(k3.clone(), next_pipeline_id());
    assert!(cache.lookup(&k1).is_some());
    assert_eq!(cache.lookup(&k2), None); // evicted
}

#[test]
fn pso_cache_hit_rate_tracking() {
    let mut cache = PsoCache::new(8);
    let key = make_key("relu");
    cache.insert(key.clone(), next_pipeline_id());
    let _ = cache.lookup(&key); // hit
    let _ = cache.lookup(&key); // hit
    let _ = cache.lookup(&make_key("miss")); // miss
    assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
}

#[test]
fn pso_cache_zero_capacity_always_evicts() {
    let mut cache = PsoCache::new(0);
    let key = make_key("test");
    // insert does immediate eviction (pop_front from empty deque is None, but entry map stays empty-ish)
    cache.insert(key.clone(), next_pipeline_id());
    // With capacity 0, the insert evicts before adding so len is bounded to 0+1 but the
    // first item inserted has nothing to evict. Second insert will evict the first.
    let k2 = make_key("test2");
    cache.insert(k2.clone(), next_pipeline_id());
    assert!(cache.len() <= 1);
}

#[test]
fn pso_cache_duplicate_key_overwrites() {
    let mut cache = PsoCache::new(4);
    let key = make_key("dup");
    let id1 = next_pipeline_id();
    let id2 = next_pipeline_id();
    cache.insert(key.clone(), id1);
    cache.insert(key.clone(), id2);
    assert_eq!(cache.lookup(&key), Some(id2));
}

#[test]
fn pso_cache_distinct_keys_coexist() {
    let mut cache = PsoCache::new(64);
    let ids: Vec<_> = (0..32)
        .map(|i| {
            let key = make_key(&format!("shader_{i}"));
            let id = next_pipeline_id();
            cache.insert(key, id);
            id
        })
        .collect();
    assert_eq!(cache.len(), 32);
    for (i, expected) in ids.iter().enumerate() {
        let key = make_key(&format!("shader_{i}"));
        assert_eq!(cache.lookup(&key), Some(*expected));
    }
}

// ---------------------------------------------------------------------------
// 2. Function Specialization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FunctionSpecialization {
    base_name: String,
    constants: Vec<(String, u32)>,
}

impl FunctionSpecialization {
    fn new(name: &str) -> Self {
        Self { base_name: name.to_string(), constants: vec![] }
    }

    fn with_constant(mut self, name: &str, value: u32) -> Self {
        self.constants.push((name.to_string(), value));
        self
    }

    fn variant_key(&self) -> String {
        let mut key = self.base_name.clone();
        for (name, val) in &self.constants {
            key.push_str(&format!("_{name}={val}"));
        }
        key
    }

    /// Simulate constant folding: if all constants known at compile time,
    /// fold the variant into a specialized key.
    fn can_fold(&self) -> bool {
        !self.constants.is_empty()
    }
}

#[test]
fn specialization_creates_variant_key() {
    let spec = FunctionSpecialization::new("gemm").with_constant("TILE_M", 16);
    assert_eq!(spec.variant_key(), "gemm_TILE_M=16");
}

#[test]
fn specialization_multiple_constants() {
    let spec =
        FunctionSpecialization::new("gemm").with_constant("TILE_M", 16).with_constant("TILE_N", 8);
    assert_eq!(spec.variant_key(), "gemm_TILE_M=16_TILE_N=8");
}

#[test]
fn specialization_no_constants_no_fold() {
    let spec = FunctionSpecialization::new("relu");
    assert!(!spec.can_fold());
}

#[test]
fn specialization_with_constants_can_fold() {
    let spec = FunctionSpecialization::new("relu").with_constant("INPLACE", 1);
    assert!(spec.can_fold());
}

#[test]
fn specialization_distinct_variants_differ() {
    let a = FunctionSpecialization::new("softmax").with_constant("DIM", 128);
    let b = FunctionSpecialization::new("softmax").with_constant("DIM", 256);
    assert_ne!(a.variant_key(), b.variant_key());
}

#[test]
fn specialization_same_params_equal() {
    let a = FunctionSpecialization::new("ln").with_constant("EPS", 5);
    let b = FunctionSpecialization::new("ln").with_constant("EPS", 5);
    assert_eq!(a.variant_key(), b.variant_key());
}

#[test]
fn specialization_order_matters() {
    let a = FunctionSpecialization::new("f").with_constant("A", 1).with_constant("B", 2);
    let b = FunctionSpecialization::new("f").with_constant("B", 2).with_constant("A", 1);
    // Order-dependent variant keys
    assert_ne!(a.variant_key(), b.variant_key());
}

#[test]
fn specialization_batch_creation() {
    let tile_sizes = [8, 16, 32, 64];
    let variants: Vec<_> = tile_sizes
        .iter()
        .map(|&s| FunctionSpecialization::new("gemm").with_constant("TILE", s))
        .collect();
    assert_eq!(variants.len(), 4);
    let keys: std::collections::HashSet<_> = variants.iter().map(|v| v.variant_key()).collect();
    assert_eq!(keys.len(), 4); // all distinct
}

// ---------------------------------------------------------------------------
// 3. Threadgroup Size Selection
// ---------------------------------------------------------------------------

const APPLE_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

#[derive(Debug, Clone, Copy)]
struct ThreadgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl ThreadgroupSize {
    fn total(&self) -> u32 {
        self.x * self.y * self.z
    }

    fn is_valid_apple(&self) -> bool {
        self.total() <= APPLE_MAX_THREADS_PER_THREADGROUP && self.x > 0 && self.y > 0 && self.z > 0
    }
}

fn optimal_1d_threadgroup(workload: u32) -> ThreadgroupSize {
    let size = workload.min(256).next_power_of_two().min(APPLE_MAX_THREADS_PER_THREADGROUP);
    ThreadgroupSize { x: size, y: 1, z: 1 }
}

fn optimal_2d_threadgroup(width: u32, height: u32) -> ThreadgroupSize {
    let x = width.min(32).max(1);
    let y = height.min(32).max(1);
    let mut tg = ThreadgroupSize { x, y, z: 1 };
    // Shrink to Apple limit
    while tg.total() > APPLE_MAX_THREADS_PER_THREADGROUP && tg.y > 1 {
        tg.y /= 2;
    }
    while tg.total() > APPLE_MAX_THREADS_PER_THREADGROUP && tg.x > 1 {
        tg.x /= 2;
    }
    tg
}

fn optimal_3d_threadgroup(x: u32, y: u32, z: u32) -> ThreadgroupSize {
    let tx = x.min(8).max(1);
    let ty = y.min(8).max(1);
    let tz = z.min(16).max(1);
    let mut tg = ThreadgroupSize { x: tx, y: ty, z: tz };
    while tg.total() > APPLE_MAX_THREADS_PER_THREADGROUP && tg.z > 1 {
        tg.z /= 2;
    }
    while tg.total() > APPLE_MAX_THREADS_PER_THREADGROUP && tg.y > 1 {
        tg.y /= 2;
    }
    tg
}

fn dispatch_groups(workload: u32, threadgroup: u32) -> u32 {
    (workload + threadgroup - 1) / threadgroup
}

#[test]
fn threadgroup_1d_small_workload() {
    let tg = optimal_1d_threadgroup(32);
    assert!(tg.is_valid_apple());
    assert_eq!(tg.x, 32);
}

#[test]
fn threadgroup_1d_large_workload_caps_at_256() {
    let tg = optimal_1d_threadgroup(100_000);
    assert!(tg.is_valid_apple());
    assert_eq!(tg.x, 256);
}

#[test]
fn threadgroup_2d_square() {
    let tg = optimal_2d_threadgroup(16, 16);
    assert!(tg.is_valid_apple());
    assert!(tg.total() <= APPLE_MAX_THREADS_PER_THREADGROUP);
}

#[test]
fn threadgroup_2d_large_shrinks() {
    let tg = optimal_2d_threadgroup(1024, 1024);
    assert!(tg.is_valid_apple());
    assert!(tg.total() <= APPLE_MAX_THREADS_PER_THREADGROUP);
}

#[test]
fn threadgroup_3d_fits_apple_limit() {
    let tg = optimal_3d_threadgroup(8, 8, 16);
    assert!(tg.is_valid_apple());
    assert!(tg.total() <= APPLE_MAX_THREADS_PER_THREADGROUP);
}

#[test]
fn threadgroup_3d_oversized_shrinks() {
    let tg = optimal_3d_threadgroup(64, 64, 64);
    assert!(tg.is_valid_apple());
}

#[test]
fn threadgroup_dispatch_groups_correct() {
    assert_eq!(dispatch_groups(1000, 256), 4);
    assert_eq!(dispatch_groups(256, 256), 1);
    assert_eq!(dispatch_groups(1, 256), 1);
}

#[test]
fn threadgroup_minimum_dimensions() {
    let tg = optimal_2d_threadgroup(0, 0);
    assert!(tg.is_valid_apple());
    assert!(tg.x >= 1 && tg.y >= 1);
}

// ---------------------------------------------------------------------------
// 4. Indirect Dispatch
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct IndirectDispatchArgs {
    threadgroups_x: u32,
    threadgroups_y: u32,
    threadgroups_z: u32,
}

#[derive(Debug)]
struct IndirectCommandBuffer {
    commands: Vec<IndirectDispatchArgs>,
    max_commands: usize,
}

impl IndirectCommandBuffer {
    fn new(max_commands: usize) -> Self {
        Self { commands: Vec::new(), max_commands }
    }

    fn encode(&mut self, args: IndirectDispatchArgs) -> bool {
        if self.commands.len() >= self.max_commands {
            return false;
        }
        self.commands.push(args);
        true
    }

    fn batch_size(&self) -> usize {
        self.commands.len()
    }

    fn total_threadgroups(&self) -> u64 {
        self.commands
            .iter()
            .map(|a| a.threadgroups_x as u64 * a.threadgroups_y as u64 * a.threadgroups_z as u64)
            .sum()
    }

    fn reset(&mut self) {
        self.commands.clear();
    }
}

#[test]
fn indirect_dispatch_encode_single() {
    let mut icb = IndirectCommandBuffer::new(64);
    assert!(icb.encode(IndirectDispatchArgs {
        threadgroups_x: 4,
        threadgroups_y: 1,
        threadgroups_z: 1
    }));
    assert_eq!(icb.batch_size(), 1);
}

#[test]
fn indirect_dispatch_respects_max() {
    let mut icb = IndirectCommandBuffer::new(2);
    icb.encode(IndirectDispatchArgs { threadgroups_x: 1, threadgroups_y: 1, threadgroups_z: 1 });
    icb.encode(IndirectDispatchArgs { threadgroups_x: 1, threadgroups_y: 1, threadgroups_z: 1 });
    assert!(!icb.encode(IndirectDispatchArgs {
        threadgroups_x: 1,
        threadgroups_y: 1,
        threadgroups_z: 1
    }));
}

#[test]
fn indirect_dispatch_total_threadgroups() {
    let mut icb = IndirectCommandBuffer::new(64);
    icb.encode(IndirectDispatchArgs { threadgroups_x: 4, threadgroups_y: 2, threadgroups_z: 1 });
    icb.encode(IndirectDispatchArgs { threadgroups_x: 8, threadgroups_y: 1, threadgroups_z: 1 });
    assert_eq!(icb.total_threadgroups(), 8 + 8);
}

#[test]
fn indirect_dispatch_reset() {
    let mut icb = IndirectCommandBuffer::new(8);
    icb.encode(IndirectDispatchArgs { threadgroups_x: 1, threadgroups_y: 1, threadgroups_z: 1 });
    icb.reset();
    assert_eq!(icb.batch_size(), 0);
}

#[test]
fn indirect_dispatch_batch_encoding() {
    let mut icb = IndirectCommandBuffer::new(128);
    for i in 1..=64 {
        assert!(icb.encode(IndirectDispatchArgs {
            threadgroups_x: i,
            threadgroups_y: 1,
            threadgroups_z: 1
        }));
    }
    assert_eq!(icb.batch_size(), 64);
}

#[test]
fn indirect_dispatch_3d_volume() {
    let mut icb = IndirectCommandBuffer::new(4);
    icb.encode(IndirectDispatchArgs { threadgroups_x: 4, threadgroups_y: 4, threadgroups_z: 4 });
    assert_eq!(icb.total_threadgroups(), 64);
}

#[test]
fn indirect_dispatch_empty_total_zero() {
    let icb = IndirectCommandBuffer::new(16);
    assert_eq!(icb.total_threadgroups(), 0);
}

#[test]
fn indirect_dispatch_reuse_after_reset() {
    let mut icb = IndirectCommandBuffer::new(2);
    icb.encode(IndirectDispatchArgs { threadgroups_x: 1, threadgroups_y: 1, threadgroups_z: 1 });
    icb.encode(IndirectDispatchArgs { threadgroups_x: 1, threadgroups_y: 1, threadgroups_z: 1 });
    assert!(!icb.encode(IndirectDispatchArgs {
        threadgroups_x: 1,
        threadgroups_y: 1,
        threadgroups_z: 1
    }));
    icb.reset();
    assert!(icb.encode(IndirectDispatchArgs {
        threadgroups_x: 1,
        threadgroups_y: 1,
        threadgroups_z: 1
    }));
}

// ---------------------------------------------------------------------------
// 5. Pipeline Compilation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum CompilationStatus {
    Queued,
    Compiling,
    Ready,
    Failed,
}

struct AsyncPipelineCompiler {
    queue: VecDeque<PipelineKey>,
    status: HashMap<PipelineKey, CompilationStatus>,
    compiled: HashMap<PipelineKey, PipelineId>,
    fallback: Option<PipelineId>,
}

impl AsyncPipelineCompiler {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            status: HashMap::new(),
            compiled: HashMap::new(),
            fallback: None,
        }
    }

    fn set_fallback(&mut self, id: PipelineId) {
        self.fallback = Some(id);
    }

    fn submit(&mut self, key: PipelineKey) {
        self.status.insert(key.clone(), CompilationStatus::Queued);
        self.queue.push_back(key);
    }

    /// Process one item from queue (simulate async step).
    fn tick(&mut self) -> bool {
        if let Some(key) = self.queue.pop_front() {
            let status = self.status.get(&key).copied().unwrap_or(CompilationStatus::Queued);
            if status == CompilationStatus::Queued {
                self.status.insert(key.clone(), CompilationStatus::Compiling);
                self.queue.push_front(key);
                return true;
            }
            if status == CompilationStatus::Compiling {
                let id = next_pipeline_id();
                self.compiled.insert(key.clone(), id);
                self.status.insert(key, CompilationStatus::Ready);
                return true;
            }
        }
        false
    }

    fn get_or_fallback(&self, key: &PipelineKey) -> Option<PipelineId> {
        self.compiled.get(key).copied().or(self.fallback)
    }

    fn status(&self, key: &PipelineKey) -> CompilationStatus {
        self.status.get(key).copied().unwrap_or(CompilationStatus::Queued)
    }

    fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

#[test]
fn compilation_submit_queues() {
    let mut compiler = AsyncPipelineCompiler::new();
    compiler.submit(make_key("test"));
    assert_eq!(compiler.status(&make_key("test")), CompilationStatus::Queued);
}

#[test]
fn compilation_tick_transitions_to_compiling() {
    let mut compiler = AsyncPipelineCompiler::new();
    compiler.submit(make_key("test"));
    compiler.tick();
    assert_eq!(compiler.status(&make_key("test")), CompilationStatus::Compiling);
}

#[test]
fn compilation_two_ticks_ready() {
    let mut compiler = AsyncPipelineCompiler::new();
    compiler.submit(make_key("test"));
    compiler.tick();
    compiler.tick();
    assert_eq!(compiler.status(&make_key("test")), CompilationStatus::Ready);
}

#[test]
fn compilation_fallback_while_compiling() {
    let mut compiler = AsyncPipelineCompiler::new();
    let fb = next_pipeline_id();
    compiler.set_fallback(fb);
    compiler.submit(make_key("heavy"));
    compiler.tick(); // compiling, not ready
    assert_eq!(compiler.get_or_fallback(&make_key("heavy")), Some(fb));
}

#[test]
fn compilation_ready_returns_real() {
    let mut compiler = AsyncPipelineCompiler::new();
    let fb = next_pipeline_id();
    compiler.set_fallback(fb);
    compiler.submit(make_key("shader"));
    compiler.tick();
    compiler.tick();
    let result = compiler.get_or_fallback(&make_key("shader"));
    assert!(result.is_some());
    assert_ne!(result.unwrap(), fb);
}

#[test]
fn compilation_multiple_queued() {
    let mut compiler = AsyncPipelineCompiler::new();
    compiler.submit(make_key("a"));
    compiler.submit(make_key("b"));
    compiler.submit(make_key("c"));
    assert_eq!(compiler.queue_len(), 3);
}

#[test]
fn compilation_process_all() {
    let mut compiler = AsyncPipelineCompiler::new();
    for i in 0..4 {
        compiler.submit(make_key(&format!("s{i}")));
    }
    // Process all: each needs 2 ticks
    for _ in 0..20 {
        if !compiler.tick() {
            break;
        }
    }
    for i in 0..4 {
        assert_eq!(compiler.status(&make_key(&format!("s{i}"))), CompilationStatus::Ready);
    }
}

#[test]
fn compilation_no_fallback_returns_none() {
    let compiler = AsyncPipelineCompiler::new();
    assert_eq!(compiler.get_or_fallback(&make_key("x")), None);
}

// ---------------------------------------------------------------------------
// 6. Resource Binding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum ResourceType {
    Buffer,
    Texture,
    Sampler,
}

#[derive(Debug, Clone)]
struct BindGroupEntry {
    binding: u32,
    resource_type: ResourceType,
    offset: usize,
    size: usize,
}

struct ArgumentBufferLayout {
    entries: Vec<BindGroupEntry>,
    alignment: usize,
}

impl ArgumentBufferLayout {
    fn new(alignment: usize) -> Self {
        Self { entries: Vec::new(), alignment }
    }

    fn add(&mut self, binding: u32, rt: ResourceType, size: usize) {
        let offset = if let Some(last) = self.entries.last() {
            let raw = last.offset + last.size;
            // Align up
            (raw + self.alignment - 1) / self.alignment * self.alignment
        } else {
            0
        };
        self.entries.push(BindGroupEntry { binding, resource_type: rt, offset, size });
    }

    fn total_size(&self) -> usize {
        self.entries.last().map_or(0, |e| {
            let raw = e.offset + e.size;
            (raw + self.alignment - 1) / self.alignment * self.alignment
        })
    }

    fn binding_count(&self) -> usize {
        self.entries.len()
    }

    fn is_resident(&self, binding: u32) -> bool {
        self.entries.iter().any(|e| e.binding == binding)
    }
}

#[test]
fn resource_binding_layout_empty() {
    let layout = ArgumentBufferLayout::new(16);
    assert_eq!(layout.total_size(), 0);
    assert_eq!(layout.binding_count(), 0);
}

#[test]
fn resource_binding_single_buffer() {
    let mut layout = ArgumentBufferLayout::new(16);
    layout.add(0, ResourceType::Buffer, 256);
    assert_eq!(layout.binding_count(), 1);
    assert_eq!(layout.total_size(), 256);
}

#[test]
fn resource_binding_alignment() {
    let mut layout = ArgumentBufferLayout::new(16);
    layout.add(0, ResourceType::Buffer, 10); // 10 bytes, aligned to 16
    layout.add(1, ResourceType::Buffer, 10);
    // First entry at offset 0 size 10; second at offset 16 size 10
    assert_eq!(layout.entries[1].offset, 16);
}

#[test]
fn resource_binding_mixed_types() {
    let mut layout = ArgumentBufferLayout::new(8);
    layout.add(0, ResourceType::Buffer, 64);
    layout.add(1, ResourceType::Texture, 32);
    layout.add(2, ResourceType::Sampler, 8);
    assert_eq!(layout.binding_count(), 3);
    assert!(layout.total_size() > 0);
}

#[test]
fn resource_binding_residency_check() {
    let mut layout = ArgumentBufferLayout::new(8);
    layout.add(0, ResourceType::Buffer, 64);
    layout.add(3, ResourceType::Texture, 32);
    assert!(layout.is_resident(0));
    assert!(layout.is_resident(3));
    assert!(!layout.is_resident(1));
}

#[test]
fn resource_binding_many_entries() {
    let mut layout = ArgumentBufferLayout::new(16);
    for i in 0..16 {
        layout.add(i, ResourceType::Buffer, 64);
    }
    assert_eq!(layout.binding_count(), 16);
    assert!(layout.total_size() >= 16 * 64);
}

#[test]
fn resource_binding_offsets_monotonic() {
    let mut layout = ArgumentBufferLayout::new(8);
    for i in 0..8 {
        layout.add(i, ResourceType::Buffer, 24);
    }
    for w in layout.entries.windows(2) {
        assert!(w[1].offset > w[0].offset);
    }
}

#[test]
fn resource_binding_power_of_two_alignment() {
    for align in [4, 8, 16, 32, 64] {
        let mut layout = ArgumentBufferLayout::new(align);
        layout.add(0, ResourceType::Buffer, 1);
        layout.add(1, ResourceType::Buffer, 1);
        assert_eq!(layout.entries[1].offset % align, 0);
    }
}

// ---------------------------------------------------------------------------
// 7. Shader Variant Management
// ---------------------------------------------------------------------------

struct ShaderVariantManager {
    variants: HashMap<String, Vec<String>>,
    usage_counts: HashMap<String, u64>,
}

impl ShaderVariantManager {
    fn new() -> Self {
        Self { variants: HashMap::new(), usage_counts: HashMap::new() }
    }

    fn register(&mut self, base: &str, variant_key: &str) {
        self.variants.entry(base.to_string()).or_default().push(variant_key.to_string());
        self.usage_counts.entry(variant_key.to_string()).or_insert(0);
    }

    fn record_use(&mut self, variant_key: &str) {
        if let Some(c) = self.usage_counts.get_mut(variant_key) {
            *c += 1;
        }
    }

    fn variant_count(&self, base: &str) -> usize {
        self.variants.get(base).map_or(0, |v| v.len())
    }

    fn total_variants(&self) -> usize {
        self.variants.values().map(|v| v.len()).sum()
    }

    /// Prune variants with zero usage (dead code elimination simulation).
    fn prune_unused(&mut self) -> usize {
        let unused: Vec<String> = self
            .usage_counts
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(k, _)| k.clone())
            .collect();
        let count = unused.len();
        for key in &unused {
            self.usage_counts.remove(key);
            for variants in self.variants.values_mut() {
                variants.retain(|v| v != key);
            }
        }
        count
    }
}

#[test]
fn variant_register_and_count() {
    let mut mgr = ShaderVariantManager::new();
    mgr.register("gemm", "gemm_tile16");
    mgr.register("gemm", "gemm_tile32");
    assert_eq!(mgr.variant_count("gemm"), 2);
}

#[test]
fn variant_total_across_shaders() {
    let mut mgr = ShaderVariantManager::new();
    mgr.register("gemm", "gemm_t16");
    mgr.register("gemm", "gemm_t32");
    mgr.register("relu", "relu_inplace");
    assert_eq!(mgr.total_variants(), 3);
}

#[test]
fn variant_prune_removes_unused() {
    let mut mgr = ShaderVariantManager::new();
    mgr.register("gemm", "gemm_t16");
    mgr.register("gemm", "gemm_t32");
    mgr.record_use("gemm_t16");
    let pruned = mgr.prune_unused();
    assert_eq!(pruned, 1); // gemm_t32 unused
    assert_eq!(mgr.variant_count("gemm"), 1);
}

#[test]
fn variant_prune_keeps_used() {
    let mut mgr = ShaderVariantManager::new();
    mgr.register("softmax", "softmax_dim128");
    mgr.record_use("softmax_dim128");
    let pruned = mgr.prune_unused();
    assert_eq!(pruned, 0);
    assert_eq!(mgr.variant_count("softmax"), 1);
}

#[test]
fn variant_unknown_base_returns_zero() {
    let mgr = ShaderVariantManager::new();
    assert_eq!(mgr.variant_count("nonexistent"), 0);
}

#[test]
fn variant_usage_tracking() {
    let mut mgr = ShaderVariantManager::new();
    mgr.register("ln", "ln_eps5");
    mgr.record_use("ln_eps5");
    mgr.record_use("ln_eps5");
    mgr.record_use("ln_eps5");
    assert_eq!(*mgr.usage_counts.get("ln_eps5").unwrap(), 3);
}

#[test]
fn variant_prune_all_unused() {
    let mut mgr = ShaderVariantManager::new();
    for i in 0..10 {
        mgr.register("f", &format!("f_{i}"));
    }
    let pruned = mgr.prune_unused();
    assert_eq!(pruned, 10);
    assert_eq!(mgr.total_variants(), 0);
}

#[test]
fn variant_empty_manager_prune() {
    let mut mgr = ShaderVariantManager::new();
    assert_eq!(mgr.prune_unused(), 0);
}

// ---------------------------------------------------------------------------
// 8. Workload Balancing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct GpuOccupancy {
    active_waves: u32,
    max_waves: u32,
    simd_width: u32,
    threadgroup_size: u32,
}

impl GpuOccupancy {
    fn occupancy_ratio(&self) -> f64 {
        if self.max_waves == 0 {
            return 0.0;
        }
        self.active_waves as f64 / self.max_waves as f64
    }

    fn simd_utilization(&self) -> f64 {
        if self.simd_width == 0 {
            return 0.0;
        }
        let used = self.threadgroup_size % self.simd_width;
        if used == 0 { 1.0 } else { used as f64 / self.simd_width as f64 }
    }
}

fn estimate_waves(total_threads: u32, threadgroup_size: u32, simd_width: u32) -> u32 {
    if threadgroup_size == 0 || simd_width == 0 {
        return 0;
    }
    let groups = (total_threads + threadgroup_size - 1) / threadgroup_size;
    let waves_per_group = (threadgroup_size + simd_width - 1) / simd_width;
    groups * waves_per_group
}

fn schedule_workgroups(total: u32, group_size: u32, max_concurrent: u32) -> Vec<u32> {
    if group_size == 0 {
        return vec![];
    }
    let groups = (total + group_size - 1) / group_size;
    let mut batches = Vec::new();
    let mut remaining = groups;
    while remaining > 0 {
        let batch = remaining.min(max_concurrent);
        batches.push(batch);
        remaining -= batch;
    }
    batches
}

#[test]
fn occupancy_full() {
    let occ =
        GpuOccupancy { active_waves: 48, max_waves: 48, simd_width: 32, threadgroup_size: 256 };
    assert!((occ.occupancy_ratio() - 1.0).abs() < 1e-9);
}

#[test]
fn occupancy_half() {
    let occ =
        GpuOccupancy { active_waves: 24, max_waves: 48, simd_width: 32, threadgroup_size: 256 };
    assert!((occ.occupancy_ratio() - 0.5).abs() < 1e-9);
}

#[test]
fn simd_utilization_aligned() {
    let occ = GpuOccupancy { active_waves: 1, max_waves: 1, simd_width: 32, threadgroup_size: 256 };
    assert!((occ.simd_utilization() - 1.0).abs() < 1e-9);
}

#[test]
fn simd_utilization_partial() {
    let occ = GpuOccupancy { active_waves: 1, max_waves: 1, simd_width: 32, threadgroup_size: 100 };
    // 100 % 32 = 4, utilization = 4/32 = 0.125
    assert!((occ.simd_utilization() - 0.125).abs() < 1e-9);
}

#[test]
fn estimate_waves_basic() {
    assert_eq!(estimate_waves(1024, 256, 32), 4 * 8);
}

#[test]
fn estimate_waves_single_group() {
    assert_eq!(estimate_waves(64, 256, 32), 1 * 8);
}

#[test]
fn schedule_workgroups_fits_one_batch() {
    let batches = schedule_workgroups(256, 256, 64);
    assert_eq!(batches, vec![1]);
}

#[test]
fn schedule_workgroups_multiple_batches() {
    let batches = schedule_workgroups(2048, 256, 4);
    // 8 groups, 4 max concurrent → 2 batches of 4
    assert_eq!(batches, vec![4, 4]);
}

// ---------------------------------------------------------------------------
// 9. Pipeline Barriers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResourceId(u32);

#[derive(Debug, Clone, Copy, PartialEq)]
enum AccessType {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
struct ResourceAccess {
    resource: ResourceId,
    access: AccessType,
    pass_index: u32,
}

struct HazardTracker {
    accesses: Vec<ResourceAccess>,
}

impl HazardTracker {
    fn new() -> Self {
        Self { accesses: Vec::new() }
    }

    fn record(&mut self, resource: ResourceId, access: AccessType, pass_index: u32) {
        self.accesses.push(ResourceAccess { resource, access, pass_index });
    }

    /// Detect write-after-read and read-after-write hazards.
    fn find_hazards(&self) -> Vec<(ResourceId, u32, u32)> {
        let mut hazards = Vec::new();
        for (i, a) in self.accesses.iter().enumerate() {
            for b in &self.accesses[i + 1..] {
                if a.resource == b.resource && a.pass_index != b.pass_index {
                    let has_write = matches!(a.access, AccessType::Write | AccessType::ReadWrite)
                        || matches!(b.access, AccessType::Write | AccessType::ReadWrite);
                    if has_write {
                        hazards.push((a.resource, a.pass_index, b.pass_index));
                    }
                }
            }
        }
        hazards
    }

    fn needs_barrier(&self, before: u32, after: u32) -> bool {
        self.find_hazards().iter().any(|&(_, p1, p2)| p1 == before && p2 == after)
    }
}

#[test]
fn barrier_no_hazards_for_reads() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::Read, 0);
    tracker.record(ResourceId(0), AccessType::Read, 1);
    assert!(tracker.find_hazards().is_empty());
}

#[test]
fn barrier_war_hazard_detected() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::Write, 0);
    tracker.record(ResourceId(0), AccessType::Read, 1);
    assert_eq!(tracker.find_hazards().len(), 1);
}

#[test]
fn barrier_raw_hazard_detected() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::Read, 0);
    tracker.record(ResourceId(0), AccessType::Write, 1);
    assert_eq!(tracker.find_hazards().len(), 1);
}

#[test]
fn barrier_waw_hazard_detected() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::Write, 0);
    tracker.record(ResourceId(0), AccessType::Write, 1);
    assert_eq!(tracker.find_hazards().len(), 1);
}

#[test]
fn barrier_needs_barrier_check() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(1), AccessType::Write, 0);
    tracker.record(ResourceId(1), AccessType::Read, 1);
    assert!(tracker.needs_barrier(0, 1));
    assert!(!tracker.needs_barrier(1, 2));
}

#[test]
fn barrier_independent_resources_no_hazard() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::Write, 0);
    tracker.record(ResourceId(1), AccessType::Write, 1);
    assert!(tracker.find_hazards().is_empty());
}

#[test]
fn barrier_multiple_resources_mixed() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::Write, 0);
    tracker.record(ResourceId(0), AccessType::Read, 1);
    tracker.record(ResourceId(1), AccessType::Read, 0);
    tracker.record(ResourceId(1), AccessType::Read, 1);
    assert_eq!(tracker.find_hazards().len(), 1); // only resource 0
}

#[test]
fn barrier_readwrite_access_creates_hazard() {
    let mut tracker = HazardTracker::new();
    tracker.record(ResourceId(0), AccessType::ReadWrite, 0);
    tracker.record(ResourceId(0), AccessType::Read, 1);
    assert!(!tracker.find_hazards().is_empty());
}

// ---------------------------------------------------------------------------
// 10. Argument Buffers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum ArgumentBufferTier {
    Tier1,
    Tier2,
}

struct ArgumentBuffer {
    tier: ArgumentBufferTier,
    indices: Vec<u32>,
    encoded_resources: HashMap<u32, (ResourceType, usize)>,
    max_entries: usize,
}

impl ArgumentBuffer {
    fn new(tier: ArgumentBufferTier) -> Self {
        let max_entries = match tier {
            ArgumentBufferTier::Tier1 => 31, // Tier 1 limited to 31 entries
            ArgumentBufferTier::Tier2 => 500_000,
        };
        Self { tier, indices: Vec::new(), encoded_resources: HashMap::new(), max_entries }
    }

    fn encode(&mut self, index: u32, rt: ResourceType, size: usize) -> bool {
        if self.encoded_resources.len() >= self.max_entries {
            return false;
        }
        self.indices.push(index);
        self.encoded_resources.insert(index, (rt, size));
        true
    }

    fn resource_count(&self) -> usize {
        self.encoded_resources.len()
    }

    fn has_resource(&self, index: u32) -> bool {
        self.encoded_resources.contains_key(&index)
    }

    fn is_tier2(&self) -> bool {
        self.tier == ArgumentBufferTier::Tier2
    }

    fn total_encoded_size(&self) -> usize {
        self.encoded_resources.values().map(|(_, s)| s).sum()
    }
}

#[test]
fn argbuf_tier1_creation() {
    let ab = ArgumentBuffer::new(ArgumentBufferTier::Tier1);
    assert!(!ab.is_tier2());
    assert_eq!(ab.resource_count(), 0);
}

#[test]
fn argbuf_tier2_creation() {
    let ab = ArgumentBuffer::new(ArgumentBufferTier::Tier2);
    assert!(ab.is_tier2());
}

#[test]
fn argbuf_encode_and_lookup() {
    let mut ab = ArgumentBuffer::new(ArgumentBufferTier::Tier1);
    assert!(ab.encode(0, ResourceType::Buffer, 256));
    assert!(ab.has_resource(0));
    assert!(!ab.has_resource(1));
}

#[test]
fn argbuf_tier1_limit() {
    let mut ab = ArgumentBuffer::new(ArgumentBufferTier::Tier1);
    for i in 0..31 {
        assert!(ab.encode(i, ResourceType::Buffer, 8));
    }
    assert!(!ab.encode(31, ResourceType::Buffer, 8)); // exceeds tier 1
}

#[test]
fn argbuf_tier2_large_capacity() {
    let mut ab = ArgumentBuffer::new(ArgumentBufferTier::Tier2);
    for i in 0..1000 {
        assert!(ab.encode(i, ResourceType::Buffer, 8));
    }
    assert_eq!(ab.resource_count(), 1000);
}

#[test]
fn argbuf_mixed_resource_types() {
    let mut ab = ArgumentBuffer::new(ArgumentBufferTier::Tier1);
    ab.encode(0, ResourceType::Buffer, 64);
    ab.encode(1, ResourceType::Texture, 32);
    ab.encode(2, ResourceType::Sampler, 4);
    assert_eq!(ab.resource_count(), 3);
}

#[test]
fn argbuf_total_encoded_size() {
    let mut ab = ArgumentBuffer::new(ArgumentBufferTier::Tier1);
    ab.encode(0, ResourceType::Buffer, 100);
    ab.encode(1, ResourceType::Buffer, 200);
    assert_eq!(ab.total_encoded_size(), 300);
}

#[test]
fn argbuf_index_tracking() {
    let mut ab = ArgumentBuffer::new(ArgumentBufferTier::Tier2);
    ab.encode(5, ResourceType::Buffer, 8);
    ab.encode(10, ResourceType::Texture, 16);
    ab.encode(99, ResourceType::Sampler, 4);
    assert!(ab.has_resource(5));
    assert!(ab.has_resource(10));
    assert!(ab.has_resource(99));
    assert!(!ab.has_resource(0));
}

// ---------------------------------------------------------------------------
// 11. Apple Silicon Pipeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppleGpuFamily {
    Apple7, // M1
    Apple8, // M2
    Apple9, // M3
}

#[derive(Debug, Clone)]
struct AppleGpuCapabilities {
    family: AppleGpuFamily,
    max_threadgroup_memory: usize,
    max_threads_per_threadgroup: u32,
    simdgroup_size: u32,
    supports_tile_shading: bool,
    supports_simdgroup_reduction: bool,
    supports_bfloat16: bool,
}

impl AppleGpuCapabilities {
    fn for_family(family: AppleGpuFamily) -> Self {
        match family {
            AppleGpuFamily::Apple7 => Self {
                family,
                max_threadgroup_memory: 32_768,
                max_threads_per_threadgroup: 1024,
                simdgroup_size: 32,
                supports_tile_shading: true,
                supports_simdgroup_reduction: true,
                supports_bfloat16: false,
            },
            AppleGpuFamily::Apple8 => Self {
                family,
                max_threadgroup_memory: 32_768,
                max_threads_per_threadgroup: 1024,
                simdgroup_size: 32,
                supports_tile_shading: true,
                supports_simdgroup_reduction: true,
                supports_bfloat16: true,
            },
            AppleGpuFamily::Apple9 => Self {
                family,
                max_threadgroup_memory: 65_536,
                max_threads_per_threadgroup: 1024,
                simdgroup_size: 32,
                supports_tile_shading: true,
                supports_simdgroup_reduction: true,
                supports_bfloat16: true,
            },
        }
    }

    fn optimal_threadgroup_for_reduction(&self, elements: u32) -> u32 {
        // Apple GPUs prefer multiples of simdgroup_size
        let ideal = elements.min(self.max_threads_per_threadgroup);
        let aligned = (ideal / self.simdgroup_size) * self.simdgroup_size;
        aligned.max(self.simdgroup_size)
    }
}

#[test]
fn apple_m1_capabilities() {
    let caps = AppleGpuCapabilities::for_family(AppleGpuFamily::Apple7);
    assert_eq!(caps.max_threads_per_threadgroup, 1024);
    assert!(!caps.supports_bfloat16);
    assert!(caps.supports_tile_shading);
}

#[test]
fn apple_m2_bf16_support() {
    let caps = AppleGpuCapabilities::for_family(AppleGpuFamily::Apple8);
    assert!(caps.supports_bfloat16);
}

#[test]
fn apple_m3_extended_threadgroup_memory() {
    let caps = AppleGpuCapabilities::for_family(AppleGpuFamily::Apple9);
    assert_eq!(caps.max_threadgroup_memory, 65_536);
}

#[test]
fn apple_simdgroup_size_always_32() {
    for family in [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9] {
        let caps = AppleGpuCapabilities::for_family(family);
        assert_eq!(caps.simdgroup_size, 32);
    }
}

#[test]
fn apple_reduction_threadgroup_aligned() {
    let caps = AppleGpuCapabilities::for_family(AppleGpuFamily::Apple7);
    let tg = caps.optimal_threadgroup_for_reduction(1000);
    assert_eq!(tg % 32, 0);
    assert!(tg <= 1024);
}

#[test]
fn apple_reduction_small_elements() {
    let caps = AppleGpuCapabilities::for_family(AppleGpuFamily::Apple8);
    let tg = caps.optimal_threadgroup_for_reduction(16);
    assert_eq!(tg, 32); // minimum is one simdgroup
}

#[test]
fn apple_all_families_support_simdgroup_reduction() {
    for family in [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9] {
        assert!(AppleGpuCapabilities::for_family(family).supports_simdgroup_reduction);
    }
}

#[test]
fn apple_all_families_max_1024_threads() {
    for family in [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9] {
        assert_eq!(AppleGpuCapabilities::for_family(family).max_threads_per_threadgroup, 1024);
    }
}

// ---------------------------------------------------------------------------
// 12. Regression Detection
// ---------------------------------------------------------------------------

struct RegressionDetector {
    compilation_time_budget_us: u64,
    max_pso_cache_entries: usize,
    max_bindings_per_group: usize,
}

impl RegressionDetector {
    fn new() -> Self {
        Self {
            compilation_time_budget_us: 50_000, // 50ms
            max_pso_cache_entries: 256,
            max_bindings_per_group: 31, // Metal tier 1
        }
    }

    fn check_compilation_time(&self, actual_us: u64) -> Result<(), String> {
        if actual_us > self.compilation_time_budget_us {
            Err(format!(
                "compilation time {actual_us}us exceeds budget {}us",
                self.compilation_time_budget_us
            ))
        } else {
            Ok(())
        }
    }

    fn check_cache_size(&self, actual: usize) -> Result<(), String> {
        if actual > self.max_pso_cache_entries {
            Err(format!("PSO cache size {actual} exceeds max {}", self.max_pso_cache_entries))
        } else {
            Ok(())
        }
    }

    fn check_binding_count(&self, actual: usize) -> Result<(), String> {
        if actual > self.max_bindings_per_group {
            Err(format!(
                "binding count {actual} exceeds tier-1 max {}",
                self.max_bindings_per_group
            ))
        } else {
            Ok(())
        }
    }

    fn check_all(
        &self,
        compilation_us: u64,
        cache_size: usize,
        binding_count: usize,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if let Err(e) = self.check_compilation_time(compilation_us) {
            errors.push(e);
        }
        if let Err(e) = self.check_cache_size(cache_size) {
            errors.push(e);
        }
        if let Err(e) = self.check_binding_count(binding_count) {
            errors.push(e);
        }
        errors
    }
}

#[test]
fn regression_compilation_time_ok() {
    let det = RegressionDetector::new();
    assert!(det.check_compilation_time(10_000).is_ok());
}

#[test]
fn regression_compilation_time_exceeded() {
    let det = RegressionDetector::new();
    assert!(det.check_compilation_time(100_000).is_err());
}

#[test]
fn regression_cache_size_ok() {
    let det = RegressionDetector::new();
    assert!(det.check_cache_size(128).is_ok());
}

#[test]
fn regression_cache_size_exceeded() {
    let det = RegressionDetector::new();
    assert!(det.check_cache_size(512).is_err());
}

#[test]
fn regression_binding_count_ok() {
    let det = RegressionDetector::new();
    assert!(det.check_binding_count(16).is_ok());
}

#[test]
fn regression_binding_count_exceeded() {
    let det = RegressionDetector::new();
    assert!(det.check_binding_count(64).is_err());
}

#[test]
fn regression_check_all_passes() {
    let det = RegressionDetector::new();
    let errors = det.check_all(1_000, 64, 8);
    assert!(errors.is_empty());
}

#[test]
fn regression_check_all_multiple_failures() {
    let det = RegressionDetector::new();
    let errors = det.check_all(999_999, 999, 999);
    assert_eq!(errors.len(), 3);
}
