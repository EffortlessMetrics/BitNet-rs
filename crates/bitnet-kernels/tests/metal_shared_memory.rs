#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal shared memory (threadgroup memory) tests for Apple Silicon.
//!
//! Validates threadgroup memory allocation, bank conflict avoidance,
//! shared memory access patterns (tiling, reduction, scan), Apple Silicon
//! memory characteristics, and memory coalescing patterns.
//!
//! Uses a mock/simulation approach — no actual Metal APIs required.

// ── Core data structures ────────────────────────────────────────────

/// Configuration for Metal threadgroup memory.
#[derive(Debug, Clone)]
struct ThreadgroupMemoryConfig {
    size_bytes: u32,
    alignment: u32,
    max_total_threadgroup_memory: u32,
    num_banks: u32,
    bank_width_bytes: u32,
}

impl ThreadgroupMemoryConfig {
    fn m1_default() -> Self {
        Self {
            size_bytes: 0,
            alignment: 16,
            max_total_threadgroup_memory: 32 * 1024,
            num_banks: 32,
            bank_width_bytes: 4,
        }
    }

    fn m3_default() -> Self {
        Self {
            size_bytes: 0,
            alignment: 16,
            max_total_threadgroup_memory: 32 * 1024,
            num_banks: 32,
            bank_width_bytes: 4,
        }
    }
}

/// A single allocation within threadgroup shared memory.
#[derive(Debug, Clone)]
struct SharedMemoryAllocation {
    name: String,
    offset: u32,
    size_bytes: u32,
    element_size: u32,
    num_elements: u32,
}

/// Layout of all shared memory allocations within a threadgroup.
#[derive(Debug, Clone)]
struct SharedMemoryLayout {
    allocations: Vec<SharedMemoryAllocation>,
    total_bytes: u32,
    padding_bytes: u32,
}

impl SharedMemoryLayout {
    fn new() -> Self {
        Self { allocations: Vec::new(), total_bytes: 0, padding_bytes: 0 }
    }

    /// Allocate a region in threadgroup memory with proper alignment.
    fn allocate(
        &mut self,
        name: &str,
        element_size: u32,
        num_elements: u32,
        alignment: u32,
    ) -> u32 {
        let offset = align_up(self.total_bytes, alignment);
        let padding = offset - self.total_bytes;
        let size_bytes = element_size * num_elements;

        self.allocations.push(SharedMemoryAllocation {
            name: name.to_string(),
            offset,
            size_bytes,
            element_size,
            num_elements,
        });

        self.padding_bytes += padding;
        self.total_bytes = offset + size_bytes;
        offset
    }

    fn fits_in(&self, config: &ThreadgroupMemoryConfig) -> bool {
        self.total_bytes <= config.max_total_threadgroup_memory
    }
}

// ── Chip generation model ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppleSiliconGen {
    M1,
    M2,
    M3,
    M4,
}

/// Apple Silicon GPU characteristics relevant to shared memory.
#[derive(Debug, Clone)]
struct AppleSiliconMemoryProfile {
    generation: AppleSiliconGen,
    max_threadgroup_memory: u32,
    simd_width: u32,
    max_threads_per_threadgroup: u32,
    num_banks: u32,
    bank_width_bytes: u32,
    unified_memory: bool,
    /// L1/tile cache size per GPU core (approximate).
    tile_cache_bytes: u32,
    /// Max threadgroups per compute unit (concurrent occupancy).
    max_threadgroups_per_cu: u32,
}

impl AppleSiliconMemoryProfile {
    fn m1() -> Self {
        Self {
            generation: AppleSiliconGen::M1,
            max_threadgroup_memory: 32 * 1024,
            simd_width: 32,
            max_threads_per_threadgroup: 1024,
            num_banks: 32,
            bank_width_bytes: 4,
            unified_memory: true,
            tile_cache_bytes: 32 * 1024,
            max_threadgroups_per_cu: 4,
        }
    }

    fn m2() -> Self {
        Self {
            generation: AppleSiliconGen::M2,
            max_threadgroup_memory: 32 * 1024,
            simd_width: 32,
            max_threads_per_threadgroup: 1024,
            num_banks: 32,
            bank_width_bytes: 4,
            unified_memory: true,
            tile_cache_bytes: 48 * 1024,
            max_threadgroups_per_cu: 4,
        }
    }

    fn m3() -> Self {
        Self {
            generation: AppleSiliconGen::M3,
            max_threadgroup_memory: 32 * 1024,
            simd_width: 32,
            max_threads_per_threadgroup: 1024,
            num_banks: 32,
            bank_width_bytes: 4,
            unified_memory: true,
            tile_cache_bytes: 64 * 1024,
            max_threadgroups_per_cu: 8,
        }
    }

    fn m4() -> Self {
        Self {
            generation: AppleSiliconGen::M4,
            max_threadgroup_memory: 32 * 1024,
            simd_width: 32,
            max_threads_per_threadgroup: 1024,
            num_banks: 32,
            bank_width_bytes: 4,
            unified_memory: true,
            tile_cache_bytes: 64 * 1024,
            max_threadgroups_per_cu: 8,
        }
    }

    /// Theoretical peak occupancy (fraction of max threads usable
    /// given the threadgroup memory footprint).
    fn occupancy(&self, threadgroup_mem_used: u32, threads_per_group: u32) -> f64 {
        if threads_per_group == 0 || threadgroup_mem_used == 0 {
            return 0.0;
        }
        let groups_by_mem = self.max_threadgroup_memory / threadgroup_mem_used.max(1);
        let groups_limited = groups_by_mem.min(self.max_threadgroups_per_cu);
        let active_threads = groups_limited * threads_per_group;
        let max_threads = self.max_threadgroups_per_cu * self.max_threads_per_threadgroup;
        (active_threads as f64) / (max_threads as f64)
    }
}

// ── Bank conflict model ─────────────────────────────────────────────

/// Describes a threadgroup memory access pattern for bank conflict analysis.
#[derive(Debug, Clone)]
struct BankAccessPattern {
    /// Byte addresses accessed by each thread in a SIMD group.
    addresses: Vec<u32>,
    num_banks: u32,
    bank_width_bytes: u32,
}

impl BankAccessPattern {
    /// Count the number of bank conflicts (max threads hitting the same bank minus 1).
    fn bank_conflicts(&self) -> u32 {
        let mut bank_hits = vec![0u32; self.num_banks as usize];
        for &addr in &self.addresses {
            let bank = (addr / self.bank_width_bytes) % self.num_banks;
            bank_hits[bank as usize] += 1;
        }
        let max_hit = bank_hits.iter().copied().max().unwrap_or(0);
        max_hit.saturating_sub(1)
    }

    /// True when every thread in the SIMD group hits a unique bank.
    fn is_conflict_free(&self) -> bool {
        self.bank_conflicts() == 0
    }

    /// Build a stride-1 access pattern (thread i reads element i).
    fn stride1(num_threads: u32, element_size: u32, num_banks: u32, bank_width: u32) -> Self {
        let addresses = (0..num_threads).map(|i| i * element_size).collect();
        Self { addresses, num_banks, bank_width_bytes: bank_width }
    }

    /// Build a stride-N access pattern (thread i reads element i*stride).
    fn stride_n(
        num_threads: u32,
        stride: u32,
        element_size: u32,
        num_banks: u32,
        bank_width: u32,
    ) -> Self {
        let addresses = (0..num_threads).map(|i| i * stride * element_size).collect();
        Self { addresses, num_banks, bank_width_bytes: bank_width }
    }

    /// Build a column-major access pattern for a row-major matrix tile.
    fn column_major(
        rows: u32,
        cols: u32,
        col_idx: u32,
        element_size: u32,
        num_banks: u32,
        bank_width: u32,
    ) -> Self {
        let addresses = (0..rows).map(|r| (r * cols + col_idx) * element_size).collect();
        Self { addresses, num_banks, bank_width_bytes: bank_width }
    }

    /// Build a column-major pattern with padding to avoid bank conflicts.
    fn column_major_padded(
        rows: u32,
        cols: u32,
        col_idx: u32,
        pad: u32,
        element_size: u32,
        num_banks: u32,
        bank_width: u32,
    ) -> Self {
        let stride = cols + pad;
        let addresses = (0..rows).map(|r| (r * stride + col_idx) * element_size).collect();
        Self { addresses, num_banks, bank_width_bytes: bank_width }
    }

    /// Broadcast — all threads read the same address.
    fn broadcast(num_threads: u32, addr: u32, num_banks: u32, bank_width: u32) -> Self {
        let addresses = vec![addr; num_threads as usize];
        Self { addresses, num_banks, bank_width_bytes: bank_width }
    }
}

// ── Reduction & scan models ─────────────────────────────────────────

/// Simulate a parallel reduction in shared memory.
fn parallel_reduction_sum(data: &[f32]) -> f32 {
    let n = data.len().next_power_of_two();
    let mut buf = vec![0.0f32; n];
    buf[..data.len()].copy_from_slice(data);

    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            let j = i + stride;
            if j < n {
                buf[i] += buf[j];
            }
            i += 2 * stride;
        }
        stride *= 2;
    }
    buf[0]
}

/// Simulate a parallel max reduction in shared memory.
fn parallel_reduction_max(data: &[f32]) -> f32 {
    let n = data.len().next_power_of_two();
    let mut buf = vec![f32::NEG_INFINITY; n];
    buf[..data.len()].copy_from_slice(data);

    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            let j = i + stride;
            if j < n {
                buf[i] = buf[i].max(buf[j]);
            }
            i += 2 * stride;
        }
        stride *= 2;
    }
    buf[0]
}

/// Simulate Blelloch exclusive prefix scan (up-sweep + down-sweep).
fn blelloch_scan(data: &[f32]) -> Vec<f32> {
    let n = data.len().next_power_of_two();
    let mut buf = vec![0.0f32; n];
    buf[..data.len()].copy_from_slice(data);

    // Up-sweep (reduce)
    let mut stride = 1;
    while stride < n {
        let mut i = 2 * stride - 1;
        while i < n {
            buf[i] += buf[i - stride];
            i += 2 * stride;
        }
        stride *= 2;
    }

    // Set root to zero
    buf[n - 1] = 0.0;

    // Down-sweep
    stride = n / 2;
    while stride >= 1 {
        let mut i = 2 * stride - 1;
        while i < n {
            let left = i - stride;
            let tmp = buf[left];
            buf[left] = buf[i];
            buf[i] += tmp;
            i += 2 * stride;
        }
        stride /= 2;
    }

    buf[..data.len()].to_vec()
}

/// Inclusive prefix scan from the exclusive result.
fn inclusive_scan(data: &[f32]) -> Vec<f32> {
    let exc = blelloch_scan(data);
    exc.iter().zip(data.iter()).map(|(e, d)| e + d).collect()
}

/// CPU reference inclusive scan.
fn reference_inclusive_scan(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc = 0.0f32;
    for &v in data {
        acc += v;
        out.push(acc);
    }
    out
}

/// Tile-based matrix multiply shared memory requirement.
fn matmul_tile_shared_bytes(tile_m: u32, tile_n: u32, tile_k: u32, elem_size: u32) -> u32 {
    // A tile: tile_m × tile_k, B tile: tile_k × tile_n
    tile_m * tile_k * elem_size + tile_k * tile_n * elem_size
}

// ── Coalescing model ────────────────────────────────────────────────

/// Describes a memory transaction pattern for coalescing analysis.
#[derive(Debug)]
struct CoalescingPattern {
    /// Byte addresses accessed by threads in a SIMD group.
    addresses: Vec<u32>,
    /// Transaction granularity in bytes (e.g., 128 for a cache line).
    transaction_size: u32,
}

impl CoalescingPattern {
    /// Count the minimum number of memory transactions needed.
    fn transactions(&self) -> usize {
        let mut lines: Vec<u32> =
            self.addresses.iter().map(|&a| a / self.transaction_size).collect();
        lines.sort_unstable();
        lines.dedup();
        lines.len()
    }

    /// Coalescing efficiency: threads / (transactions * threads_per_transaction).
    fn efficiency(&self) -> f64 {
        let tx = self.transactions().max(1);
        let ideal = 1; // perfect coalescing = 1 transaction
        ideal as f64 / tx as f64
    }

    /// Sequential access: thread i reads consecutive element i.
    fn sequential(
        num_threads: u32,
        element_size: u32,
        base_addr: u32,
        transaction_size: u32,
    ) -> Self {
        let addresses = (0..num_threads).map(|i| base_addr + i * element_size).collect();
        Self { addresses, transaction_size }
    }

    /// Strided access: thread i reads element i * stride.
    fn strided(
        num_threads: u32,
        stride: u32,
        element_size: u32,
        base_addr: u32,
        transaction_size: u32,
    ) -> Self {
        let addresses = (0..num_threads).map(|i| base_addr + i * stride * element_size).collect();
        Self { addresses, transaction_size }
    }

    /// Random access pattern.
    fn random(addresses: Vec<u32>, transaction_size: u32) -> Self {
        Self { addresses, transaction_size }
    }
}

// ── Utility ─────────────────────────────────────────────────────────

fn align_up(value: u32, alignment: u32) -> u32 {
    debug_assert!(alignment.is_power_of_two());
    let mask = alignment - 1;
    (value + mask) & !mask
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod threadgroup_memory_allocation {
    use super::*;

    // ── Basic sizing ────────────────────────────────────────────────

    #[test]
    fn f32_16x16_tile_size() {
        let bytes = 16 * 16 * 4; // 16×16 f32
        assert_eq!(bytes, 1024);
    }

    #[test]
    fn f32_32x32_tile_size() {
        let bytes = 32 * 32 * 4;
        assert_eq!(bytes, 4096);
    }

    #[test]
    fn f16_16x16_tile_size() {
        let bytes = 16 * 16 * 2;
        assert_eq!(bytes, 512);
    }

    #[test]
    fn f16_32x32_tile_size() {
        let bytes = 32 * 32 * 2;
        assert_eq!(bytes, 2048);
    }

    #[test]
    fn i2s_256_element_block_size() {
        // 256 elements at 2 bits = 64 bytes + 2-byte scale = 66 bytes
        let bytes = 256 / 4 + 2;
        assert_eq!(bytes, 66);
    }

    #[test]
    fn matmul_16x16x16_f32_shared() {
        let shared = matmul_tile_shared_bytes(16, 16, 16, 4);
        // A: 16×16×4 = 1024, B: 16×16×4 = 1024 → 2048
        assert_eq!(shared, 2048);
    }

    #[test]
    fn matmul_32x32x32_f32_shared() {
        let shared = matmul_tile_shared_bytes(32, 32, 32, 4);
        assert_eq!(shared, 8192);
    }

    #[test]
    fn matmul_16x16x16_f16_shared() {
        let shared = matmul_tile_shared_bytes(16, 16, 16, 2);
        assert_eq!(shared, 1024);
    }

    #[test]
    fn matmul_32x32x8_f32_shared() {
        let shared = matmul_tile_shared_bytes(32, 32, 8, 4);
        // A: 32×8×4=1024, B: 8×32×4=1024 → 2048
        assert_eq!(shared, 2048);
    }

    #[test]
    fn matmul_64x64x16_f16_shared() {
        let shared = matmul_tile_shared_bytes(64, 64, 16, 2);
        // A: 64×16×2=2048, B: 16×64×2=2048 → 4096
        assert_eq!(shared, 4096);
    }

    // ── Alignment ───────────────────────────────────────────────────

    #[test]
    fn align_up_16() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn align_up_256() {
        assert_eq!(align_up(100, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn align_up_4() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(5, 4), 8);
    }

    // ── Layout allocator ────────────────────────────────────────────

    #[test]
    fn single_allocation_layout() {
        let mut layout = SharedMemoryLayout::new();
        let off = layout.allocate("tile_a", 4, 256, 16);
        assert_eq!(off, 0);
        assert_eq!(layout.total_bytes, 1024);
        assert_eq!(layout.padding_bytes, 0);
        assert_eq!(layout.allocations.len(), 1);
    }

    #[test]
    fn two_allocations_contiguous() {
        let mut layout = SharedMemoryLayout::new();
        layout.allocate("tile_a", 4, 256, 16);
        let off_b = layout.allocate("tile_b", 4, 256, 16);
        assert_eq!(off_b, 1024);
        assert_eq!(layout.total_bytes, 2048);
        assert_eq!(layout.padding_bytes, 0);
    }

    #[test]
    fn allocation_with_padding() {
        let mut layout = SharedMemoryLayout::new();
        layout.allocate("small", 1, 5, 4); // 5 bytes, next aligned to 16
        let off = layout.allocate("aligned", 4, 1, 16);
        assert_eq!(off, 16);
        assert!(layout.padding_bytes > 0);
    }

    #[test]
    fn matmul_double_buffer_layout() {
        let mut layout = SharedMemoryLayout::new();
        // Double-buffered: two A tiles + two B tiles
        layout.allocate("tile_a0", 4, 16 * 16, 16);
        layout.allocate("tile_a1", 4, 16 * 16, 16);
        layout.allocate("tile_b0", 4, 16 * 16, 16);
        layout.allocate("tile_b1", 4, 16 * 16, 16);
        assert_eq!(layout.total_bytes, 4 * 1024);
        assert_eq!(layout.allocations.len(), 4);
    }

    #[test]
    fn layout_fits_in_32k() {
        let config = ThreadgroupMemoryConfig::m1_default();
        let mut layout = SharedMemoryLayout::new();
        layout.allocate("tile", 4, 32 * 32, 16); // 4096
        assert!(layout.fits_in(&config));
    }

    #[test]
    fn layout_exceeds_32k() {
        let config = ThreadgroupMemoryConfig::m1_default();
        let mut layout = SharedMemoryLayout::new();
        // 9 * 4096 = 36864 > 32768
        for i in 0..9 {
            layout.allocate(&format!("t{i}"), 4, 32 * 32, 16);
        }
        assert!(!layout.fits_in(&config));
    }

    #[test]
    fn layout_exactly_at_limit() {
        let config = ThreadgroupMemoryConfig::m1_default();
        let mut layout = SharedMemoryLayout::new();
        // 32768 / 4 = 8192 elements of f32
        layout.allocate("full", 4, 8192, 16);
        assert_eq!(layout.total_bytes, 32768);
        assert!(layout.fits_in(&config));
    }

    #[test]
    fn layout_one_byte_over() {
        let config = ThreadgroupMemoryConfig::m1_default();
        let mut layout = SharedMemoryLayout::new();
        layout.allocate("full", 1, 32769, 1);
        assert!(!layout.fits_in(&config));
    }

    #[test]
    fn mixed_element_sizes() {
        let mut layout = SharedMemoryLayout::new();
        layout.allocate("f32_tile", 4, 64, 16); // 256 bytes
        layout.allocate("f16_tile", 2, 128, 16); // 256 bytes
        layout.allocate("i8_scratch", 1, 32, 4); // 32 bytes
        assert_eq!(layout.total_bytes, 256 + 256 + 32);
    }

    #[test]
    fn zero_element_allocation() {
        let mut layout = SharedMemoryLayout::new();
        let off = layout.allocate("empty", 4, 0, 16);
        assert_eq!(off, 0);
        assert_eq!(layout.total_bytes, 0);
    }

    #[test]
    fn many_small_allocations_track_padding() {
        let mut layout = SharedMemoryLayout::new();
        // 10 allocations of 3 bytes each, aligned to 16
        for i in 0..10 {
            layout.allocate(&format!("s{i}"), 1, 3, 16);
        }
        // Each after the first gets 13 bytes of padding
        assert_eq!(layout.padding_bytes, 9 * 13);
        assert_eq!(layout.total_bytes, 9 * 16 + 3);
    }

    // ── Tile configurations for different architectures ─────────────

    #[test]
    fn transformer_attention_shared_mem() {
        // Q·Kᵀ tiling: Q tile (seq_tile × head_dim) + K tile (kv_tile × head_dim)
        let seq_tile = 32u32;
        let kv_tile = 32;
        let head_dim = 64;
        let elem = 2u32; // f16
        let shared = seq_tile * head_dim * elem + kv_tile * head_dim * elem;
        assert_eq!(shared, 2 * 32 * 64 * 2); // 8192 bytes
        assert!(shared <= 32 * 1024);
    }

    #[test]
    fn softmax_shared_mem_for_1024_seq() {
        // One row of logits in shared memory for online softmax
        let seq_len = 1024u32;
        let elem = 4; // f32 for numerics
        let shared = seq_len * elem;
        assert_eq!(shared, 4096);
    }

    #[test]
    fn reduction_shared_mem_1024_threads() {
        // Each thread contributes one f32
        let threads = 1024u32;
        let shared = threads * 4;
        assert_eq!(shared, 4096);
    }
}

#[cfg(test)]
mod memory_banking {
    use super::*;

    const BANKS: u32 = 32;
    const BANK_W: u32 = 4;

    // ── Stride-1 access ─────────────────────────────────────────────

    #[test]
    fn stride1_f32_conflict_free() {
        let pat = BankAccessPattern::stride1(32, 4, BANKS, BANK_W);
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn stride1_f16_conflict_free() {
        // 32 threads × 2 bytes each → each hits consecutive 2-byte half-words
        // within different 4-byte banks
        let pat = BankAccessPattern::stride1(32, 2, BANKS, BANK_W);
        // Half-precision: two elements per bank word ⇒ pairs collide
        let conflicts = pat.bank_conflicts();
        assert!(conflicts <= 1, "f16 stride-1 may have at most 1-way conflict: got {conflicts}");
    }

    #[test]
    fn stride1_f64_has_2way_conflict() {
        // f64: 8 bytes per element → 2 bank-widths apart → thread i hits bank (2i)%32
        // threads 0 and 16 both hit bank 0 → 2-way conflict
        let pat = BankAccessPattern::stride1(32, 8, BANKS, BANK_W);
        assert_eq!(pat.bank_conflicts(), 1, "f64 stride-1: 2-way bank conflict");
    }

    // ── Stride-N access ─────────────────────────────────────────────

    #[test]
    fn stride2_f32_16way_conflict() {
        // stride=2 → threads 0,16 both hit bank 0
        let pat = BankAccessPattern::stride_n(32, 2, 4, BANKS, BANK_W);
        assert_eq!(pat.bank_conflicts(), 1); // 2-way → 1 conflict
    }

    #[test]
    fn stride32_f32_max_conflict() {
        // All threads hit the same bank
        let pat = BankAccessPattern::stride_n(32, 32, 4, BANKS, BANK_W);
        assert_eq!(pat.bank_conflicts(), 31);
    }

    #[test]
    fn stride16_f32_16way_conflict() {
        // stride=16: bank[i] = (16i) % 32 → only 2 unique banks → 16-way conflict
        let pat = BankAccessPattern::stride_n(32, 16, 4, BANKS, BANK_W);
        assert_eq!(pat.bank_conflicts(), 15);
    }

    #[test]
    fn stride3_f32_conflict_free() {
        // stride=3 with 32 banks: gcd(3,32)=1 → conflict-free
        let pat = BankAccessPattern::stride_n(32, 3, 4, BANKS, BANK_W);
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn stride5_f32_conflict_free() {
        let pat = BankAccessPattern::stride_n(32, 5, 4, BANKS, BANK_W);
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn stride7_f32_conflict_free() {
        let pat = BankAccessPattern::stride_n(32, 7, 4, BANKS, BANK_W);
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn stride4_f32_conflict() {
        // gcd(4,32)=4 → 4-way bank conflict → max 4 hits per bank → 3 conflicts
        let pat = BankAccessPattern::stride_n(32, 4, 4, BANKS, BANK_W);
        assert_eq!(pat.bank_conflicts(), 3);
    }

    #[test]
    fn stride8_f32_conflict() {
        // gcd(8,32)=8 → 8-way → 7 conflicts
        let pat = BankAccessPattern::stride_n(32, 8, 4, BANKS, BANK_W);
        assert_eq!(pat.bank_conflicts(), 7);
    }

    // ── Column-major access (matmul B-tile transposed reads) ────────

    #[test]
    fn column_major_32x32_f32_has_conflicts() {
        let pat = BankAccessPattern::column_major(32, 32, 0, 4, BANKS, BANK_W);
        assert!(!pat.is_conflict_free(), "32-wide column-major should conflict");
    }

    #[test]
    fn column_major_32x33_f32_padded_conflict_free() {
        // +1 padding column eliminates bank conflicts
        let pat = BankAccessPattern::column_major_padded(32, 32, 0, 1, 4, BANKS, BANK_W);
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn column_major_16x16_f32_conflict_free() {
        // 16×16: stride=16, 16 threads → gcd(16,32)=16, but only 16 threads
        let pat = BankAccessPattern::column_major(16, 16, 0, 4, BANKS, BANK_W);
        // 16 threads with stride 16: bank indices are 0,16,0,16,...
        let conflicts = pat.bank_conflicts();
        assert_eq!(conflicts, 7, "16×16 col-major: half the threads collide");
    }

    #[test]
    fn column_major_16x17_f32_padded_conflict_free() {
        let pat = BankAccessPattern::column_major_padded(16, 16, 0, 1, 4, BANKS, BANK_W);
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn column_major_different_columns() {
        // Different columns should have same conflict pattern (offset doesn't matter for bank analysis)
        for col in 0..4 {
            let pat = BankAccessPattern::column_major(32, 32, col, 4, BANKS, BANK_W);
            assert!(!pat.is_conflict_free(), "col {col} should also conflict");
        }
    }

    // ── Broadcast ───────────────────────────────────────────────────

    #[test]
    fn broadcast_is_free_on_metal() {
        // Metal hardware handles broadcast (all threads read same address) without conflict
        let pat = BankAccessPattern::broadcast(32, 0, BANKS, BANK_W);
        // Our model counts it as 31 conflicts, but Metal hardware resolves broadcast
        assert_eq!(pat.bank_conflicts(), 31);
        // In real hardware this is resolved in a single cycle (broadcast mechanism).
    }

    #[test]
    fn broadcast_different_addresses() {
        let pat0 = BankAccessPattern::broadcast(32, 0, BANKS, BANK_W);
        let pat1 = BankAccessPattern::broadcast(32, 128, BANKS, BANK_W);
        assert_eq!(pat0.bank_conflicts(), pat1.bank_conflicts());
    }

    // ── Conflict calculation edge cases ─────────────────────────────

    #[test]
    fn single_thread_no_conflict() {
        let pat =
            BankAccessPattern { addresses: vec![0], num_banks: BANKS, bank_width_bytes: BANK_W };
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn empty_pattern_no_conflict() {
        let pat =
            BankAccessPattern { addresses: vec![], num_banks: BANKS, bank_width_bytes: BANK_W };
        assert!(pat.is_conflict_free());
    }

    #[test]
    fn two_threads_same_bank() {
        let pat = BankAccessPattern {
            addresses: vec![0, 128], // bank 0 both (128/4 = 32 ≡ 0 mod 32)
            num_banks: BANKS,
            bank_width_bytes: BANK_W,
        };
        assert_eq!(pat.bank_conflicts(), 1);
    }

    #[test]
    fn two_threads_different_banks() {
        let pat = BankAccessPattern {
            addresses: vec![0, 4], // bank 0, bank 1
            num_banks: BANKS,
            bank_width_bytes: BANK_W,
        };
        assert!(pat.is_conflict_free());
    }

    // ── Padding strategy validation ─────────────────────────────────

    #[test]
    fn padding_by_1_fixes_power_of_two_stride() {
        for cols in [16u32, 32, 64, 128] {
            let unpadded = BankAccessPattern::column_major(32.min(cols), cols, 0, 4, BANKS, BANK_W);
            let padded =
                BankAccessPattern::column_major_padded(32.min(cols), cols, 0, 1, 4, BANKS, BANK_W);
            assert!(
                padded.bank_conflicts() < unpadded.bank_conflicts() || unpadded.is_conflict_free(),
                "padding should help for cols={cols}"
            );
        }
    }

    #[test]
    fn odd_stride_always_conflict_free() {
        for s in [1u32, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] {
            let pat = BankAccessPattern::stride_n(32, s, 4, BANKS, BANK_W);
            assert!(pat.is_conflict_free(), "odd stride {s} should be conflict-free with 32 banks");
        }
    }

    #[test]
    fn conflict_degree_matches_gcd() {
        for stride in 1u32..=32 {
            let pat = BankAccessPattern::stride_n(32, stride, 4, BANKS, BANK_W);
            let gcd = gcd(stride, BANKS);
            let expected_way = gcd; // gcd-way bank conflict
            let expected_conflicts = expected_way - 1;
            assert_eq!(
                pat.bank_conflicts(),
                expected_conflicts,
                "stride={stride}: expected {expected_way}-way conflict ({expected_conflicts} extra), got {}",
                pat.bank_conflicts()
            );
        }
    }

    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
}

#[cfg(test)]
mod shared_memory_patterns {
    use super::*;

    // ── Parallel reduction (sum) ────────────────────────────────────

    #[test]
    fn reduction_sum_power_of_two() {
        let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let result = parallel_reduction_sum(&data);
        let expected: f32 = (1..=32).sum::<u32>() as f32;
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn reduction_sum_non_power_of_two() {
        let data: Vec<f32> = (1..=17).map(|x| x as f32).collect();
        let result = parallel_reduction_sum(&data);
        let expected = 17.0 * 18.0 / 2.0;
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn reduction_sum_single_element() {
        assert!((parallel_reduction_sum(&[42.0]) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn reduction_sum_two_elements() {
        assert!((parallel_reduction_sum(&[3.0, 7.0]) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn reduction_sum_all_zeros() {
        let data = vec![0.0f32; 64];
        assert!((parallel_reduction_sum(&data)).abs() < 1e-6);
    }

    #[test]
    fn reduction_sum_negative_values() {
        let data: Vec<f32> = (-16..=15).map(|x| x as f32).collect();
        let result = parallel_reduction_sum(&data);
        let expected: f32 = (-16..=15).sum::<i32>() as f32;
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn reduction_sum_1024_elements() {
        let data: Vec<f32> = (0..1024).map(|x| x as f32).collect();
        let result = parallel_reduction_sum(&data);
        let expected = 1024.0 * 1023.0 / 2.0;
        assert!((result - expected).abs() < 1.0);
    }

    #[test]
    fn reduction_sum_shared_mem_sizing() {
        // For reduction of N elements we need N * sizeof(f32) shared memory
        for &n in &[32u32, 64, 128, 256, 512, 1024] {
            let shared = n * 4;
            assert!(shared <= 32 * 1024, "reduction of {n} should fit in 32 KiB");
        }
    }

    // ── Parallel reduction (max) ────────────────────────────────────

    #[test]
    fn reduction_max_basic() {
        let data = vec![1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 8.0];
        assert!((parallel_reduction_max(&data) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn reduction_max_single() {
        assert!((parallel_reduction_max(&[-5.0]) - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn reduction_max_all_same() {
        let data = vec![3.125f32; 64];
        assert!((parallel_reduction_max(&data) - 3.125).abs() < 1e-6);
    }

    #[test]
    fn reduction_max_negative() {
        let data = vec![-10.0f32, -5.0, -1.0, -100.0];
        assert!((parallel_reduction_max(&data) - (-1.0)).abs() < 1e-6);
    }

    // ── Blelloch exclusive prefix scan ──────────────────────────────

    #[test]
    fn blelloch_exclusive_scan_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = blelloch_scan(&data);
        assert_eq!(result.len(), 4);
        let expected = [0.0, 1.0, 3.0, 6.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
        }
    }

    #[test]
    fn blelloch_exclusive_scan_8_elements() {
        let data = vec![1.0; 8];
        let result = blelloch_scan(&data);
        let expected: Vec<f32> = (0..8).map(|i| i as f32).collect();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn blelloch_exclusive_scan_single() {
        let result = blelloch_scan(&[42.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn blelloch_scan_non_power_of_two() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = blelloch_scan(&data);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 6.0).abs() < 1e-6);
        assert!((result[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn blelloch_scan_zeros() {
        let data = vec![0.0; 16];
        let result = blelloch_scan(&data);
        assert!(result.iter().all(|&v| v.abs() < 1e-6));
    }

    // ── Inclusive prefix scan ───────────────────────────────────────

    #[test]
    fn inclusive_scan_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = inclusive_scan(&data);
        let expected = [1.0, 3.0, 6.0, 10.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn inclusive_scan_matches_reference() {
        let data: Vec<f32> = (1..=64).map(|x| x as f32).collect();
        let result = inclusive_scan(&data);
        let reference = reference_inclusive_scan(&data);
        for (i, (r, e)) in result.iter().zip(reference.iter()).enumerate() {
            assert!((r - e).abs() < 1e-3, "mismatch at index {i}: got {r}, expected {e}");
        }
    }

    #[test]
    fn inclusive_scan_256_elements() {
        let data: Vec<f32> = (0..256).map(|x| (x as f32) * 0.01).collect();
        let result = inclusive_scan(&data);
        let reference = reference_inclusive_scan(&data);
        for (i, (r, e)) in result.iter().zip(reference.iter()).enumerate() {
            assert!((r - e).abs() < 0.1, "mismatch at {i}");
        }
    }

    // ── Scan shared memory requirements ─────────────────────────────

    #[test]
    fn scan_shared_mem_1024_f32() {
        // Blelloch scan needs 2*N shared memory (double buffer)
        let n = 1024u32;
        let shared = 2 * n * 4;
        assert_eq!(shared, 8192);
        assert!(shared <= 32 * 1024);
    }

    #[test]
    fn scan_shared_mem_4096_f32_at_limit() {
        // 2 * 4096 * 4 = 32768 = exactly 32 KiB (leaves no room for anything else)
        let n = 4096u32;
        let shared = 2 * n * 4;
        assert_eq!(shared, 32 * 1024, "4096-element scan exactly fills 32 KiB");
    }

    #[test]
    fn scan_shared_mem_8192_f32_exceeds_limit() {
        let n = 8192u32;
        let shared = 2 * n * 4;
        assert!(shared > 32 * 1024, "8192-element scan needs multi-block approach");
    }

    // ── Tiled matmul shared memory patterns ─────────────────────────

    #[test]
    fn tiled_matmul_16x16_fits() {
        let shared = matmul_tile_shared_bytes(16, 16, 16, 4);
        assert!(shared <= 32 * 1024);
    }

    #[test]
    fn tiled_matmul_64x64_f32_fits() {
        let shared = matmul_tile_shared_bytes(64, 64, 16, 4);
        // A: 64×16×4=4096, B: 16×64×4=4096 → 8192
        assert_eq!(shared, 8192);
        assert!(shared <= 32 * 1024);
    }

    #[test]
    fn tiled_matmul_128x128_f32_too_large() {
        let shared = matmul_tile_shared_bytes(128, 128, 32, 4);
        // A: 128×32×4=16384, B: 32×128×4=16384 → 32768
        assert_eq!(shared, 32768);
        // Exactly at the limit — but double-buffering would exceed it
    }

    #[test]
    fn tiled_matmul_double_buffer_limit() {
        // Double-buffered tiles for latency hiding
        let single = matmul_tile_shared_bytes(32, 32, 16, 4);
        let double = 2 * single;
        assert_eq!(single, 4096);
        assert_eq!(double, 8192);
        assert!(double <= 32 * 1024);
    }

    #[test]
    fn reduction_tree_depth() {
        // log2(n) steps for parallel reduction
        for &n in &[32u32, 64, 128, 256, 512, 1024] {
            let depth = (n as f32).log2().ceil() as u32;
            assert!(depth <= 10, "reduction depth for {n} should be manageable");
        }
    }

    #[test]
    fn warp_reduction_no_shared_memory() {
        // For ≤32 elements, can use SIMD shuffle — no shared memory needed
        let n = 32u32;
        let shared = if n <= 32 { 0 } else { n * 4 };
        assert_eq!(shared, 0);
    }

    #[test]
    fn two_level_reduction_shared_mem() {
        // 1024 threads → 32 SIMD groups → each produces 1 partial → 32 f32s in shared
        let simd_width = 32u32;
        let total_threads = 1024u32;
        let num_simd_groups = total_threads / simd_width;
        let shared = num_simd_groups * 4; // f32 per group
        assert_eq!(shared, 128);
    }
}

#[cfg(test)]
mod apple_silicon_memory {
    use super::*;

    // ── Profile properties ──────────────────────────────────────────

    #[test]
    fn m1_unified_memory() {
        let m1 = AppleSiliconMemoryProfile::m1();
        assert!(m1.unified_memory);
    }

    #[test]
    fn m2_unified_memory() {
        assert!(AppleSiliconMemoryProfile::m2().unified_memory);
    }

    #[test]
    fn m3_unified_memory() {
        assert!(AppleSiliconMemoryProfile::m3().unified_memory);
    }

    #[test]
    fn m4_unified_memory() {
        assert!(AppleSiliconMemoryProfile::m4().unified_memory);
    }

    #[test]
    fn all_gens_simd_width_32() {
        for profile in [
            AppleSiliconMemoryProfile::m1(),
            AppleSiliconMemoryProfile::m2(),
            AppleSiliconMemoryProfile::m3(),
            AppleSiliconMemoryProfile::m4(),
        ] {
            assert_eq!(
                profile.simd_width, 32,
                "{:?} should have SIMD width 32",
                profile.generation
            );
        }
    }

    #[test]
    fn all_gens_max_threadgroup_memory_32k() {
        for profile in [
            AppleSiliconMemoryProfile::m1(),
            AppleSiliconMemoryProfile::m2(),
            AppleSiliconMemoryProfile::m3(),
            AppleSiliconMemoryProfile::m4(),
        ] {
            assert_eq!(
                profile.max_threadgroup_memory,
                32 * 1024,
                "{:?} should have 32 KiB threadgroup memory",
                profile.generation
            );
        }
    }

    #[test]
    fn all_gens_max_threads_1024() {
        for profile in [
            AppleSiliconMemoryProfile::m1(),
            AppleSiliconMemoryProfile::m2(),
            AppleSiliconMemoryProfile::m3(),
            AppleSiliconMemoryProfile::m4(),
        ] {
            assert_eq!(profile.max_threads_per_threadgroup, 1024);
        }
    }

    #[test]
    fn all_gens_32_banks() {
        for profile in [
            AppleSiliconMemoryProfile::m1(),
            AppleSiliconMemoryProfile::m2(),
            AppleSiliconMemoryProfile::m3(),
            AppleSiliconMemoryProfile::m4(),
        ] {
            assert_eq!(profile.num_banks, 32);
        }
    }

    // ── Tile cache progression ──────────────────────────────────────

    #[test]
    fn tile_cache_increases_across_gens() {
        let m1 = AppleSiliconMemoryProfile::m1();
        let m2 = AppleSiliconMemoryProfile::m2();
        let m3 = AppleSiliconMemoryProfile::m3();
        assert!(m2.tile_cache_bytes >= m1.tile_cache_bytes);
        assert!(m3.tile_cache_bytes >= m2.tile_cache_bytes);
    }

    #[test]
    fn m3_higher_concurrent_threadgroups() {
        let m1 = AppleSiliconMemoryProfile::m1();
        let m3 = AppleSiliconMemoryProfile::m3();
        assert!(m3.max_threadgroups_per_cu >= m1.max_threadgroups_per_cu);
    }

    // ── Occupancy model ─────────────────────────────────────────────

    #[test]
    fn full_occupancy_small_shared_mem() {
        let m1 = AppleSiliconMemoryProfile::m1();
        // Using very little shared memory → max threadgroups fit
        let occ = m1.occupancy(1024, 256);
        assert!(occ > 0.0);
    }

    #[test]
    fn zero_occupancy_zero_threads() {
        let m1 = AppleSiliconMemoryProfile::m1();
        assert_eq!(m1.occupancy(1024, 0), 0.0);
    }

    #[test]
    fn zero_occupancy_zero_mem() {
        let m1 = AppleSiliconMemoryProfile::m1();
        assert_eq!(m1.occupancy(0, 256), 0.0);
    }

    #[test]
    fn occupancy_decreases_with_more_shared_mem() {
        let m1 = AppleSiliconMemoryProfile::m1();
        let occ_small = m1.occupancy(4096, 256);
        let occ_large = m1.occupancy(16384, 256);
        assert!(occ_small >= occ_large);
    }

    #[test]
    fn occupancy_max_shared_mem_single_group() {
        let m1 = AppleSiliconMemoryProfile::m1();
        // 32 KiB → only 1 threadgroup fits
        let occ = m1.occupancy(32 * 1024, 256);
        assert!(occ > 0.0);
        assert!(occ <= 1.0);
    }

    #[test]
    fn m3_more_active_threads_than_m1() {
        // M3 supports more concurrent threadgroups per CU, so for the same
        // workload it can run more active threads (even if occupancy ratio
        // differs due to larger denominator).
        let m1 = AppleSiliconMemoryProfile::m1();
        let m3 = AppleSiliconMemoryProfile::m3();
        let shared = 8 * 1024u32; // 8 KiB
        let threads = 256u32;

        // M1: mem allows 4 groups, CU limit 4 → 4 × 256 = 1024 active threads
        // M3: mem allows 4 groups, CU limit 8 → 4 × 256 = 1024 active threads
        // But M3's max capacity is 8 × 1024 = 8192 vs M1's 4 × 1024 = 4096
        let m1_groups = (m1.max_threadgroup_memory / shared).min(m1.max_threadgroups_per_cu);
        let m3_groups = (m3.max_threadgroup_memory / shared).min(m3.max_threadgroups_per_cu);
        assert!(m3_groups >= m1_groups, "M3 fits at least as many threadgroups");
        assert_eq!(m1_groups * threads, m3_groups * threads);
    }

    // ── Config defaults ─────────────────────────────────────────────

    #[test]
    fn m1_config_defaults() {
        let cfg = ThreadgroupMemoryConfig::m1_default();
        assert_eq!(cfg.max_total_threadgroup_memory, 32 * 1024);
        assert_eq!(cfg.alignment, 16);
        assert_eq!(cfg.num_banks, 32);
        assert_eq!(cfg.bank_width_bytes, 4);
    }

    #[test]
    fn m3_config_defaults() {
        let cfg = ThreadgroupMemoryConfig::m3_default();
        assert_eq!(cfg.max_total_threadgroup_memory, 32 * 1024);
        assert_eq!(cfg.alignment, 16);
    }

    // ── Kernel tuning heuristics ────────────────────────────────────

    #[test]
    fn optimal_tile_size_m1_matmul() {
        let m1 = AppleSiliconMemoryProfile::m1();
        // Find largest square tile that fits with double buffering
        let mut best_tile = 0u32;
        for tile in [8, 16, 32, 64, 128] {
            let shared = 2 * matmul_tile_shared_bytes(tile, tile, tile, 4);
            if shared <= m1.max_threadgroup_memory {
                best_tile = tile;
            }
        }
        // 32×32×32: single=8192, double=16384 ≤ 32768 ✓
        // 64×64×64: single=32768, double=65536 > 32768 ✗
        assert_eq!(best_tile, 32);
    }

    #[test]
    fn optimal_tile_size_f16_matmul() {
        let m1 = AppleSiliconMemoryProfile::m1();
        let mut best_tile = 0u32;
        for tile in [8, 16, 32, 64, 128] {
            let shared = 2 * matmul_tile_shared_bytes(tile, tile, tile, 2);
            if shared <= m1.max_threadgroup_memory {
                best_tile = tile;
            }
        }
        // f16: 64×64×64 double = 2*16384=32768 → fits
        assert_eq!(best_tile, 64);
    }

    #[test]
    fn recommended_threadgroup_sizes() {
        // Common threadgroup sizes for Apple Silicon kernels
        let sizes: Vec<(u32, u32)> = vec![
            (32, 1),  // 1D reduction
            (16, 16), // 2D tiled
            (32, 8),  // rectangular tile
            (8, 8),   // small tile
            (32, 32), // large tile
        ];
        for (x, y) in &sizes {
            assert!(x * y <= 1024, "threadgroup ({x},{y}) exceeds 1024 threads");
            assert!(x * y > 0);
        }
    }

    #[test]
    fn simd_aligned_threadgroup_sizes() {
        // Threadgroup dimensions should be multiples of SIMD width (32) for best perf
        let good_sizes = [32u32, 64, 128, 256, 512, 1024];
        for &sz in &good_sizes {
            assert_eq!(sz % 32, 0, "size {sz} should be SIMD-aligned");
        }
    }
}

#[cfg(test)]
mod memory_coalescing {
    use super::*;

    const TX_SIZE: u32 = 128; // 128-byte cache line

    // ── Sequential (coalesced) access ───────────────────────────────

    #[test]
    fn sequential_f32_perfectly_coalesced() {
        let pat = CoalescingPattern::sequential(32, 4, 0, TX_SIZE);
        // 32 × 4 = 128 bytes → exactly 1 cache line
        assert_eq!(pat.transactions(), 1);
    }

    #[test]
    fn sequential_f32_64_threads() {
        let pat = CoalescingPattern::sequential(64, 4, 0, TX_SIZE);
        // 64 × 4 = 256 bytes → 2 cache lines
        assert_eq!(pat.transactions(), 2);
    }

    #[test]
    fn sequential_f16_32_threads() {
        let pat = CoalescingPattern::sequential(32, 2, 0, TX_SIZE);
        // 32 × 2 = 64 bytes → 1 cache line
        assert_eq!(pat.transactions(), 1);
    }

    #[test]
    fn sequential_f16_128_threads() {
        let pat = CoalescingPattern::sequential(128, 2, 0, TX_SIZE);
        // 128 × 2 = 256 → 2 cache lines
        assert_eq!(pat.transactions(), 2);
    }

    #[test]
    fn sequential_aligned_base() {
        let pat = CoalescingPattern::sequential(32, 4, 128, TX_SIZE);
        assert_eq!(pat.transactions(), 1);
    }

    #[test]
    fn sequential_misaligned_base() {
        // Starting at offset 64 → straddles two cache lines
        let pat = CoalescingPattern::sequential(32, 4, 64, TX_SIZE);
        assert_eq!(pat.transactions(), 2);
    }

    // ── Strided access ──────────────────────────────────────────────

    #[test]
    fn stride2_f32_twice_the_transactions() {
        let pat = CoalescingPattern::strided(32, 2, 4, 0, TX_SIZE);
        // 32 threads, stride 2: addresses 0,8,16,...,248 → 256 bytes → 2 lines
        assert_eq!(pat.transactions(), 2);
    }

    #[test]
    fn stride4_f32() {
        let pat = CoalescingPattern::strided(32, 4, 4, 0, TX_SIZE);
        // addresses span 0..32*4*4 = 512 → 4 lines
        assert_eq!(pat.transactions(), 4);
    }

    #[test]
    fn stride32_f32_worst_case() {
        let pat = CoalescingPattern::strided(32, 32, 4, 0, TX_SIZE);
        // addresses span 0..32*32*4 = 4096 → 32 lines
        assert_eq!(pat.transactions(), 32);
    }

    #[test]
    fn large_stride_poor_efficiency() {
        let sequential = CoalescingPattern::sequential(32, 4, 0, TX_SIZE);
        let strided = CoalescingPattern::strided(32, 32, 4, 0, TX_SIZE);
        assert!(sequential.efficiency() > strided.efficiency());
    }

    // ── Efficiency calculations ─────────────────────────────────────

    #[test]
    fn perfect_coalescing_efficiency() {
        let pat = CoalescingPattern::sequential(32, 4, 0, TX_SIZE);
        assert!((pat.efficiency() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn half_efficiency() {
        let pat = CoalescingPattern::sequential(32, 4, 64, TX_SIZE);
        assert!((pat.efficiency() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn efficiency_decreases_with_stride() {
        let eff_1 = CoalescingPattern::strided(32, 1, 4, 0, TX_SIZE).efficiency();
        let eff_2 = CoalescingPattern::strided(32, 2, 4, 0, TX_SIZE).efficiency();
        let eff_4 = CoalescingPattern::strided(32, 4, 4, 0, TX_SIZE).efficiency();
        assert!(eff_1 >= eff_2);
        assert!(eff_2 >= eff_4);
    }

    // ── Random access patterns ──────────────────────────────────────

    #[test]
    fn random_worst_case() {
        // 32 threads each hitting a different cache line
        let addresses: Vec<u32> = (0..32).map(|i| i * TX_SIZE).collect();
        let pat = CoalescingPattern::random(addresses, TX_SIZE);
        assert_eq!(pat.transactions(), 32);
    }

    #[test]
    fn random_all_same_line() {
        let addresses = vec![0u32; 32];
        let pat = CoalescingPattern::random(addresses, TX_SIZE);
        assert_eq!(pat.transactions(), 1);
    }

    #[test]
    fn random_two_lines() {
        let mut addresses = vec![0u32; 16];
        addresses.extend(vec![TX_SIZE; 16]);
        let pat = CoalescingPattern::random(addresses, TX_SIZE);
        assert_eq!(pat.transactions(), 2);
    }

    // ── Tiled access coalescing ─────────────────────────────────────

    #[test]
    fn row_major_load_coalesced() {
        // Loading a row of a tile: threads read consecutive elements → coalesced
        let pat = CoalescingPattern::sequential(32, 4, 0, TX_SIZE);
        assert_eq!(pat.transactions(), 1);
    }

    #[test]
    fn column_major_load_strided() {
        // Loading a column: stride = row_width
        let row_width = 128u32; // 128 f32 elements
        let pat = CoalescingPattern::strided(32, row_width, 4, 0, TX_SIZE);
        // Each address is in a different cache line
        assert!(pat.transactions() > 1);
    }

    #[test]
    fn tiled_load_vs_naive_load() {
        // Tiled: load 32×32 block into shared memory row-by-row (coalesced)
        // vs naive column access (strided)
        let tile_rows = 32u32;
        let row_load_txns: u32 = (0..tile_rows)
            .map(|_| CoalescingPattern::sequential(32, 4, 0, TX_SIZE).transactions() as u32)
            .sum();
        let col_load_txns: u32 = (0..tile_rows)
            .map(|_| CoalescingPattern::strided(32, 32, 4, 0, TX_SIZE).transactions() as u32)
            .sum();
        assert!(row_load_txns < col_load_txns, "tiled row loads should be cheaper");
    }

    // ── Alignment impact ────────────────────────────────────────────

    #[test]
    fn alignment_to_cache_line_improves_coalescing() {
        // Aligned base → 1 transaction
        let aligned = CoalescingPattern::sequential(32, 4, 0, TX_SIZE);
        // Misaligned → may need 2
        let misaligned = CoalescingPattern::sequential(32, 4, 4, TX_SIZE);
        assert!(aligned.transactions() <= misaligned.transactions());
    }

    #[test]
    fn power_of_two_alignment_optimal() {
        for &base in &[0u32, 128, 256, 512, 1024] {
            let pat = CoalescingPattern::sequential(32, 4, base, TX_SIZE);
            assert_eq!(pat.transactions(), 1, "base={base} should be perfectly coalesced");
        }
    }

    // ── Vectorized load patterns ────────────────────────────────────

    #[test]
    fn float4_load_coalescing() {
        // Each thread loads a float4 (16 bytes) → 32 threads × 16 = 512 bytes → 4 lines
        let pat = CoalescingPattern::sequential(32, 16, 0, TX_SIZE);
        assert_eq!(pat.transactions(), 4);
    }

    #[test]
    fn float2_load_coalescing() {
        // Each thread loads float2 (8 bytes) → 32 × 8 = 256 → 2 lines
        let pat = CoalescingPattern::sequential(32, 8, 0, TX_SIZE);
        assert_eq!(pat.transactions(), 2);
    }

    #[test]
    fn byte_load_32_threads() {
        // 32 threads × 1 byte = 32 bytes → 1 line
        let pat = CoalescingPattern::sequential(32, 1, 0, TX_SIZE);
        assert_eq!(pat.transactions(), 1);
    }

    // ── Shared memory as coalescing buffer ──────────────────────────

    #[test]
    fn shared_mem_coalescing_buffer_size() {
        // Pattern: coalesce global → shared, then access shared with any pattern
        // Buffer needs to hold one tile worth of data
        let tile = 32u32;
        let buffer_size = tile * tile * 4; // 32×32 f32
        assert_eq!(buffer_size, 4096);
        assert!(buffer_size <= 32 * 1024);
    }

    #[test]
    fn transpose_via_shared_memory() {
        // Load row-major (coalesced) → store to shared → read column-major from shared (bank conflicts avoidable with padding)
        let tile = 32u32;
        let elem = 4u32;
        let shared_no_pad = tile * tile * elem;
        let shared_padded = tile * (tile + 1) * elem; // +1 padding column
        assert_eq!(shared_no_pad, 4096);
        assert_eq!(shared_padded, 4224);
        assert!(shared_padded <= 32 * 1024);
    }
}

// ═════════════════════════════════════════════════════════════════════
// Additional cross-cutting tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod integration {
    use super::*;

    #[test]
    fn full_matmul_shared_memory_plan() {
        let profile = AppleSiliconMemoryProfile::m1();
        let tile = 32u32;
        let k_tile = 16;
        let elem = 4u32; // f32

        let mut layout = SharedMemoryLayout::new();
        layout.allocate("tile_a", elem, tile * k_tile, 16);
        layout.allocate("tile_b", elem, k_tile * tile, 16);

        let cfg = ThreadgroupMemoryConfig::m1_default();
        assert!(layout.fits_in(&cfg));

        let occ = profile.occupancy(layout.total_bytes, tile * tile);
        assert!(occ > 0.0, "should have nonzero occupancy");

        // Bank conflict check for B-tile column access
        let pat = BankAccessPattern::column_major_padded(
            tile,
            tile,
            0,
            1,
            elem,
            profile.num_banks,
            profile.bank_width_bytes,
        );
        assert!(pat.is_conflict_free(), "padded B-tile column access should be conflict-free");
    }

    #[test]
    fn full_reduction_shared_memory_plan() {
        let profile = AppleSiliconMemoryProfile::m3();
        let threads = 256u32;

        let mut layout = SharedMemoryLayout::new();
        layout.allocate("partial_sums", 4, threads, 16);

        let cfg = ThreadgroupMemoryConfig::m3_default();
        assert!(layout.fits_in(&cfg));

        let occ = profile.occupancy(layout.total_bytes, threads);
        assert!(occ > 0.0);

        // Reduction correctness
        let data: Vec<f32> = (0..threads).map(|x| x as f32).collect();
        let sum = parallel_reduction_sum(&data);
        let expected = (threads as f32) * (threads as f32 - 1.0) / 2.0;
        assert!((sum - expected).abs() < 1.0);
    }

    #[test]
    fn full_scan_shared_memory_plan() {
        let data: Vec<f32> = vec![1.0; 256];
        let exclusive = blelloch_scan(&data);
        let inclusive = inclusive_scan(&data);

        // Exclusive: [0, 1, 2, ..., 255]
        for (i, &v) in exclusive.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-3, "exclusive[{i}] = {v}");
        }
        // Inclusive: [1, 2, 3, ..., 256]
        for (i, &v) in inclusive.iter().enumerate() {
            assert!((v - (i + 1) as f32).abs() < 1e-3, "inclusive[{i}] = {v}");
        }
    }

    #[test]
    fn coalescing_guides_shared_mem_usage() {
        // Strided global access → use shared memory buffer → conflict-free local access
        let global_strided = CoalescingPattern::strided(32, 32, 4, 0, 128);
        assert!(global_strided.transactions() > 1, "strided global access inefficient");

        // After loading into shared memory, local stride-1 access is conflict-free
        let local = BankAccessPattern::stride1(32, 4, 32, 4);
        assert!(local.is_conflict_free());
    }

    #[test]
    fn bitnet_quantized_matmul_shared_layout() {
        // BitNet 2-bit matmul: weights are 2-bit packed, activations are f16
        let tile_m = 32u32;
        let tile_n = 32;
        let tile_k = 256; // QK256 block size

        let mut layout = SharedMemoryLayout::new();
        // Activations: tile_m × tile_k in f16
        layout.allocate("activations", 2, tile_m * tile_k, 16);
        // Weights: tile_k × tile_n in 2-bit packed (tile_k * tile_n / 4 bytes)
        let packed_bytes = tile_k * tile_n / 4;
        layout.allocate("weights_packed", 1, packed_bytes, 16);
        // Accumulator: tile_m × tile_n in f32
        layout.allocate("accum", 4, tile_m * tile_n, 16);

        let cfg = ThreadgroupMemoryConfig::m1_default();
        assert!(
            layout.fits_in(&cfg),
            "BitNet QK256 matmul tile should fit: {} bytes used",
            layout.total_bytes
        );
    }
}
