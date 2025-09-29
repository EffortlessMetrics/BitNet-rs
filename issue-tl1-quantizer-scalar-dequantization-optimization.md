# [Quantization] TL1 Quantizer Scalar Dequantization Production Implementation

## Problem Description

The `TL1Quantizer::dequantize_scalar` function uses a simplified linear dequantization approach instead of implementing proper Table Lookup (TL1) dequantization with lookup tables. This prevents the quantizer from achieving the precision and efficiency benefits that TL1 quantization is designed to provide.

## Environment

- **File**: `crates/bitnet-quantization/src/tl1.rs`
- **Function**: `TL1Quantizer::dequantize_scalar`
- **Component**: TL1 Quantization System
- **Rust Version**: 1.90.0+ (2024 edition)
- **Features**: `cpu`, `gpu` (quantization used in both backends)

## Root Cause Analysis

The current TL1 dequantization implementation lacks the fundamental characteristic of Table Lookup quantization - the use of precomputed lookup tables for efficient and accurate dequantization:

### **Current Implementation:**
```rust
fn dequantize_scalar(
    &self,
    quantized: &[i8],
    scales: &[f32],
    zero_points: &[i32],
) -> Result<Vec<f32>> {
    let mut dequantized = vec![0.0f32; quantized.len()];

    dequantized
        .par_chunks_mut(self.config.block_size)
        .zip(quantized.par_chunks(self.config.block_size))
        .zip(scales.par_iter())
        .zip(zero_points.par_iter())
        .for_each(|(((dequant_block, quant_block), &scale), &zero_point)| {
            for (i, &value) in quant_block.iter().enumerate() {
                let adjusted = if self.config.use_asymmetric {
                    value as i32 - zero_point
                } else {
                    value as i32
                };
                dequant_block[i] = adjusted as f32 * scale;
            }
        });

    Ok(dequantized)
}
```

### **Issues Identified:**
1. **Missing Lookup Tables**: No precomputed lookup tables for efficient dequantization
2. **Linear Quantization Logic**: Using simple linear quantization instead of TL1 algorithm
3. **No Table Reconstruction**: Missing logic to rebuild block-specific lookup tables
4. **Inefficient Computation**: Not leveraging the speed benefits of table lookups
5. **Precision Loss**: Not achieving the precision improvements of proper TL1

### **TL1 Theory:**
Table Lookup (TL1) quantization uses precomputed lookup tables to map quantized values back to floating-point values:
- **Benefits**: Faster dequantization, better precision control, efficient memory usage
- **Mechanism**: Precompute mapping tables for value ranges, use direct table lookups
- **Optimization**: Block-wise tables adapted to local value distributions

## Impact Assessment

### **Severity**: Medium
### **Affected Operations**: Neural network dequantization and inference accuracy
### **Business Impact**: Suboptimal quantization performance and precision

**Current Limitations:**
- Cannot achieve TL1 quantization efficiency benefits
- Missing precision improvements from proper table lookup
- Suboptimal dequantization performance
- Inconsistent with TL1 quantization specification

## Proposed Solution

### **Primary Approach**: Complete TL1 Lookup Table Implementation

Implement a production-ready TL1 dequantization system with proper lookup tables, block-wise adaptation, and optimized table reconstruction.

### **Implementation Strategy:**

#### **1. Lookup Table Infrastructure**
```rust
#[derive(Debug, Clone)]
pub struct LookupTable {
    table: Vec<f32>,
    min_val: f32,
    max_val: f32,
    num_entries: usize,
    precision_bits: u8,
    is_asymmetric: bool,
}

impl LookupTable {
    pub fn new(
        min_val: f32,
        max_val: f32,
        precision_bits: u8,
        is_asymmetric: bool,
    ) -> Self {
        let num_entries = if is_asymmetric {
            1 << precision_bits  // 2^precision_bits entries
        } else {
            1 << (precision_bits - 1)  // Symmetric: half for negative, half for positive
        };

        let mut table = Vec::with_capacity(num_entries);
        let step = (max_val - min_val) / (num_entries - 1) as f32;

        for i in 0..num_entries {
            let value = if is_asymmetric {
                min_val + i as f32 * step
            } else {
                // Symmetric quantization around zero
                let offset = i as i32 - (num_entries / 2) as i32;
                (offset as f32) * step
            };
            table.push(value);
        }

        Self {
            table,
            min_val,
            max_val,
            num_entries,
            precision_bits,
            is_asymmetric,
        }
    }

    pub fn dequantize(&self, quantized_value: i8) -> f32 {
        let index = if self.is_asymmetric {
            quantized_value as usize
        } else {
            // Map signed value to table index
            ((quantized_value as i32) + (self.num_entries / 2) as i32) as usize
        };

        // Bounds checking with clamping
        let clamped_index = index.min(self.num_entries - 1);
        self.table[clamped_index]
    }

    pub fn from_statistics(
        data: &[f32],
        precision_bits: u8,
        is_asymmetric: bool,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot create lookup table from empty data"));
        }

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            // Handle constant data
            return Ok(Self::new(min_val, min_val + 1e-6, precision_bits, is_asymmetric));
        }

        Ok(Self::new(min_val, max_val, precision_bits, is_asymmetric))
    }
}
```

#### **2. Enhanced TL1 Configuration**
```rust
#[derive(Debug, Clone)]
pub struct TL1Config {
    pub block_size: usize,
    pub precision_bits: u8,
    pub use_asymmetric: bool,
    pub adaptive_tables: bool,      // New: Enable block-adaptive tables
    pub table_cache_size: usize,    // New: Cache frequently used tables
    pub min_block_variance: f32,    // New: Minimum variance for table creation
}

impl Default for TL1Config {
    fn default() -> Self {
        Self {
            block_size: 128,
            precision_bits: 4,  // 16 table entries for 4-bit quantization
            use_asymmetric: false,
            adaptive_tables: true,
            table_cache_size: 1024,
            min_block_variance: 1e-6,
        }
    }
}
```

#### **3. Production TL1 Dequantization**
```rust
impl TL1Quantizer {
    fn dequantize_scalar(
        &self,
        quantized: &[i8],
        scales: &[f32],
        zero_points: &[i32],
    ) -> Result<Vec<f32>> {
        if quantized.is_empty() {
            return Ok(Vec::new());
        }

        let mut dequantized = vec![0.0f32; quantized.len()];

        // Process in blocks with proper TL1 lookup tables
        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .zip(zero_points.par_iter())
            .try_for_each(|(((dequant_block, quant_block), &scale), &zero_point)| -> Result<()> {
                // Create or retrieve block-specific lookup table
                let lookup_table = if self.config.adaptive_tables {
                    self.create_adaptive_lookup_table(quant_block, scale, zero_point)?
                } else {
                    self.create_standard_lookup_table(scale, zero_point)?
                };

                // Perform fast table lookup dequantization
                for (i, &quantized_val) in quant_block.iter().enumerate() {
                    dequant_block[i] = lookup_table.dequantize(quantized_val);
                }

                Ok(())
            })?;

        Ok(dequantized)
    }

    fn create_adaptive_lookup_table(
        &self,
        quantized_block: &[i8],
        scale: f32,
        zero_point: i32,
    ) -> Result<LookupTable> {
        // Analyze quantized values to determine optimal table parameters
        let min_quantized = quantized_block.iter().min().copied().unwrap_or(0) as i32;
        let max_quantized = quantized_block.iter().max().copied().unwrap_or(0) as i32;

        // Convert quantized range back to float range
        let min_val = if self.config.use_asymmetric {
            (min_quantized - zero_point) as f32 * scale
        } else {
            min_quantized as f32 * scale
        };

        let max_val = if self.config.use_asymmetric {
            (max_quantized - zero_point) as f32 * scale
        } else {
            max_quantized as f32 * scale
        };

        Ok(LookupTable::new(
            min_val,
            max_val,
            self.config.precision_bits,
            self.config.use_asymmetric,
        ))
    }

    fn create_standard_lookup_table(
        &self,
        scale: f32,
        zero_point: i32,
    ) -> Result<LookupTable> {
        // Standard table based on quantization parameters
        let num_levels = 1 << self.config.precision_bits;
        let half_levels = num_levels / 2;

        let (min_val, max_val) = if self.config.use_asymmetric {
            let min_quantized = -zero_point;
            let max_quantized = num_levels - 1 - zero_point;
            (min_quantized as f32 * scale, max_quantized as f32 * scale)
        } else {
            // Symmetric quantization
            let max_quantized = half_levels - 1;
            (-(max_quantized as f32) * scale, max_quantized as f32 * scale)
        };

        Ok(LookupTable::new(
            min_val,
            max_val,
            self.config.precision_bits,
            self.config.use_asymmetric,
        ))
    }
}
```

#### **4. Table Caching and Optimization**
```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TableKey {
    scale_bits: u32,      // Float bits for scale
    zero_point: i32,
    precision_bits: u8,
    is_asymmetric: bool,
}

impl TableKey {
    fn new(scale: f32, zero_point: i32, precision_bits: u8, is_asymmetric: bool) -> Self {
        Self {
            scale_bits: scale.to_bits(),
            zero_point,
            precision_bits,
            is_asymmetric,
        }
    }
}

pub struct TableCache {
    cache: Arc<Mutex<HashMap<TableKey, Arc<LookupTable>>>>,
    max_size: usize,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
}

impl TableCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
        }
    }

    pub fn get_or_create_table(
        &self,
        scale: f32,
        zero_point: i32,
        precision_bits: u8,
        is_asymmetric: bool,
    ) -> Result<Arc<LookupTable>> {
        let key = TableKey::new(scale, zero_point, precision_bits, is_asymmetric);

        {
            let cache = self.cache.lock().unwrap();
            if let Some(table) = cache.get(&key) {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                return Ok(table.clone());
            }
        }

        // Cache miss - create new table
        self.miss_count.fetch_add(1, Ordering::Relaxed);

        let min_val = if is_asymmetric {
            (-zero_point as f32) * scale
        } else {
            let half_range = (1 << (precision_bits - 1)) - 1;
            (-half_range as f32) * scale
        };

        let max_val = if is_asymmetric {
            ((1 << precision_bits) - 1 - zero_point) as f32 * scale
        } else {
            let half_range = (1 << (precision_bits - 1)) - 1;
            (half_range as f32) * scale
        };

        let table = Arc::new(LookupTable::new(min_val, max_val, precision_bits, is_asymmetric));

        // Add to cache with size management
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() >= self.max_size {
                // Simple LRU-like eviction: remove a random entry
                if let Some(key_to_remove) = cache.keys().next().cloned() {
                    cache.remove(&key_to_remove);
                }
            }
            cache.insert(key, table.clone());
        }

        Ok(table)
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        let total = hits + misses;

        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl TL1Quantizer {
    fn new_with_cache(config: TL1Config) -> Self {
        Self {
            config: config.clone(),
            table_cache: Arc::new(TableCache::new(config.table_cache_size)),
        }
    }

    fn dequantize_scalar_cached(
        &self,
        quantized: &[i8],
        scales: &[f32],
        zero_points: &[i32],
    ) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .zip(zero_points.par_iter())
            .try_for_each(|(((dequant_block, quant_block), &scale), &zero_point)| -> Result<()> {
                // Get cached lookup table
                let lookup_table = self.table_cache.get_or_create_table(
                    scale,
                    zero_point,
                    self.config.precision_bits,
                    self.config.use_asymmetric,
                )?;

                // Fast table lookup dequantization
                for (i, &quantized_val) in quant_block.iter().enumerate() {
                    dequant_block[i] = lookup_table.dequantize(quantized_val);
                }

                Ok(())
            })?;

        Ok(dequantized)
    }
}
```

## Implementation Plan

### **Phase 1: Lookup Table Infrastructure (Week 1)**

#### **Task 1.1: Core Lookup Table Implementation**
```rust
// Implement LookupTable struct with proper table generation
impl LookupTable {
    pub fn new(min_val: f32, max_val: f32, precision_bits: u8, is_asymmetric: bool) -> Self;
    pub fn dequantize(&self, quantized_value: i8) -> f32;
    pub fn from_statistics(data: &[f32], precision_bits: u8, is_asymmetric: bool) -> Result<Self>;
}
```

#### **Task 1.2: Enhanced Configuration**
```rust
// Update TL1Config with table-specific parameters
pub struct TL1Config {
    pub adaptive_tables: bool,
    pub table_cache_size: usize,
    pub min_block_variance: f32,
    // ... existing fields
}
```

### **Phase 2: TL1 Dequantization Implementation (Week 2)**

#### **Task 2.1: Production Dequantization Function**
```rust
// Replace simplified implementation with proper TL1 algorithm
fn dequantize_scalar(
    &self,
    quantized: &[i8],
    scales: &[f32],
    zero_points: &[i32],
) -> Result<Vec<f32>> {
    // Use lookup tables for efficient dequantization
}
```

#### **Task 2.2: Adaptive Table Creation**
```rust
// Implement block-adaptive table creation
fn create_adaptive_lookup_table(&self, ...) -> Result<LookupTable>;
fn create_standard_lookup_table(&self, ...) -> Result<LookupTable>;
```

### **Phase 3: Optimization and Caching (Week 3)**

#### **Task 3.1: Table Caching System**
```rust
// Implement efficient table caching
pub struct TableCache {
    cache: Arc<Mutex<HashMap<TableKey, Arc<LookupTable>>>>,
    // ... cache management
}
```

#### **Task 3.2: Performance Optimization**
- SIMD optimization for table lookups
- Memory layout optimization for cache efficiency
- Parallel processing optimization

## Testing Strategy

### **Unit Tests:**
```rust
#[cfg(test)]
mod tl1_tests {
    use super::*;

    #[test]
    fn test_lookup_table_creation() {
        let table = LookupTable::new(-1.0, 1.0, 4, false);
        assert_eq!(table.num_entries, 8); // 2^(4-1) for symmetric

        // Test boundary values
        assert_eq!(table.dequantize(-4), -1.0);
        assert_eq!(table.dequantize(3), 1.0);
    }

    #[test]
    fn test_asymmetric_quantization() {
        let table = LookupTable::new(0.0, 15.0, 4, true);
        assert_eq!(table.num_entries, 16); // 2^4 for asymmetric

        // Test asymmetric mapping
        assert_eq!(table.dequantize(0), 0.0);
        assert_eq!(table.dequantize(15), 15.0);
    }

    #[test]
    fn test_tl1_dequantization_accuracy() {
        let config = TL1Config {
            block_size: 32,
            precision_bits: 4,
            use_asymmetric: false,
            adaptive_tables: true,
            ..Default::default()
        };

        let quantizer = TL1Quantizer::new(config);

        // Test data
        let quantized = vec![-4, -2, 0, 2, 4];
        let scales = vec![0.1];
        let zero_points = vec![0];

        let dequantized = quantizer.dequantize_scalar(&quantized, &scales, &zero_points).unwrap();

        // Verify table lookup results
        assert!((dequantized[0] - (-0.4)).abs() < 1e-6);
        assert!((dequantized[2] - 0.0).abs() < 1e-6);
        assert!((dequantized[4] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_table_cache_efficiency() {
        let cache = TableCache::new(10);

        // Create multiple tables with same parameters
        let table1 = cache.get_or_create_table(0.1, 0, 4, false).unwrap();
        let table2 = cache.get_or_create_table(0.1, 0, 4, false).unwrap();

        // Should be the same cached instance
        assert!(Arc::ptr_eq(&table1, &table2));
        assert!(cache.hit_rate() > 0.0);
    }
}
```

### **Performance Tests:**
```rust
#[test]
fn test_tl1_vs_linear_performance() {
    let data_size = 1000000;
    let quantized: Vec<i8> = (0..data_size).map(|i| (i % 256) as i8).collect();
    let scales = vec![0.01; data_size / 128];
    let zero_points = vec![0; data_size / 128];

    // Test TL1 performance
    let tl1_config = TL1Config::default();
    let tl1_quantizer = TL1Quantizer::new(tl1_config);

    let start = Instant::now();
    let _result = tl1_quantizer.dequantize_scalar(&quantized, &scales, &zero_points).unwrap();
    let tl1_duration = start.elapsed();

    // TL1 should be faster than linear approach for large datasets
    println!("TL1 dequantization time: {:?}", tl1_duration);
}
```

## Alternative Approaches

### **Alternative 1: Simple Linear Scaling**
**Approach**: Keep current linear scaling approach
**Pros**: Simpler implementation, no table overhead
**Cons**: Missing TL1 benefits, slower for repeated operations

### **Alternative 2: Hardware-Specific Lookup Tables**
**Approach**: Use CPU/GPU-specific optimized table implementations
**Pros**: Maximum performance on specific hardware
**Cons**: Higher complexity, platform-specific code

### **Alternative 3: Hybrid Approach**
**Approach**: Use tables for frequently accessed blocks, linear for others
**Pros**: Balanced complexity and performance
**Cons**: More complex caching logic

**Selected Approach**: Primary table lookup implementation provides the best balance of TL1 compliance and performance benefits.

## Performance Considerations

### **Memory Usage:**
- **Table Storage**: ~64 bytes per table (16 entries × 4 bytes for 4-bit quantization)
- **Cache Overhead**: Configurable cache size with LRU management
- **Total Overhead**: <1% of typical model memory usage

### **Computational Performance:**
- **Table Lookup**: O(1) dequantization per value
- **Table Creation**: Amortized over multiple uses via caching
- **Expected Speedup**: 2-5x faster than linear scaling for repeated operations

## Success Metrics

### **Functionality:**
- [ ] Proper TL1 lookup table generation and usage
- [ ] Accurate dequantization for all supported precision levels
- [ ] Block-adaptive table creation for improved precision
- [ ] Efficient table caching with high hit rates

### **Performance:**
- [ ] 2-5x speedup over linear dequantization for large datasets
- [ ] <1% memory overhead for table storage and caching
- [ ] >90% cache hit rate for typical inference workloads
- [ ] No accuracy degradation compared to reference implementation

### **Quality:**
- [ ] Unit test coverage >95% for TL1 functionality
- [ ] Performance benchmarks validate speedup claims
- [ ] Accuracy tests confirm mathematical correctness
- [ ] Integration tests validate end-to-end quantization pipeline

## Acceptance Criteria

- [ ] `dequantize_scalar` uses proper TL1 lookup tables instead of linear scaling
- [ ] Lookup tables are correctly generated based on quantization parameters
- [ ] Block-adaptive tables improve precision for varying data distributions
- [ ] Table caching provides performance benefits for repeated operations
- [ ] Performance improves significantly over linear approach for large datasets
- [ ] Mathematical accuracy is maintained or improved
- [ ] Unit tests validate all TL1 operations and edge cases
- [ ] Documentation explains TL1 table lookup benefits and usage

## Labels

- `quantization`
- `tl1-algorithm`
- `performance-optimization`
- `lookup-tables`
- `cpu-gpu-common`

## Related Issues

- **Dependencies**: Issue #XXX (Quantization Infrastructure)
- **Related**: Issue #XXX (I2S Quantization), Issue #XXX (Performance Optimization)
- **Enables**: Efficient TL1 quantization, improved inference performance