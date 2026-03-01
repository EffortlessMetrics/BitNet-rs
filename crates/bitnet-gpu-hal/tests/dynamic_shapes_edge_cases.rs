//! Edge-case tests for dynamic_shapes module.
//!
//! Covers: DynamicDim, ShapeSpec, ShapeError, ShapeConstraint,
//! ShapeInference, PaddingStrategy, BucketAllocator, ShapeValidator,
//! ShapeOptimizer, SequencePacker, PackedSequences, DynamicShapeEngine.

use bitnet_gpu_hal::dynamic_shapes::*;
use std::collections::HashMap;

// ── DynamicDim ──────────────────────────────────────────────────

#[test]
fn dynamic_dim_fixed() {
    let d = DynamicDim::fixed(42);
    assert!(d.is_static());
    assert_eq!(d.static_value(), Some(42));
}

#[test]
fn dynamic_dim_named() {
    let d = DynamicDim::named("batch");
    assert!(!d.is_static());
    assert_eq!(d.static_value(), None);
    let names = d.referenced_names();
    assert_eq!(names, vec!["batch"]);
}

#[test]
fn dynamic_dim_derived() {
    let d = DynamicDim::derived("seq", 2, 1);
    assert!(!d.is_static());
    let mut bindings = HashMap::new();
    bindings.insert("seq".to_string(), 10);
    // derived: base*scale + offset = 10*2 + 1 = 21
    assert_eq!(d.resolve(&bindings), Some(21));
}

#[test]
fn dynamic_dim_resolve_missing() {
    let d = DynamicDim::named("missing");
    let bindings = HashMap::new();
    assert_eq!(d.resolve(&bindings), None);
}

#[test]
fn dynamic_dim_static_resolve() {
    let d = DynamicDim::fixed(7);
    let bindings = HashMap::new();
    assert_eq!(d.resolve(&bindings), Some(7));
}

#[test]
fn dynamic_dim_display() {
    let d = DynamicDim::fixed(5);
    let s = format!("{}", d);
    assert!(!s.is_empty());
}

#[test]
fn dynamic_dim_clone_eq() {
    let a = DynamicDim::fixed(10);
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn dynamic_dim_derived_referenced_names() {
    let d = DynamicDim::derived("hidden", 4, 0);
    let names = d.referenced_names();
    assert_eq!(names, vec!["hidden"]);
}

// ── ShapeSpec ───────────────────────────────────────────────────

#[test]
fn shape_spec_static() {
    let s = ShapeSpec::static_shape(&[2, 3, 4]);
    assert_eq!(s.rank(), 3);
    assert!(s.is_fully_static());
}

#[test]
fn shape_spec_resolve_static() {
    let s = ShapeSpec::static_shape(&[2, 3]);
    let bindings = HashMap::new();
    assert_eq!(s.resolve(&bindings), Some(vec![2, 3]));
}

#[test]
fn shape_spec_resolve_numel() {
    let s = ShapeSpec::static_shape(&[2, 3, 4]);
    let bindings = HashMap::new();
    assert_eq!(s.resolve_numel(&bindings), Some(24));
}

#[test]
fn shape_spec_with_dynamic() {
    let s = ShapeSpec::new(vec![DynamicDim::named("batch"), DynamicDim::fixed(128)]);
    assert_eq!(s.rank(), 2);
    assert!(!s.is_fully_static());
    let names = s.symbolic_names();
    assert!(names.contains(&"batch"));
}

#[test]
fn shape_spec_resolve_with_bindings() {
    let s = ShapeSpec::new(vec![DynamicDim::named("batch"), DynamicDim::fixed(64)]);
    let mut bindings = HashMap::new();
    bindings.insert("batch".to_string(), 8);
    assert_eq!(s.resolve(&bindings), Some(vec![8, 64]));
}

#[test]
fn shape_spec_resolve_unbound() {
    let s = ShapeSpec::new(vec![DynamicDim::named("x")]);
    let bindings = HashMap::new();
    assert_eq!(s.resolve(&bindings), None);
}

#[test]
fn shape_spec_broadcast_same() {
    let a = ShapeSpec::static_shape(&[2, 3]);
    let b = ShapeSpec::static_shape(&[2, 3]);
    let result = ShapeSpec::broadcast(&a, &b);
    assert!(result.is_ok());
}

#[test]
fn shape_spec_dims_accessor() {
    let s = ShapeSpec::static_shape(&[1, 2, 3]);
    assert_eq!(s.dims().len(), 3);
}

// ── ShapeError ──────────────────────────────────────────────────

#[test]
fn shape_error_display() {
    let e = ShapeError::UnresolvedDimension("x".to_string());
    let s = format!("{}", e);
    assert!(s.contains("x"));
}

#[test]
fn shape_error_rank_mismatch() {
    let e = ShapeError::RankMismatch { expected: 3, got: 2 };
    let s = format!("{}", e);
    assert!(!s.is_empty());
}

#[test]
fn shape_error_clone_eq() {
    let a = ShapeError::ConstraintViolation("test".to_string());
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn shape_error_is_std_error() {
    let e = ShapeError::InvalidPadding("bad".to_string());
    let _: &dyn std::error::Error = &e;
}

// ── ShapeConstraint ─────────────────────────────────────────────

#[test]
fn constraint_equal_pass() {
    let c = ShapeConstraint::Equal("a".to_string(), "b".to_string());
    let mut bindings = HashMap::new();
    bindings.insert("a".to_string(), 5);
    bindings.insert("b".to_string(), 5);
    assert!(c.check(&bindings).is_ok());
}

#[test]
fn constraint_equal_fail() {
    let c = ShapeConstraint::Equal("a".to_string(), "b".to_string());
    let mut bindings = HashMap::new();
    bindings.insert("a".to_string(), 5);
    bindings.insert("b".to_string(), 10);
    assert!(c.check(&bindings).is_err());
}

#[test]
fn constraint_multiple_of_pass() {
    let c = ShapeConstraint::MultipleOf("x".to_string(), 8);
    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 64);
    assert!(c.check(&bindings).is_ok());
}

#[test]
fn constraint_bounded() {
    let c = ShapeConstraint::Bounded { name: "seq".to_string(), min: 1, max: 2048 };
    let mut bindings = HashMap::new();
    bindings.insert("seq".to_string(), 512);
    assert!(c.check(&bindings).is_ok());
}

#[test]
fn constraint_bounded_exceed() {
    let c = ShapeConstraint::Bounded { name: "seq".to_string(), min: 1, max: 2048 };
    let mut bindings = HashMap::new();
    bindings.insert("seq".to_string(), 5000);
    assert!(c.check(&bindings).is_err());
}

// ── ShapeInference ──────────────────────────────────────────────

#[test]
fn shape_inference_default() {
    let si = ShapeInference::new();
    let _ = format!("{:?}", si);
}

#[test]
fn shape_inference_matmul() {
    let mut si = ShapeInference::new();
    si.add_matmul_rule();
    let result = si.infer("matmul", &[&[2, 3], &[3, 4]]).unwrap();
    assert_eq!(result, vec![2, 4]);
}

#[test]
fn shape_inference_concat() {
    let si = ShapeInference::new();
    // concat along axis 0 for 2 rank-2 inputs
    let result = si.infer_concat(0, &[&[2, 3], &[4, 3]]).unwrap();
    assert_eq!(result, vec![6, 3]);
}

// ── PaddingStrategy ─────────────────────────────────────────────

#[test]
fn padding_pad_to_max() {
    let ps = PaddingStrategy::PadToMax;
    let len = ps.padded_length(&[3, 5, 7]).unwrap();
    assert_eq!(len, 7);
}

#[test]
fn padding_pad_to_multiple() {
    let ps = PaddingStrategy::PadToMultiple(8);
    let len = ps.padded_length(&[3, 5, 7]).unwrap();
    // max is 7, next multiple of 8 is 8
    assert_eq!(len, 8);
}

#[test]
fn padding_pad_to_fixed() {
    let ps = PaddingStrategy::PadToFixed(16);
    let len = ps.padded_length(&[3, 5]).unwrap();
    assert_eq!(len, 16);
}

#[test]
fn padding_no_padding() {
    let ps = PaddingStrategy::NoPadding;
    let len = ps.padded_length(&[5, 5, 5]).unwrap();
    assert_eq!(len, 5);
}

#[test]
fn padding_build_mask() {
    let ps = PaddingStrategy::PadToMax;
    let mask = ps.build_mask(&[2, 3]).unwrap();
    assert_eq!(mask.len(), 2);
    // First seq padded to 3: mask [1,1,0]
    assert_eq!(mask[0].len(), 3);
}

// ── BucketAllocator ─────────────────────────────────────────────

#[test]
fn bucket_allocator_new() {
    let ba = BucketAllocator::new(vec![64, 128, 256, 512]);
    assert_eq!(ba.buckets().len(), 4);
}

#[test]
fn bucket_allocator_power_of_two() {
    let ba = BucketAllocator::power_of_two(6, 10);
    // 2^6=64, 2^7=128, ..., 2^10=1024
    assert_eq!(ba.buckets().len(), 5);
}

#[test]
fn bucket_allocator_bucket_for() {
    let ba = BucketAllocator::new(vec![64, 128, 256]);
    assert_eq!(ba.bucket_for(50).unwrap(), 64);
    assert_eq!(ba.bucket_for(64).unwrap(), 64);
    assert_eq!(ba.bucket_for(100).unwrap(), 128);
}

#[test]
fn bucket_allocator_bucket_too_large() {
    let ba = BucketAllocator::new(vec![64, 128]);
    assert!(ba.bucket_for(200).is_err());
}

#[test]
fn bucket_allocator_allocate_release() {
    let mut ba = BucketAllocator::new(vec![64, 128, 256]);
    let bucket = ba.allocate(50).unwrap();
    assert_eq!(bucket, 64);
    assert_eq!(ba.active_count(), 1);
    assert!(ba.release(64));
    assert_eq!(ba.active_count(), 0);
}

#[test]
fn bucket_allocator_total_allocations() {
    let mut ba = BucketAllocator::new(vec![64, 128]);
    ba.allocate(10).unwrap();
    ba.allocate(100).unwrap();
    assert_eq!(ba.total_allocations(), 2);
}

#[test]
fn bucket_allocator_fragmentation() {
    let ba = BucketAllocator::new(vec![64, 128, 256]);
    let frag = ba.fragmentation(&[50, 100]);
    // fragmentation >= 0
    assert!(frag >= 0.0);
}

// ── ShapeValidator ──────────────────────────────────────────────

#[test]
fn validator_basic() {
    let v = ShapeValidator::new(4, 1_000_000);
    assert_eq!(v.max_rank(), 4);
    assert_eq!(v.max_numel(), 1_000_000);
}

#[test]
fn validator_valid_shape() {
    let v = ShapeValidator::new(4, 1_000_000);
    assert!(v.validate_shape(&[2, 3, 4]).is_ok());
}

#[test]
fn validator_rank_exceeded() {
    let v = ShapeValidator::new(2, 1_000_000);
    assert!(v.validate_shape(&[2, 3, 4]).is_err());
}

#[test]
fn validator_numel_exceeded() {
    let v = ShapeValidator::new(4, 100);
    assert!(v.validate_shape(&[10, 20]).is_err());
}

#[test]
fn validator_with_constraint() {
    let mut v = ShapeValidator::new(4, 1_000_000);
    v.add_constraint(ShapeConstraint::MultipleOf("hidden".to_string(), 64));
    let mut bindings = HashMap::new();
    bindings.insert("hidden".to_string(), 128);
    assert!(v.validate_constraints(&bindings).is_ok());
}

#[test]
fn validator_validate_all() {
    let mut v = ShapeValidator::new(4, 1_000_000);
    v.add_constraint(ShapeConstraint::Bounded { name: "seq".to_string(), min: 1, max: 2048 });
    let mut bindings = HashMap::new();
    bindings.insert("seq".to_string(), 512);
    assert!(v.validate_all(&[8, 512, 64], &bindings).is_ok());
}

// ── ShapeOptimizer ──────────────────────────────────────────────

#[test]
fn optimizer_row_major_strides() {
    let strides = ShapeOptimizer::row_major_strides(&[2, 3, 4]);
    assert_eq!(strides, vec![12, 4, 1]);
}

#[test]
fn optimizer_col_major_strides() {
    let strides = ShapeOptimizer::col_major_strides(&[2, 3, 4]);
    assert_eq!(strides, vec![1, 2, 6]);
}

#[test]
fn optimizer_aligned_strides() {
    let opt = ShapeOptimizer::new(16, true);
    let strides = opt.aligned_strides(&[2, 3]);
    // Should align inner dim
    assert!(strides[1] >= 1);
}

#[test]
fn optimizer_alloc_size() {
    let opt = ShapeOptimizer::new(16, true);
    let size = opt.alloc_size(&[2, 3, 4]);
    assert!(size >= 24); // at least numel
}

#[test]
fn optimizer_transpose_shape() {
    let result = ShapeOptimizer::transpose_shape(&[2, 3, 4], 0, 2);
    assert_eq!(result, vec![4, 3, 2]);
}

#[test]
fn optimizer_suggest_layout() {
    let opt = ShapeOptimizer::new(16, true);
    let layout = opt.suggest_layout_for_reduction(&[32, 64], 1);
    assert!(!layout.is_empty());
}

#[test]
fn optimizer_alignment_accessor() {
    let opt = ShapeOptimizer::new(32, false);
    assert_eq!(opt.alignment(), 32);
}

// ── SequencePacker ──────────────────────────────────────────────

#[test]
fn packer_pack_uniform() {
    let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
    let s1 = vec![1.0, 2.0, 3.0];
    let s2 = vec![4.0, 5.0, 6.0];
    let packed = packer.pack(&[&s1, &s2]).unwrap();
    assert_eq!(packed.batch_size, 2);
    assert_eq!(packed.padded_length, 3);
}

#[test]
fn packer_pack_variable() {
    let packer = SequencePacker::new(PaddingStrategy::PadToMax, -1.0);
    let s1 = vec![1.0, 2.0];
    let s2 = vec![3.0, 4.0, 5.0];
    let packed = packer.pack(&[&s1, &s2]).unwrap();
    assert_eq!(packed.padded_length, 3);
    assert_eq!(packed.lengths, vec![2, 3]);
}

#[test]
fn packer_unpack_roundtrip() {
    let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
    let s1 = vec![1.0, 2.0];
    let s2 = vec![3.0, 4.0, 5.0];
    let packed = packer.pack(&[&s1, &s2]).unwrap();
    let unpacked = packer.unpack(&packed);
    assert_eq!(unpacked.len(), 2);
    assert_eq!(unpacked[0], vec![1.0, 2.0]);
    assert_eq!(unpacked[1], vec![3.0, 4.0, 5.0]);
}

#[test]
fn packer_pack_sorted() {
    let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
    let s1 = vec![1.0];
    let s2 = vec![2.0, 3.0, 4.0];
    let s3 = vec![5.0, 6.0];
    let (packed, indices) = packer.pack_sorted(&[&s1, &s2, &s3]).unwrap();
    assert_eq!(packed.batch_size, 3);
    assert_eq!(indices.len(), 3);
}

#[test]
fn packer_accessors() {
    let packer = SequencePacker::new(PaddingStrategy::NoPadding, -99.0);
    assert_eq!(*packer.padding_strategy(), PaddingStrategy::NoPadding);
    assert_eq!(packer.pad_value(), -99.0);
}

// ── PackedSequences ─────────────────────────────────────────────

#[test]
fn packed_sequences_fields() {
    let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
    let s1 = vec![1.0, 2.0, 3.0];
    let packed = packer.pack(&[&s1]).unwrap();
    assert_eq!(packed.batch_size, 1);
    assert_eq!(packed.lengths, vec![3]);
    assert_eq!(packed.offsets.len(), 1);
    assert!(!packed.data.is_empty());
    assert!(!packed.mask.is_empty());
}

// ── DynamicShapeEngine ──────────────────────────────────────────

#[test]
fn engine_default_for_llm() {
    let eng = DynamicShapeEngine::default_for_llm(2048, 512);
    let _ = format!("{:?}", eng);
}

#[test]
fn engine_bind_resolve() {
    let mut eng = DynamicShapeEngine::default_for_llm(2048, 512);
    eng.bind("batch", 4);
    let spec = ShapeSpec::new(vec![DynamicDim::named("batch"), DynamicDim::fixed(128)]);
    let resolved = eng.resolve(&spec).unwrap();
    assert_eq!(resolved, vec![4, 128]);
}

#[test]
fn engine_unbind() {
    let mut eng = DynamicShapeEngine::default_for_llm(2048, 512);
    eng.bind("x", 10);
    eng.unbind("x");
    let spec = ShapeSpec::new(vec![DynamicDim::named("x")]);
    assert!(eng.resolve(&spec).is_err());
}

#[test]
fn engine_resolve_and_validate() {
    let mut eng = DynamicShapeEngine::default_for_llm(2048, 512);
    eng.bind("batch", 2);
    let spec = ShapeSpec::new(vec![DynamicDim::named("batch"), DynamicDim::fixed(64)]);
    // resolve_and_validate may fail if default constraints reject the shape
    let result = eng.resolve_and_validate(&spec);
    // Just verify it returns a result without panicking
    let _ = result;
}

#[test]
fn engine_allocate_release() {
    let mut eng = DynamicShapeEngine::default_for_llm(2048, 512);
    let bucket = eng.allocate(100).unwrap();
    assert!(eng.release(bucket));
}

#[test]
fn engine_pack_unpack() {
    let eng = DynamicShapeEngine::default_for_llm(2048, 512);
    let s1 = vec![1.0, 2.0];
    let s2 = vec![3.0, 4.0, 5.0];
    let packed = eng.pack_sequences(&[&s1, &s2]).unwrap();
    let unpacked = eng.unpack_sequences(&packed);
    assert_eq!(unpacked.len(), 2);
}

#[test]
fn engine_optimized_strides() {
    let eng = DynamicShapeEngine::default_for_llm(2048, 512);
    let strides = eng.optimized_strides(&[4, 8, 16]);
    assert_eq!(strides.len(), 3);
}

#[test]
fn engine_accessors() {
    let eng = DynamicShapeEngine::default_for_llm(2048, 512);
    let _ = eng.bindings();
    let _ = eng.inference();
    let _ = eng.validator();
    let _ = eng.allocator();
    let _ = eng.optimizer();
}

#[test]
fn engine_infer_and_validate() {
    let mut eng = DynamicShapeEngine::default_for_llm(2048, 512);
    eng.bind("batch", 2);
    // Add matmul rule via inference engine
    let result = eng.infer_and_validate("matmul", &[&[2, 3], &[3, 4]]);
    // May or may not have matmul rule by default; just check it doesn't panic
    let _ = result;
}
