//! Edge-case tests for sparse_operations module.
//!
//! Covers: SparseFormat, SparseMatrix (CSR/CSC/COO), SparseError,
//! SparseMatMul, SparseSoftmax, SparseAttention, TopKSparsifier,
//! BlockSparseFormat, SparseConverter, SparseAnalyzer, SparsityStats,
//! SparseEngine.

use bitnet_gpu_hal::sparse_operations::*;
use std::collections::HashSet;

// ── SparseFormat ────────────────────────────────────────────────

#[test]
fn sparse_format_all_variants() {
    let variants = vec![
        SparseFormat::CSR,
        SparseFormat::CSC,
        SparseFormat::COO,
        SparseFormat::BSR,
        SparseFormat::ELL,
    ];
    assert_eq!(variants.len(), 5);
}

#[test]
fn sparse_format_display() {
    let s = format!("{}", SparseFormat::CSR);
    assert!(!s.is_empty());
    let s2 = format!("{}", SparseFormat::COO);
    assert!(!s2.is_empty());
    assert_ne!(s, s2);
}

#[test]
fn sparse_format_clone_eq() {
    let a = SparseFormat::BSR;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn sparse_format_debug() {
    let dbg = format!("{:?}", SparseFormat::ELL);
    assert!(dbg.contains("ELL"));
}

#[test]
fn sparse_format_hash_dedup() {
    let mut set = HashSet::new();
    set.insert(SparseFormat::CSR);
    set.insert(SparseFormat::CSR);
    set.insert(SparseFormat::CSC);
    assert_eq!(set.len(), 2);
}

// ── SparseError ─────────────────────────────────────────────────

#[test]
fn sparse_error_display() {
    let e = SparseError("test error".to_string());
    let display = format!("{}", e);
    assert!(display.contains("test error"));
}

#[test]
fn sparse_error_debug() {
    let e = SparseError("msg".to_string());
    let dbg = format!("{:?}", e);
    assert!(dbg.contains("msg"));
}

#[test]
fn sparse_error_clone_eq() {
    let a = SparseError("a".to_string());
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn sparse_error_is_std_error() {
    let e = SparseError("err".to_string());
    let _: &dyn std::error::Error = &e;
}

// ── SparseMatrix construction ───────────────────────────────────

#[test]
fn sparse_matrix_new_csr_identity_3x3() {
    let m =
        SparseMatrix::new_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 3);
    assert_eq!(m.nnz, 3);
    assert_eq!(m.format, SparseFormat::CSR);
}

#[test]
fn sparse_matrix_new_csr_empty_rows() {
    let m = SparseMatrix::new_csr(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!(m.nnz, 0);
}

#[test]
fn sparse_matrix_new_coo_basic() {
    let m = SparseMatrix::new_coo(2, 2, vec![0, 1], vec![0, 1], vec![5.0, 10.0]).unwrap();
    assert_eq!(m.nnz, 2);
    assert_eq!(m.format, SparseFormat::COO);
}

#[test]
fn sparse_matrix_new_csc_basic() {
    let m = SparseMatrix::new_csc(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
    assert_eq!(m.nnz, 2);
    assert_eq!(m.format, SparseFormat::CSC);
}

#[test]
fn sparse_matrix_empty() {
    let m = SparseMatrix::empty(5, 10, SparseFormat::CSR);
    assert_eq!(m.rows, 5);
    assert_eq!(m.cols, 10);
    assert_eq!(m.nnz, 0);
}

#[test]
fn sparse_matrix_empty_coo() {
    let m = SparseMatrix::empty(3, 3, SparseFormat::COO);
    assert_eq!(m.nnz, 0);
    assert_eq!(m.format, SparseFormat::COO);
}

#[test]
fn sparse_matrix_to_dense_identity() {
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
    let dense = m.to_dense();
    assert_eq!(dense.len(), 4);
    assert_eq!(dense[0], 1.0);
    assert_eq!(dense[1], 0.0);
    assert_eq!(dense[2], 0.0);
    assert_eq!(dense[3], 1.0);
}

#[test]
fn sparse_matrix_to_dense_empty() {
    let m = SparseMatrix::empty(2, 3, SparseFormat::CSR);
    let dense = m.to_dense();
    assert_eq!(dense.len(), 6);
    assert!(dense.iter().all(|&v| v == 0.0));
}

#[test]
fn sparse_matrix_from_dense_csr_all_zero() {
    let data = vec![0.0; 9];
    let m = SparseMatrix::from_dense_csr(3, 3, &data);
    assert_eq!(m.nnz, 0);
}

#[test]
fn sparse_matrix_from_dense_csr_full() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let m = SparseMatrix::from_dense_csr(2, 2, &data);
    assert_eq!(m.nnz, 4);
    let roundtrip = m.to_dense();
    assert_eq!(roundtrip, data);
}

#[test]
fn sparse_matrix_from_dense_csr_partial() {
    let data = vec![1.0, 0.0, 0.0, 2.0];
    let m = SparseMatrix::from_dense_csr(2, 2, &data);
    assert_eq!(m.nnz, 2);
    let roundtrip = m.to_dense();
    assert_eq!(roundtrip, data);
}

#[test]
fn sparse_matrix_density() {
    let m =
        SparseMatrix::new_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
    let d = m.density();
    assert!((d - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn sparse_matrix_density_full() {
    let data = vec![1.0; 4];
    let m = SparseMatrix::from_dense_csr(2, 2, &data);
    assert!((m.density() - 1.0).abs() < 1e-10);
}

#[test]
fn sparse_matrix_density_empty() {
    let m = SparseMatrix::empty(3, 3, SparseFormat::CSR);
    assert!((m.density()).abs() < 1e-10);
}

#[test]
fn sparse_matrix_clone() {
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 7.0]).unwrap();
    let m2 = m.clone();
    assert_eq!(m2.nnz, 2);
    assert_eq!(m2.values, vec![3.0, 7.0]);
}

// ── SparseMatMul ────────────────────────────────────────────────

#[test]
fn sparse_matmul_default() {
    let mm = SparseMatMul::default();
    let _ = format!("{:?}", mm);
}

#[test]
fn sparse_matmul_new() {
    let mm = SparseMatMul::new();
    let _ = format!("{:?}", mm);
}

#[test]
fn sparse_matmul_identity_times_vector() {
    let mm = SparseMatMul::new();
    let identity =
        SparseMatrix::new_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
    let b = vec![1.0, 2.0, 3.0];
    let result = mm.mul_csr_dense(&identity, &b, 1).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn sparse_matmul_zero_matrix() {
    let mm = SparseMatMul::new();
    let zero = SparseMatrix::empty(2, 3, SparseFormat::CSR);
    let b = vec![1.0, 2.0, 3.0];
    let result = mm.mul_csr_dense(&zero, &b, 1).unwrap();
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn sparse_matmul_2x2_times_2x1() {
    let mm = SparseMatMul::new();
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap();
    let b = vec![4.0, 5.0];
    let result = mm.mul_csr_dense(&m, &b, 1).unwrap();
    assert_eq!(result, vec![8.0, 15.0]);
}

#[test]
fn dense_matmul_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0];
    let result = SparseMatMul::dense_matmul(&a, &b, 2, 2, 1);
    assert_eq!(result, vec![17.0, 39.0]);
}

#[test]
fn dense_matmul_identity() {
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![7.0, 8.0];
    let result = SparseMatMul::dense_matmul(&a, &b, 2, 2, 1);
    assert_eq!(result, vec![7.0, 8.0]);
}

// ── SparseSoftmax ───────────────────────────────────────────────

#[test]
fn sparse_softmax_default() {
    let sm = SparseSoftmax::default();
    let _ = format!("{:?}", sm);
}

#[test]
fn sparse_softmax_inplace_single_row() {
    let sm = SparseSoftmax::new();
    let mut mat =
        SparseMatrix::new_csr(1, 3, vec![0, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
    sm.softmax_csr_inplace(&mut mat).unwrap();
    let sum: f32 = mat.values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(mat.values[0] < mat.values[1]);
    assert!(mat.values[1] < mat.values[2]);
}

#[test]
fn dense_softmax_uniform() {
    let data = vec![1.0, 1.0, 1.0, 1.0];
    let result = SparseSoftmax::dense_softmax(&data, 1, 4);
    for v in &result {
        assert!((v - 0.25).abs() < 1e-5);
    }
}

#[test]
fn dense_softmax_two_rows() {
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let result = SparseSoftmax::dense_softmax(&data, 2, 2);
    assert!((result[0] + result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] + result[3] - 1.0).abs() < 1e-5);
}

// ── SparseAttention ─────────────────────────────────────────────

#[test]
fn sparse_attention_build_mask_small() {
    let cfg = SparseAttentionConfig { local_window: 2, stride: 2 };
    let sa = SparseAttention::new(cfg);
    let mask = sa.build_mask(4);
    assert_eq!(mask.rows, 4);
    assert_eq!(mask.cols, 4);
    assert!(mask.nnz > 0);
}

#[test]
fn sparse_attention_build_mask_seq1() {
    let cfg = SparseAttentionConfig { local_window: 1, stride: 1 };
    let sa = SparseAttention::new(cfg);
    let mask = sa.build_mask(1);
    assert_eq!(mask.rows, 1);
    assert!(mask.nnz >= 1);
}

#[test]
fn sparse_attention_apply_trivial() {
    let cfg = SparseAttentionConfig { local_window: 4, stride: 1 };
    let sa = SparseAttention::new(cfg);
    let seq_len = 2;
    let head_dim = 2;
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let result = sa.apply(&q, &k, &v, seq_len, head_dim);
    assert!(result.is_ok());
    let out = result.unwrap();
    assert_eq!(out.len(), seq_len * head_dim);
}

#[test]
fn sparse_attention_debug() {
    let cfg = SparseAttentionConfig { local_window: 3, stride: 2 };
    let sa = SparseAttention::new(cfg);
    let dbg = format!("{:?}", sa);
    assert!(dbg.contains("SparseAttention"));
}

// ── TopKSparsifier ──────────────────────────────────────────────

#[test]
fn topk_sparsifier_k1() {
    let ts = TopKSparsifier::new(1);
    assert_eq!(ts.k(), 1);
    let data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 4.0];
    let sparse = ts.sparsify(&data, 2, 3);
    assert_eq!(sparse.nnz, 2);
}

#[test]
fn topk_sparsifier_k_equals_cols() {
    let ts = TopKSparsifier::new(3);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sparse = ts.sparsify(&data, 2, 3);
    assert_eq!(sparse.nnz, 6);
}

#[test]
fn topk_sparsifier_preserves_values() {
    let ts = TopKSparsifier::new(2);
    let data = vec![10.0, 20.0, 5.0];
    let sparse = ts.sparsify(&data, 1, 3);
    assert_eq!(sparse.nnz, 2);
    let mut vals: Vec<f32> = sparse.values.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(vals[0], 20.0);
    assert_eq!(vals[1], 10.0);
}

#[test]
fn topk_sparsifier_clone_debug() {
    let ts = TopKSparsifier::new(5);
    let ts2 = ts.clone();
    assert_eq!(ts2.k(), 5);
    let dbg = format!("{:?}", ts);
    assert!(dbg.contains("TopKSparsifier"));
}

// ── BlockSparseFormat ───────────────────────────────────────────

#[test]
fn block_sparse_empty() {
    let bsf = BlockSparseFormat::empty(4, 4, 2, 2);
    assert_eq!(bsf.rows, 4);
    assert_eq!(bsf.cols, 4);
    assert_eq!(bsf.num_blocks(), 0);
}

#[test]
fn block_sparse_to_dense_empty() {
    let bsf = BlockSparseFormat::empty(4, 4, 2, 2);
    let dense = bsf.to_dense();
    assert_eq!(dense.len(), 16);
    assert!(dense.iter().all(|&v| v == 0.0));
}

#[test]
fn block_sparse_from_dense_all_zero() {
    let data = vec![0.0; 16];
    let bsf = BlockSparseFormat::from_dense(4, 4, 2, 2, &data, 0.0);
    assert_eq!(bsf.num_blocks(), 0);
}

#[test]
fn block_sparse_from_dense_roundtrip() {
    let data = vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0];
    let bsf = BlockSparseFormat::from_dense(4, 4, 2, 2, &data, 0.0);
    assert!(bsf.num_blocks() > 0);
    let roundtrip = bsf.to_dense();
    assert_eq!(roundtrip.len(), 16);
}

#[test]
fn block_sparse_new_valid() {
    let bsf =
        BlockSparseFormat::new(4, 4, 2, 2, vec![0], vec![0], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(bsf.num_blocks(), 1);
}

#[test]
fn block_sparse_clone_debug() {
    let bsf = BlockSparseFormat::empty(2, 2, 1, 1);
    let bsf2 = bsf.clone();
    assert_eq!(bsf2.num_blocks(), 0);
    let dbg = format!("{:?}", bsf);
    assert!(dbg.contains("BlockSparseFormat"));
}

// ── SparseConverter ─────────────────────────────────────────────

#[test]
fn sparse_converter_default() {
    let conv = SparseConverter::default();
    let _ = format!("{:?}", conv);
}

#[test]
fn sparse_converter_csr_to_csr() {
    let conv = SparseConverter::new();
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let csr = conv.to_csr(&m).unwrap();
    assert_eq!(csr.format, SparseFormat::CSR);
    assert_eq!(csr.nnz, 2);
}

#[test]
fn sparse_converter_csr_to_coo() {
    let conv = SparseConverter::new();
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let coo = conv.to_coo(&m).unwrap();
    assert_eq!(coo.format, SparseFormat::COO);
    assert_eq!(coo.nnz, 2);
}

#[test]
fn sparse_converter_csr_to_csc() {
    let conv = SparseConverter::new();
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let csc = conv.to_csc(&m).unwrap();
    assert_eq!(csc.format, SparseFormat::CSC);
    assert_eq!(csc.nnz, 2);
}

#[test]
fn sparse_converter_empty_matrix() {
    let conv = SparseConverter::new();
    let m = SparseMatrix::empty(3, 3, SparseFormat::CSR);
    let coo = conv.to_coo(&m).unwrap();
    assert_eq!(coo.nnz, 0);
}

#[test]
fn sparse_converter_roundtrip_csr_coo_csr() {
    let conv = SparseConverter::new();
    let original =
        SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let coo = conv.to_coo(&original).unwrap();
    let back = conv.to_csr(&coo).unwrap();
    assert_eq!(back.nnz, 3);
    let dense_orig = original.to_dense();
    let dense_back = back.to_dense();
    assert_eq!(dense_orig, dense_back);
}

// ── SparseAnalyzer ──────────────────────────────────────────────

#[test]
fn sparse_analyzer_default() {
    let a = SparseAnalyzer::default();
    let _ = format!("{:?}", a);
}

#[test]
fn sparse_analyzer_identity() {
    let a = SparseAnalyzer::new();
    let m =
        SparseMatrix::new_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
    let stats = a.analyze(&m).unwrap();
    assert_eq!(stats.rows, 3);
    assert_eq!(stats.cols, 3);
    assert_eq!(stats.nnz, 3);
    assert!((stats.density - 1.0 / 3.0).abs() < 1e-10);
    assert!((stats.avg_nnz_per_row - 1.0).abs() < 1e-10);
    assert_eq!(stats.max_nnz_per_row, 1);
    assert_eq!(stats.min_nnz_per_row, 1);
}

#[test]
fn sparse_analyzer_empty() {
    let a = SparseAnalyzer::new();
    let m = SparseMatrix::empty(4, 4, SparseFormat::CSR);
    let stats = a.analyze(&m).unwrap();
    assert_eq!(stats.nnz, 0);
    assert!(stats.density.abs() < 1e-10);
}

#[test]
fn sparse_analyzer_with_blocks() {
    let a = SparseAnalyzer::new();
    let m = SparseMatrix::from_dense_csr(
        4,
        4,
        &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let stats = a.analyze_with_blocks(&m, 2).unwrap();
    assert_eq!(stats.block_size, 2);
}

// ── SparsityStats ───────────────────────────────────────────────

#[test]
fn sparsity_stats_fields() {
    let a = SparseAnalyzer::new();
    let data = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let m = SparseMatrix::from_dense_csr(3, 3, &data);
    let stats = a.analyze(&m).unwrap();
    assert_eq!(stats.rows, 3);
    assert_eq!(stats.cols, 3);
    assert_eq!(stats.nnz, 3);
    let dbg = format!("{:?}", stats);
    assert!(dbg.contains("SparsityStats"));
}

// ── SparseEngine ────────────────────────────────────────────────

#[test]
fn sparse_engine_default() {
    let eng = SparseEngine::default();
    let _ = format!("{:?}", eng);
}

#[test]
fn sparse_engine_new() {
    let eng = SparseEngine::new();
    let _ = eng.converter();
    let _ = eng.matmul();
    let _ = eng.softmax();
    let _ = eng.analyzer();
}

#[test]
fn sparse_engine_spmm() {
    let eng = SparseEngine::new();
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap();
    let b = vec![1.0, 1.0];
    let result = eng.spmm(&m, &b, 1).unwrap();
    assert_eq!(result, vec![2.0, 3.0]);
}

#[test]
fn sparse_engine_softmax_inplace() {
    let eng = SparseEngine::new();
    let mut m =
        SparseMatrix::new_csr(1, 3, vec![0, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
    eng.softmax_inplace(&mut m).unwrap();
    let sum: f32 = m.values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn sparse_engine_analyze() {
    let eng = SparseEngine::new();
    let m = SparseMatrix::from_dense_csr(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let stats = eng.analyze(&m).unwrap();
    assert_eq!(stats.nnz, 2);
}

#[test]
fn sparse_engine_sparsify_topk() {
    let eng = SparseEngine::new();
    let data = vec![10.0, 1.0, 5.0, 3.0, 8.0, 2.0];
    let sparse = eng.sparsify_topk(&data, 2, 3, 1);
    assert_eq!(sparse.nnz, 2);
}

#[test]
fn sparse_engine_convert() {
    let eng = SparseEngine::new();
    let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let coo = eng.convert(&m, SparseFormat::COO).unwrap();
    assert_eq!(coo.format, SparseFormat::COO);
    assert_eq!(coo.nnz, 2);
}

#[test]
fn sparse_engine_convert_roundtrip() {
    let eng = SparseEngine::new();
    let original =
        SparseMatrix::from_dense_csr(3, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]);
    let coo = eng.convert(&original, SparseFormat::COO).unwrap();
    let back = eng.convert(&coo, SparseFormat::CSR).unwrap();
    assert_eq!(original.to_dense(), back.to_dense());
}

#[test]
fn sparse_engine_clone() {
    let eng = SparseEngine::new();
    let eng2 = eng.clone();
    let m = SparseMatrix::empty(1, 1, SparseFormat::CSR);
    let stats = eng2.analyze(&m).unwrap();
    assert_eq!(stats.nnz, 0);
}
