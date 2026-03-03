//! Elementwise tensor math operations.
//!
//! Provides basic vector operations: add, mul, dot product,
//! reduction (sum, max, argmax), broadcasting, scaling.

/// Elementwise addition: out[i] = a[i] + b[i].
pub fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Elementwise multiplication: out[i] = a[i] * b[i].
pub fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Scalar multiply: out[i] = a[i] * s.
pub fn vec_scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}

/// Elementwise subtract: out[i] = a[i] - b[i].
pub fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Dot product: sum(a[i] * b[i]).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Sum of all elements.
pub fn vec_sum(a: &[f32]) -> f32 {
    a.iter().sum()
}

/// Mean of all elements.
pub fn vec_mean(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    vec_sum(a) / a.len() as f32
}

/// Maximum element.
pub fn vec_max(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Minimum element.
pub fn vec_min(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Index of maximum element.
pub fn vec_argmax(a: &[f32]) -> usize {
    a.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// L2 norm (Euclidean).
pub fn vec_l2_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize to unit vector.
pub fn vec_normalize(a: &[f32]) -> Vec<f32> {
    let norm = vec_l2_norm(a);
    if norm == 0.0 {
        return vec![0.0; a.len()];
    }
    a.iter().map(|x| x / norm).collect()
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let na = vec_l2_norm(a);
    let nb = vec_l2_norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Clamp each element to [min_val, max_val].
pub fn vec_clamp(a: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    a.iter().map(|x| x.clamp(min_val, max_val)).collect()
}

/// Apply function elementwise.
pub fn vec_map(a: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    a.iter().map(|&x| f(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_add() {
        let r = vec_add(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(r, vec![4.0, 6.0]);
    }

    #[test]
    fn test_vec_mul() {
        let r = vec_mul(&[2.0, 3.0], &[4.0, 5.0]);
        assert_eq!(r, vec![8.0, 15.0]);
    }

    #[test]
    fn test_vec_scale() {
        let r = vec_scale(&[1.0, 2.0, 3.0], 2.0);
        assert_eq!(r, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vec_sub() {
        let r = vec_sub(&[5.0, 3.0], &[1.0, 2.0]);
        assert_eq!(r, vec![4.0, 1.0]);
    }

    #[test]
    fn test_dot_product() {
        let d = dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec_sum_and_mean() {
        assert!((vec_sum(&[1.0, 2.0, 3.0]) - 6.0).abs() < 1e-6);
        assert!((vec_mean(&[2.0, 4.0]) - 3.0).abs() < 1e-6);
        assert_eq!(vec_mean(&[]), 0.0);
    }

    #[test]
    fn test_max_min_argmax() {
        let v = vec![1.0, 5.0, 3.0, 2.0];
        assert!((vec_max(&v) - 5.0).abs() < 1e-6);
        assert!((vec_min(&v) - 1.0).abs() < 1e-6);
        assert_eq!(vec_argmax(&v), 1);
    }

    #[test]
    fn test_l2_norm() {
        let n = vec_l2_norm(&[3.0, 4.0]);
        assert!((n - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let r = vec_normalize(&[3.0, 4.0]);
        assert!((r[0] - 0.6).abs() < 1e-6);
        assert!((r[1] - 0.8).abs() < 1e-6);
        assert_eq!(vec_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let s = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s - 1.0).abs() < 1e-6);
        let s2 = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s2.abs() < 1e-6);
    }

    #[test]
    fn test_vec_clamp() {
        let r = vec_clamp(&[-1.0, 0.5, 2.0], 0.0, 1.0);
        assert_eq!(r, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_vec_map() {
        let r = vec_map(&[1.0, 4.0, 9.0], |x| x.sqrt());
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 2.0).abs() < 1e-6);
        assert!((r[2] - 3.0).abs() < 1e-6);
    }
}
