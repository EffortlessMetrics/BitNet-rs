#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayoutInput {
    dims: Vec<u16>,
    /// Whether to use C-contiguous (row-major) or Fortran (column-major) order.
    fortran_order: bool,
}

// Minimal tensor layout for fuzzing stride and size calculations.
struct TensorLayout {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorLayout {
    /// Create a C-contiguous (row-major) layout.
    fn c_contiguous(dims: &[usize]) -> Self {
        let n = dims.len();
        let mut strides = vec![0usize; n];
        if n > 0 {
            strides[n - 1] = 1;
            for i in (0..n - 1).rev() {
                strides[i] = strides[i + 1].saturating_mul(dims[i + 1]);
            }
        }
        Self { dims: dims.to_vec(), strides }
    }

    /// Create a Fortran-contiguous (column-major) layout.
    fn f_contiguous(dims: &[usize]) -> Self {
        let n = dims.len();
        let mut strides = vec![0usize; n];
        if n > 0 {
            strides[0] = 1;
            for i in 1..n {
                strides[i] = strides[i - 1].saturating_mul(dims[i - 1]);
            }
        }
        Self { dims: dims.to_vec(), strides }
    }

    fn ndim(&self) -> usize {
        self.dims.len()
    }

    fn total_elements(&self) -> usize {
        self.dims.iter().copied().fold(1usize, |a, b| a.saturating_mul(b))
    }

    /// Check if layout is C-contiguous.
    fn is_c_contiguous(&self) -> bool {
        let n = self.ndim();
        if n == 0 {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..n).rev() {
            if self.dims[i] == 0 {
                return true; // Zero-sized dims are trivially contiguous.
            }
            if self.strides[i] != expected {
                return false;
            }
            expected = expected.saturating_mul(self.dims[i]);
        }
        true
    }

    /// Check if layout is F-contiguous.
    fn is_f_contiguous(&self) -> bool {
        let n = self.ndim();
        if n == 0 {
            return true;
        }
        let mut expected = 1usize;
        for i in 0..n {
            if self.dims[i] == 0 {
                return true;
            }
            if self.strides[i] != expected {
                return false;
            }
            expected = expected.saturating_mul(self.dims[i]);
        }
        true
    }

    /// Compute the minimum buffer size needed for this layout.
    fn min_buffer_size(&self) -> usize {
        if self.dims.iter().any(|&d| d == 0) {
            return 0;
        }
        self.dims
            .iter()
            .zip(self.strides.iter())
            .map(|(&d, &s)| if d == 0 { 0 } else { (d - 1).saturating_mul(s) })
            .fold(0usize, |a, b| a.saturating_add(b))
            .saturating_add(1)
    }
}

fuzz_target!(|input: LayoutInput| {
    // Clamp dims to avoid timeout: up to 8 dimensions, each ≤ 128.
    let dims: Vec<usize> = input.dims.iter().take(8).map(|&d| (d as usize) % 129).collect();

    if dims.is_empty() {
        return;
    }

    let layout = if input.fortran_order {
        TensorLayout::f_contiguous(&dims)
    } else {
        TensorLayout::c_contiguous(&dims)
    };

    // Invariant 1: Strides length matches dims length.
    assert_eq!(
        layout.strides.len(),
        layout.dims.len(),
        "strides length {} != dims length {}",
        layout.strides.len(),
        layout.dims.len()
    );

    // Invariant 2: Total elements = product of dimensions.
    let product: usize = dims.iter().copied().fold(1usize, |a, b| a.saturating_mul(b));
    assert_eq!(layout.total_elements(), product, "total_elements mismatch");

    // Invariant 3: Contiguity checks are consistent.
    if !input.fortran_order {
        assert!(layout.is_c_contiguous(), "C-contiguous layout failed contiguity check");
    } else {
        assert!(layout.is_f_contiguous(), "F-contiguous layout failed contiguity check");
    }

    // Invariant 4: For 1-D tensors, C and F contiguity are equivalent.
    if dims.len() == 1 {
        let c = TensorLayout::c_contiguous(&dims);
        let f = TensorLayout::f_contiguous(&dims);
        assert_eq!(c.strides, f.strides, "1-D C and F strides should match");
    }

    // Invariant 5: min_buffer_size >= total_elements for contiguous layouts.
    if !dims.contains(&0) {
        let buf = layout.min_buffer_size();
        let total = layout.total_elements();
        assert!(
            buf >= total,
            "min_buffer_size {buf} < total_elements {total} for contiguous layout"
        );
    }

    // Invariant 6: Zero-dim tensors have zero total elements.
    if dims.contains(&0) {
        assert_eq!(layout.total_elements(), 0, "zero-dim tensor should have 0 elements");
        assert_eq!(layout.min_buffer_size(), 0, "zero-dim tensor needs 0 buffer");
    }

    // Invariant 7: Strides are non-zero for non-zero dimensions (contiguous layout).
    for (i, (&d, &s)) in dims.iter().zip(layout.strides.iter()).enumerate() {
        if d > 1 {
            assert!(s > 0, "stride at dim {i} should be >0 for dim size {d}");
        }
    }
});
