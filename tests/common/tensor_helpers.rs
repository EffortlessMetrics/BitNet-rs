use bitnet_common::{ConcreteTensor, MockTensor};

/// Build a mock concrete tensor with the given shape.
/// Usage: `ct(vec![1, vocab])` or `ctv![1, vocab]`
pub fn ct(shape: Vec<usize>) -> ConcreteTensor {
    ConcreteTensor::Mock(MockTensor::new(shape))
}

/// Convenience macro: `ctv![1, vocab_size]`
#[macro_export]
macro_rules! ctv {
    ($($dim:expr),+ $(,)?) => {{
        $crate::common::tensor_helpers::ct(vec![$($dim as usize),+])
    }};
}
