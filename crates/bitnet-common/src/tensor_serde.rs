//! Tensor serialization helpers.
//!
//! Utilities for converting tensors to/from byte representations.

/// Write f32 slice to little-endian bytes.
pub fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Read f32 slice from little-endian bytes.
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Write f16 (as u16) to little-endian bytes.
pub fn f16_to_bytes(data: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Read f16 (as u16) from little-endian bytes.
pub fn bytes_to_f16(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]])).collect()
}

/// Compute a simple checksum for tensor data (XOR-based).
pub fn tensor_checksum(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Validate tensor data integrity with a checksum.
pub fn verify_checksum(data: &[u8], expected: u64) -> bool {
    tensor_checksum(data) == expected
}

/// Pack a shape array into bytes (u64 little-endian per dimension).
pub fn pack_shape(shape: &[usize]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(shape.len() * 8);
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    bytes
}

/// Unpack a shape from bytes.
pub fn unpack_shape(bytes: &[u8]) -> Vec<usize> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as usize
        })
        .collect()
}

/// Header for a serialized tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorHeader {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: u8, // 0=f32, 1=f16, 2=bf16, 3=i8, 4=i4
    pub data_offset: u64,
    pub data_length: u64,
}

impl TensorHeader {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dtype_name(&self) -> &'static str {
        match self.dtype {
            0 => "f32",
            1 => "f16",
            2 => "bf16",
            3 => "i8",
            4 => "i4",
            _ => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_roundtrip() {
        let data = vec![1.0f32, -2.5, std::f32::consts::PI, 0.0];
        let bytes = f32_to_bytes(&data);
        let back = bytes_to_f32(&bytes);
        assert_eq!(data, back);
    }

    #[test]
    fn test_f32_empty() {
        let bytes = f32_to_bytes(&[]);
        assert!(bytes.is_empty());
        assert!(bytes_to_f32(&[]).is_empty());
    }

    #[test]
    fn test_f16_roundtrip() {
        let data: Vec<u16> = vec![0x3C00, 0x4000, 0x0000]; // 1.0, 2.0, 0.0 in f16
        let bytes = f16_to_bytes(&data);
        let back = bytes_to_f16(&bytes);
        assert_eq!(data, back);
    }

    #[test]
    fn test_f16_empty() {
        assert!(f16_to_bytes(&[]).is_empty());
        assert!(bytes_to_f16(&[]).is_empty());
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = b"hello tensor";
        let c1 = tensor_checksum(data);
        let c2 = tensor_checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_checksum_different() {
        let c1 = tensor_checksum(b"hello");
        let c2 = tensor_checksum(b"world");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_verify_checksum() {
        let data = b"test data";
        let cs = tensor_checksum(data);
        assert!(verify_checksum(data, cs));
        assert!(!verify_checksum(data, cs + 1));
    }

    #[test]
    fn test_shape_roundtrip() {
        let shape = vec![3, 1024, 768];
        let bytes = pack_shape(&shape);
        let back = unpack_shape(&bytes);
        assert_eq!(shape, back);
    }

    #[test]
    fn test_shape_empty() {
        let bytes = pack_shape(&[]);
        let back = unpack_shape(&bytes);
        assert!(back.is_empty());
    }

    #[test]
    fn test_header_elements() {
        let h = TensorHeader {
            name: "weight".into(),
            shape: vec![10, 20, 30],
            dtype: 0,
            data_offset: 0,
            data_length: 24000,
        };
        assert_eq!(h.elements(), 6000);
    }

    #[test]
    fn test_header_dtype_name() {
        let h = TensorHeader {
            name: "w".into(),
            shape: vec![1],
            dtype: 0,
            data_offset: 0,
            data_length: 4,
        };
        assert_eq!(h.dtype_name(), "f32");
        let h2 = TensorHeader {
            name: "w".into(),
            shape: vec![1],
            dtype: 1,
            data_offset: 0,
            data_length: 2,
        };
        assert_eq!(h2.dtype_name(), "f16");
    }

    #[test]
    fn test_f32_byte_count() {
        let bytes = f32_to_bytes(&[1.0, 2.0, 3.0]);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn test_f16_byte_count() {
        let bytes = f16_to_bytes(&[1, 2, 3]);
        assert_eq!(bytes.len(), 6);
    }

    #[test]
    fn test_checksum_empty() {
        let cs = tensor_checksum(&[]);
        assert_ne!(cs, 0); // FNV offset basis
    }
}
