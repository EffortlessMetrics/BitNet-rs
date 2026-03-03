//! Tensor serialization and deserialization.
//!
//! Save and load tensor data in a simple binary format
//! for caching, checkpointing, and testing.

use std::io::{self, Read, Write};

/// Magic bytes for the tensor file format.
const MAGIC: &[u8; 4] = b"BTNS";
/// Format version.
const VERSION: u8 = 1;

/// Data type tag for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SerialDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
    I32 = 5,
}

impl SerialDType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::BF16),
            3 => Some(Self::I8),
            4 => Some(Self::U8),
            5 => Some(Self::I32),
            _ => None,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            SerialDType::F32 | SerialDType::I32 => 4,
            SerialDType::F16 | SerialDType::BF16 => 2,
            SerialDType::I8 | SerialDType::U8 => 1,
        }
    }
}

/// Header for a serialized tensor.
#[derive(Debug, Clone)]
pub struct TensorHeader {
    pub name: String,
    pub dtype: SerialDType,
    pub shape: Vec<usize>,
}

impl TensorHeader {
    pub fn new(name: impl Into<String>, dtype: SerialDType, shape: Vec<usize>) -> Self {
        Self { name: name.into(), dtype, shape }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data_size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Write a tensor header to a writer.
pub fn write_header<W: Write>(w: &mut W, header: &TensorHeader) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&[VERSION])?;
    w.write_all(&[header.dtype as u8])?;

    // Name length + name
    let name_bytes = header.name.as_bytes();
    let name_len = name_bytes.len() as u16;
    w.write_all(&name_len.to_le_bytes())?;
    w.write_all(name_bytes)?;

    // Number of dimensions + shape
    let ndim = header.shape.len() as u8;
    w.write_all(&[ndim])?;
    for &dim in &header.shape {
        w.write_all(&(dim as u64).to_le_bytes())?;
    }

    Ok(())
}

/// Read a tensor header from a reader.
pub fn read_header<R: Read>(r: &mut R) -> io::Result<TensorHeader> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic bytes"));
    }

    let mut ver = [0u8; 1];
    r.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported version"));
    }

    let mut dtype_byte = [0u8; 1];
    r.read_exact(&mut dtype_byte)?;
    let dtype = SerialDType::from_u8(dtype_byte[0])
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown dtype"))?;

    let mut name_len_bytes = [0u8; 2];
    r.read_exact(&mut name_len_bytes)?;
    let name_len = u16::from_le_bytes(name_len_bytes) as usize;
    let mut name_buf = vec![0u8; name_len];
    r.read_exact(&mut name_buf)?;
    let name =
        String::from_utf8(name_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut ndim = [0u8; 1];
    r.read_exact(&mut ndim)?;
    let ndim = ndim[0] as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 8];
        r.read_exact(&mut dim_bytes)?;
        shape.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    Ok(TensorHeader { name, dtype, shape })
}

/// Write f32 tensor data (header + raw bytes).
pub fn write_f32_tensor<W: Write>(
    w: &mut W,
    name: &str,
    shape: &[usize],
    data: &[f32],
) -> io::Result<()> {
    let header = TensorHeader::new(name, SerialDType::F32, shape.to_vec());
    assert_eq!(header.num_elements(), data.len(), "shape/data mismatch");
    write_header(w, &header)?;
    for &val in data {
        w.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

/// Read f32 tensor data.
pub fn read_f32_tensor<R: Read>(r: &mut R) -> io::Result<(TensorHeader, Vec<f32>)> {
    let header = read_header(r)?;
    if header.dtype != SerialDType::F32 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected F32 dtype"));
    }
    let n = header.num_elements();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        data.push(f32::from_le_bytes(buf));
    }
    Ok((header, data))
}

/// Compute a simple checksum for data integrity.
pub fn checksum(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5; // FNV-1a offset basis
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x01000193); // FNV prime
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_roundtrip() {
        let header = TensorHeader::new("test.weight", SerialDType::F32, vec![3, 4]);
        let mut buf = Vec::new();
        write_header(&mut buf, &header).unwrap();
        let mut cursor = Cursor::new(&buf);
        let read = read_header(&mut cursor).unwrap();
        assert_eq!(read.name, "test.weight");
        assert_eq!(read.dtype, SerialDType::F32);
        assert_eq!(read.shape, vec![3, 4]);
    }

    #[test]
    fn test_f32_tensor_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = Vec::new();
        write_f32_tensor(&mut buf, "layer.0.weight", &[2, 3], &data).unwrap();
        let mut cursor = Cursor::new(&buf);
        let (header, read_data) = read_f32_tensor(&mut cursor).unwrap();
        assert_eq!(header.name, "layer.0.weight");
        assert_eq!(header.shape, vec![2, 3]);
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_num_elements() {
        let h = TensorHeader::new("t", SerialDType::F32, vec![2, 3, 4]);
        assert_eq!(h.num_elements(), 24);
    }

    #[test]
    fn test_data_size_bytes() {
        let h = TensorHeader::new("t", SerialDType::F32, vec![10]);
        assert_eq!(h.data_size_bytes(), 40);
        let h2 = TensorHeader::new("t", SerialDType::F16, vec![10]);
        assert_eq!(h2.data_size_bytes(), 20);
    }

    #[test]
    fn test_serial_dtype_from_u8() {
        assert_eq!(SerialDType::from_u8(0), Some(SerialDType::F32));
        assert_eq!(SerialDType::from_u8(5), Some(SerialDType::I32));
        assert_eq!(SerialDType::from_u8(99), None);
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"XXXX\x01\x00";
        let mut cursor = Cursor::new(data);
        assert!(read_header(&mut cursor).is_err());
    }

    #[test]
    fn test_invalid_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(99); // bad version
        let mut cursor = Cursor::new(&buf);
        assert!(read_header(&mut cursor).is_err());
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = b"hello world";
        let c1 = checksum(data);
        let c2 = checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_checksum_different() {
        let c1 = checksum(b"hello");
        let c2 = checksum(b"world");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_empty_tensor() {
        let data: Vec<f32> = vec![];
        let mut buf = Vec::new();
        write_f32_tensor(&mut buf, "empty", &[0], &data).unwrap();
        let mut cursor = Cursor::new(&buf);
        let (h, d) = read_f32_tensor(&mut cursor).unwrap();
        assert_eq!(h.num_elements(), 0);
        assert!(d.is_empty());
    }

    #[test]
    fn test_scalar_tensor() {
        let data = vec![42.0f32];
        let mut buf = Vec::new();
        write_f32_tensor(&mut buf, "scalar", &[1], &data).unwrap();
        let mut cursor = Cursor::new(&buf);
        let (_, d) = read_f32_tensor(&mut cursor).unwrap();
        assert_eq!(d, vec![42.0]);
    }

    #[test]
    fn test_ndim() {
        let h = TensorHeader::new("t", SerialDType::F32, vec![2, 3, 4, 5]);
        assert_eq!(h.ndim(), 4);
    }
}
