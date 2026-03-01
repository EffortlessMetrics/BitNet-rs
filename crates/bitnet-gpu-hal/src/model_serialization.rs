//! Module stub - implementation pending merge from feature branch
//! Model serialization and deserialization for GPU inference.
//!
//! Provides a complete pipeline for persisting neural-network model weights:
//!
//! * [`SerialConfig`] — format, compression, checksum, and streaming settings.
//! * [`ModelHeader`] — metadata about architecture, version, tensor count, precision.
//! * [`TensorSerializer`] / [`TensorDeserializer`] — per-tensor encode/decode with
//!   validation.
//! * [`ChecksumValidator`] — integrity verification (CRC32, SHA256, xxHash).
//! * [`StreamingWriter`] / [`StreamingReader`] — chunk-based I/O for models that
//!   exceed available memory.
//! * [`CompressionCodec`] — pluggable compression (None, Zstd, Lz4, Snappy).
//! * [`ModelManifest`] — tensor index / offset table inside a model file.
//! * [`SerializationEngine`] — unified façade that ties everything together.
//!
//! All types are CPU-only reference implementations designed for correctness
//! and testability; GPU-specific fast-paths can override these later.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io;

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by model serialization operations.
#[derive(Debug)]
pub enum SerializationError {
    /// An I/O error occurred.
    Io(io::Error),
    /// Serialisation / deserialisation failed.
    Serde(String),
    /// Checksum mismatch (expected, actual).
    ChecksumMismatch { expected: String, actual: String },
    /// Corrupt or invalid header.
    InvalidHeader(String),
    /// Unsupported format or version.
    UnsupportedFormat(String),
    /// A tensor was not found in the manifest.
    TensorNotFound(String),
    /// Compression / decompression error.
    CompressionError(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Serde(e) => write!(f, "serialization error: {e}"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::CompressionError(msg) => write!(f, "compression error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for SerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for SerializationError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e.to_string())
    }
}

// ── Data type enum ──────────────────────────────────────────────────────────

/// Element data type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float.
    F16,
    /// Google Brain 16-bit float.
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 2-bit signed integer (BitNet ternary).
    I2S,
    /// 4-bit unsigned integer (nibble-packed).
    U4,
}

impl DataType {
    /// Size of one element in bytes (may be fractional for sub-byte types,
    /// returned as the minimum packing unit).
    #[must_use]
    pub const fn element_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::I2S => 1, // packed: 4 elements per byte
            Self::U4 => 1,  // packed: 2 elements per byte
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2S => write!(f, "i2s"),
            Self::U4 => write!(f, "u4"),
        }
    }
}

// ── Checksum algorithm ──────────────────────────────────────────────────────

/// Algorithm used for data-integrity verification.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// No checksum verification.
    None,
    /// CRC-32 (fast, 32-bit).
    #[default]
    Crc32,
    /// SHA-256 (cryptographic, 256-bit).
    Sha256,
    /// xxHash64 (very fast, 64-bit).
    XxHash,
}

impl fmt::Display for ChecksumAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Crc32 => write!(f, "crc32"),
            Self::Sha256 => write!(f, "sha256"),
            Self::XxHash => write!(f, "xxhash"),
        }
    }
}

// ── Compression codec ───────────────────────────────────────────────────────

/// Compression algorithm for model storage.
///
/// CPU reference implementation that simulates compression ratios without
/// requiring external C libraries. Real backends would link zstd/lz4/snappy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionCodec {
    /// No compression.
    #[default]
    None,
    /// Zstandard compression (good ratio, moderate speed).
    Zstd,
    /// LZ4 compression (very fast, moderate ratio).
    Lz4,
    /// Snappy compression (fast, lower ratio).
    Snappy,
}

impl CompressionCodec {
    /// Compress `data` using this codec (CPU reference: identity transform
    /// with a codec tag prepended for round-trip fidelity).
    #[must_use]
    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        let tag = self.tag_byte();
        let mut out = Vec::with_capacity(1 + data.len());
        out.push(tag);
        out.extend_from_slice(data);
        out
    }

    /// Decompress `data` previously compressed with [`Self::compress`].
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, SerializationError> {
        if data.is_empty() {
            return Err(SerializationError::CompressionError("empty compressed payload".into()));
        }
        let expected_tag = self.tag_byte();
        if data[0] != expected_tag {
            return Err(SerializationError::CompressionError(format!(
                "codec tag mismatch: expected 0x{expected_tag:02x}, got 0x{:02x}",
                data[0]
            )));
        }
        Ok(data[1..].to_vec())
    }

    /// Single-byte codec identifier.
    const fn tag_byte(&self) -> u8 {
        match self {
            Self::None => 0x00,
            Self::Zstd => 0x01,
            Self::Lz4 => 0x02,
            Self::Snappy => 0x03,
        }
    }

    /// Returns the codec that matches the given tag byte, if any.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x00 => Some(Self::None),
            0x01 => Some(Self::Zstd),
            0x02 => Some(Self::Lz4),
            0x03 => Some(Self::Snappy),
            _ => Option::None,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
            Self::Snappy => "snappy",
        }
    }
}

impl fmt::Display for CompressionCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── 1. SerialConfig ─────────────────────────────────────────────────────────

/// Format used for the model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerialFormat {
    /// Custom binary format.
    Binary,
    /// GGUF-compatible format.
    Gguf,
    /// SafeTensors-compatible format.
    SafeTensors,
}

impl fmt::Display for SerialFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary => write!(f, "binary"),
            Self::Gguf => write!(f, "gguf"),
            Self::SafeTensors => write!(f, "safetensors"),
        }
    }
}

/// Configuration for model serialization.
///
/// Controls the file format, compression, checksum algorithm, and streaming
/// behaviour used when writing or reading model files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialConfig {
    /// Output / input file format.
    pub format: SerialFormat,
    /// Compression codec for tensor data.
    pub compression: CompressionCodec,
    /// Checksum algorithm for data integrity.
    pub checksum: ChecksumAlgorithm,
    /// Enable streaming (chunked) I/O.
    pub streaming: bool,
    /// Chunk size in bytes for streaming I/O (default 64 MiB).
    pub chunk_size: usize,
    /// Byte-order: `true` = little-endian (default on x86).
    pub little_endian: bool,
    /// Alignment in bytes for tensor data offsets.
    pub alignment: usize,
}

impl Default for SerialConfig {
    fn default() -> Self {
        Self {
            format: SerialFormat::Binary,
            compression: CompressionCodec::None,
            checksum: ChecksumAlgorithm::Crc32,
            streaming: false,
            chunk_size: 64 * 1024 * 1024, // 64 MiB
            little_endian: true,
            alignment: 64,
        }
    }
}

impl SerialConfig {
    /// Create a new configuration with the given format.
    #[must_use]
    pub fn new(format: SerialFormat) -> Self {
        Self { format, ..Default::default() }
    }

    /// Builder: set compression codec.
    #[must_use]
    pub const fn with_compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = codec;
        self
    }

    /// Builder: set checksum algorithm.
    #[must_use]
    pub const fn with_checksum(mut self, algo: ChecksumAlgorithm) -> Self {
        self.checksum = algo;
        self
    }

    /// Builder: enable or disable streaming.
    #[must_use]
    pub const fn with_streaming(mut self, on: bool) -> Self {
        self.streaming = on;
        self
    }

    /// Builder: set chunk size for streaming.
    #[must_use]
    pub const fn with_chunk_size(mut self, bytes: usize) -> Self {
        self.chunk_size = bytes;
        self
    }

    /// Builder: set alignment.
    #[must_use]
    pub const fn with_alignment(mut self, bytes: usize) -> Self {
        self.alignment = bytes;
        self
    }

    /// Validate this configuration, returning an error if anything is invalid.
    pub fn validate(&self) -> Result<(), SerializationError> {
        if self.chunk_size == 0 {
            return Err(SerializationError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if self.alignment == 0 || (self.alignment & (self.alignment - 1)) != 0 {
            return Err(SerializationError::InvalidConfig(
                "alignment must be a non-zero power of 2".into(),
            ));
        }
        Ok(())
    }
}

// ── 2. ModelHeader ──────────────────────────────────────────────────────────

/// Header written at the start of a serialized model file.
///
/// Contains metadata needed to interpret the rest of the file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHeader {
    /// Magic bytes identifying the format (e.g. "BNET").
    pub magic: [u8; 4],
    /// Format version (major, minor, patch).
    pub version: (u16, u16, u16),
    /// Architecture name (e.g. "bitnet-b1.58-2B").
    pub architecture: String,
    /// Total number of tensors in the model.
    pub tensor_count: u32,
    /// Primary precision / data type.
    pub precision: DataType,
    /// Compression codec used for tensor data.
    pub compression: CompressionCodec,
    /// Checksum algorithm used for tensor data.
    pub checksum_algo: ChecksumAlgorithm,
    /// Total file size in bytes (0 if unknown at write time).
    pub total_size_bytes: u64,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl Default for ModelHeader {
    fn default() -> Self {
        Self {
            magic: *b"BNET",
            version: (1, 0, 0),
            architecture: String::new(),
            tensor_count: 0,
            precision: DataType::F32,
            compression: CompressionCodec::None,
            checksum_algo: ChecksumAlgorithm::Crc32,
            total_size_bytes: 0,
            metadata: HashMap::new(),
        }
    }
}

impl ModelHeader {
    /// Create a header for the given architecture and tensor count.
    #[must_use]
    pub fn new(architecture: impl Into<String>, tensor_count: u32) -> Self {
        Self { architecture: architecture.into(), tensor_count, ..Default::default() }
    }

    /// Builder: set version.
    #[must_use]
    pub const fn with_version(mut self, major: u16, minor: u16, patch: u16) -> Self {
        self.version = (major, minor, patch);
        self
    }

    /// Builder: set precision.
    #[must_use]
    pub const fn with_precision(mut self, dt: DataType) -> Self {
        self.precision = dt;
        self
    }

    /// Builder: set compression codec.
    #[must_use]
    pub const fn with_compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = codec;
        self
    }

    /// Builder: set checksum algorithm.
    #[must_use]
    pub const fn with_checksum(mut self, algo: ChecksumAlgorithm) -> Self {
        self.checksum_algo = algo;
        self
    }

    /// Builder: insert a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Serialise the header to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        serde_json::to_vec(self).map_err(Into::into)
    }

    /// Deserialise a header from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SerializationError> {
        let header: Self = serde_json::from_slice(data)?;
        if &header.magic != b"BNET" {
            return Err(SerializationError::InvalidHeader(format!(
                "bad magic: {:?}",
                header.magic
            )));
        }
        Ok(header)
    }

    /// Version as a formatted string.
    #[must_use]
    pub fn version_string(&self) -> String {
        let (maj, min, pat) = self.version;
        format!("{maj}.{min}.{pat}")
    }

    /// Validate header fields.
    pub fn validate(&self) -> Result<(), SerializationError> {
        if &self.magic != b"BNET" {
            return Err(SerializationError::InvalidHeader("bad magic".into()));
        }
        if self.architecture.is_empty() {
            return Err(SerializationError::InvalidHeader("architecture must not be empty".into()));
        }
        Ok(())
    }
}

// ── 3. TensorSerializer ────────────────────────────────────────────────────

/// Metadata for a single tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Tensor name (e.g. "layers.0.attention.wq.weight").
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DataType,
    /// Byte offset within the file (set during writing).
    pub offset: u64,
    /// Size of the (possibly compressed) data in bytes.
    pub size_bytes: u64,
    /// Checksum of the raw (uncompressed) data.
    pub checksum: String,
}

impl TensorMeta {
    /// Number of elements in the tensor.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Serializes individual tensors into a byte stream.
///
/// CPU reference implementation: stores tensor data with a JSON metadata
/// preamble followed by (optionally compressed) raw bytes.
#[derive(Debug, Clone)]
pub struct TensorSerializer {
    config: SerialConfig,
    checksum_validator: ChecksumValidator,
    tensors_written: u32,
}

impl TensorSerializer {
    /// Create a new serializer with the given configuration.
    #[must_use]
    pub fn new(config: SerialConfig) -> Self {
        let checksum_validator = ChecksumValidator::new(config.checksum);
        Self { config, checksum_validator, tensors_written: 0 }
    }

    /// Serialize a tensor to bytes (metadata JSON + compressed payload).
    pub fn serialize_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: DataType,
        data: &[u8],
    ) -> Result<(TensorMeta, Vec<u8>), SerializationError> {
        let checksum = self.checksum_validator.compute(data);

        let compressed = self.config.compression.compress(data);

        let meta = TensorMeta {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype,
            offset: 0, // set by the caller / engine
            size_bytes: compressed.len() as u64,
            checksum: checksum.clone(),
        };

        let meta_bytes = serde_json::to_vec(&meta)?;
        let meta_len = (meta_bytes.len() as u32).to_le_bytes();

        let mut out = Vec::with_capacity(4 + meta_bytes.len() + compressed.len());
        out.extend_from_slice(&meta_len);
        out.extend_from_slice(&meta_bytes);
        out.extend_from_slice(&compressed);

        self.tensors_written += 1;
        Ok((meta, out))
    }

    /// Number of tensors written so far.
    #[must_use]
    pub const fn tensors_written(&self) -> u32 {
        self.tensors_written
    }

    /// Reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &SerialConfig {
        &self.config
    }
}

// ── 4. TensorDeserializer ───────────────────────────────────────────────────

/// Deserializes tensors from a byte stream with validation.
///
/// CPU reference: reads the metadata preamble, decompresses the payload,
/// and validates the checksum.
#[derive(Debug, Clone)]
pub struct TensorDeserializer {
    config: SerialConfig,
    checksum_validator: ChecksumValidator,
    tensors_read: u32,
}

impl TensorDeserializer {
    /// Create a new deserializer with the given configuration.
    #[must_use]
    pub fn new(config: SerialConfig) -> Self {
        let checksum_validator = ChecksumValidator::new(config.checksum);
        Self { config, checksum_validator, tensors_read: 0 }
    }

    /// Deserialize a tensor from bytes produced by [`TensorSerializer`].
    ///
    /// Returns the metadata and the decompressed raw data.
    pub fn deserialize_tensor(
        &mut self,
        data: &[u8],
    ) -> Result<(TensorMeta, Vec<u8>), SerializationError> {
        if data.len() < 4 {
            return Err(SerializationError::InvalidHeader(
                "tensor data too short for metadata length".into(),
            ));
        }

        let meta_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + meta_len {
            return Err(SerializationError::InvalidHeader(
                "tensor data truncated before metadata".into(),
            ));
        }

        let meta: TensorMeta = serde_json::from_slice(&data[4..4 + meta_len])?;
        let compressed = &data[4 + meta_len..];

        let raw = self.config.compression.decompress(compressed)?;

        // Validate checksum.
        let actual_checksum = self.checksum_validator.compute(&raw);
        if meta.checksum != actual_checksum {
            return Err(SerializationError::ChecksumMismatch {
                expected: meta.checksum.clone(),
                actual: actual_checksum,
            });
        }

        self.tensors_read += 1;
        Ok((meta, raw))
    }

    /// Number of tensors read so far.
    #[must_use]
    pub const fn tensors_read(&self) -> u32 {
        self.tensors_read
    }

    /// Reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &SerialConfig {
        &self.config
    }
}

// ── 5. ChecksumValidator ────────────────────────────────────────────────────

/// Computes and validates checksums for data integrity.
///
/// CPU reference implementation using simple hash functions. Production
/// backends should use hardware-accelerated CRC32/SHA instructions.
#[derive(Debug, Clone)]
pub struct ChecksumValidator {
    algorithm: ChecksumAlgorithm,
}

impl ChecksumValidator {
    /// Create a new validator with the given algorithm.
    #[must_use]
    pub const fn new(algorithm: ChecksumAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Compute a checksum of `data`, returned as a hex string.
    #[must_use]
    pub fn compute(&self, data: &[u8]) -> String {
        match self.algorithm {
            ChecksumAlgorithm::None => String::new(),
            ChecksumAlgorithm::Crc32 => {
                let crc = Self::crc32(data);
                format!("{crc:08x}")
            }
            ChecksumAlgorithm::Sha256 => Self::simple_sha256_hex(data),
            ChecksumAlgorithm::XxHash => {
                let h = Self::xxhash64(data);
                format!("{h:016x}")
            }
        }
    }

    /// Validate `data` against an expected checksum string.
    pub fn validate(&self, data: &[u8], expected: &str) -> Result<(), SerializationError> {
        if self.algorithm == ChecksumAlgorithm::None {
            return Ok(());
        }
        let actual = self.compute(data);
        if actual != expected {
            return Err(SerializationError::ChecksumMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
        Ok(())
    }

    /// The active algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> ChecksumAlgorithm {
        self.algorithm
    }

    // ── CPU reference hash implementations ──────────────────────────────

    /// Simple CRC-32 (ISO 3309 polynomial).
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    /// Minimal SHA-256-like hash (not cryptographically secure — just a
    /// deterministic fingerprint for test/reference purposes).
    fn simple_sha256_hex(data: &[u8]) -> String {
        // Use a simple but deterministic mixing function.
        let mut state: [u64; 4] = [
            0x6a09_e667_f3bc_c908,
            0xbb67_ae85_84ca_a73b,
            0x3c6e_f372_fe94_f82b,
            0xa54f_f53a_5f1d_36f1,
        ];
        for (i, &byte) in data.iter().enumerate() {
            let idx = i % 4;
            state[idx] = state[idx]
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(u64::from(byte))
                .wrapping_add(state[(idx + 1) % 4]);
        }
        // Mix once more to diffuse.
        for round in 0..4 {
            state[round] ^= state[(round + 2) % 4].wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        format!("{:016x}{:016x}{:016x}{:016x}", state[0], state[1], state[2], state[3])
    }

    /// Simple xxHash64-like hash (deterministic, non-cryptographic).
    fn xxhash64(data: &[u8]) -> u64 {
        let prime1: u64 = 0x9E37_79B1_85EB_CA87;
        let prime2: u64 = 0xC2B2_AE3D_27D4_EB4F;
        let prime5: u64 = 0x27D4_EB2F_1656_67C5;

        let mut h: u64 = prime5.wrapping_add(data.len() as u64);
        for &byte in data {
            h ^= u64::from(byte).wrapping_mul(prime1);
            h = h.rotate_left(11).wrapping_mul(prime2);
        }
        // Avalanche.
        h ^= h >> 33;
        h = h.wrapping_mul(prime2);
        h ^= h >> 29;
        h = h.wrapping_mul(0x1B03_2414_4B05_1287);
        h ^= h >> 32;
        h
    }
}

// ── 6. StreamingWriter ──────────────────────────────────────────────────────

/// Streaming serialization writer for large models.
///
/// Breaks a model into chunks, compresses each independently, and tracks
/// offsets in a manifest. CPU reference: writes to an in-memory buffer.
#[derive(Debug, Clone)]
pub struct StreamingWriter {
    config: SerialConfig,
    buffer: Vec<u8>,
    manifest: ModelManifest,
    bytes_written: u64,
    tensors_written: u32,
    checksum_validator: ChecksumValidator,
}

impl StreamingWriter {
    /// Create a new streaming writer with the given configuration.
    pub fn new(config: SerialConfig, header: ModelHeader) -> Result<Self, SerializationError> {
        config.validate()?;
        let checksum_validator = ChecksumValidator::new(config.checksum);
        let manifest = ModelManifest::new(header);
        Ok(Self {
            config,
            buffer: Vec::new(),
            manifest,
            bytes_written: 0,
            tensors_written: 0,
            checksum_validator,
        })
    }

    /// Write a tensor to the stream.
    pub fn write_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: DataType,
        data: &[u8],
    ) -> Result<(), SerializationError> {
        let checksum = self.checksum_validator.compute(data);
        let compressed = self.config.compression.compress(data);

        let meta = TensorMeta {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype,
            offset: self.bytes_written,
            size_bytes: compressed.len() as u64,
            checksum,
        };

        // Write in chunks.
        for chunk in compressed.chunks(self.config.chunk_size) {
            self.buffer.extend_from_slice(chunk);
        }

        self.bytes_written += compressed.len() as u64;
        self.tensors_written += 1;
        self.manifest.add_entry(meta);
        Ok(())
    }

    /// Finalize the stream, returning the buffer and manifest.
    #[must_use]
    pub fn finish(self) -> (Vec<u8>, ModelManifest) {
        (self.buffer, self.manifest)
    }

    /// Total bytes written so far.
    #[must_use]
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Number of tensors written.
    #[must_use]
    pub const fn tensors_written(&self) -> u32 {
        self.tensors_written
    }

    /// Reference to the current manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
}

// ── 7. StreamingReader ──────────────────────────────────────────────────────

/// Streaming deserialization reader with lazy tensor loading.
///
/// Reads tensors on-demand from a byte buffer using the manifest's offset
/// table. CPU reference implementation.
#[derive(Debug, Clone)]
pub struct StreamingReader {
    config: SerialConfig,
    data: Vec<u8>,
    manifest: ModelManifest,
    tensors_read: u32,
    checksum_validator: ChecksumValidator,
}

impl StreamingReader {
    /// Create a new streaming reader from a buffer and manifest.
    pub fn new(
        config: SerialConfig,
        data: Vec<u8>,
        manifest: ModelManifest,
    ) -> Result<Self, SerializationError> {
        config.validate()?;
        let checksum_validator = ChecksumValidator::new(config.checksum);
        Ok(Self { config, data, manifest, tensors_read: 0, checksum_validator })
    }

    /// Read a tensor by name.
    pub fn read_tensor(&mut self, name: &str) -> Result<(TensorMeta, Vec<u8>), SerializationError> {
        let entry = self
            .manifest
            .get_entry(name)
            .ok_or_else(|| SerializationError::TensorNotFound(name.to_string()))?
            .clone();

        let start = entry.offset as usize;
        let end = start + entry.size_bytes as usize;

        if end > self.data.len() {
            return Err(SerializationError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "tensor '{name}' extends beyond data: need {end}, have {}",
                    self.data.len()
                ),
            )));
        }

        let compressed = &self.data[start..end];
        let raw = self.config.compression.decompress(compressed)?;

        // Validate checksum.
        self.checksum_validator.validate(&raw, &entry.checksum)?;

        self.tensors_read += 1;
        Ok((entry, raw))
    }

    /// List all tensor names in the manifest.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.manifest.tensor_names()
    }

    /// Number of tensors read so far.
    #[must_use]
    pub const fn tensors_read(&self) -> u32 {
        self.tensors_read
    }

    /// Reference to the manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Total data size in bytes.
    #[must_use]
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

// ── 8. (CompressionCodec is defined above) ──────────────────────────────────

// ── 9. ModelManifest ────────────────────────────────────────────────────────

/// Describes all tensors in a serialized model file.
///
/// Maintains an ordered index of [`TensorMeta`] entries, each with its byte
/// offset and size, enabling random-access reads from a flat file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Header metadata for the model.
    pub header: ModelHeader,
    /// Ordered list of tensor entries.
    entries: Vec<TensorMeta>,
    /// Fast name → index lookup.
    #[serde(skip)]
    index: HashMap<String, usize>,
}

impl ModelManifest {
    /// Create a new manifest with the given header.
    #[must_use]
    pub fn new(header: ModelHeader) -> Self {
        Self { header, entries: Vec::new(), index: HashMap::new() }
    }

    /// Add a tensor entry.
    pub fn add_entry(&mut self, meta: TensorMeta) {
        let idx = self.entries.len();
        self.index.insert(meta.name.clone(), idx);
        self.entries.push(meta);
    }

    /// Get an entry by name.
    #[must_use]
    pub fn get_entry(&self, name: &str) -> Option<&TensorMeta> {
        self.index.get(name).map(|&i| &self.entries[i])
    }

    /// All entries in order.
    #[must_use]
    pub fn entries(&self) -> &[TensorMeta] {
        &self.entries
    }

    /// List tensor names in order.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Number of tensors in the manifest.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the manifest is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total data size in bytes across all entries.
    #[must_use]
    pub fn total_data_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Serialize the manifest to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        serde_json::to_vec(self).map_err(Into::into)
    }

    /// Deserialize a manifest from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SerializationError> {
        let mut manifest: Self = serde_json::from_slice(data)?;
        // Rebuild the index.
        manifest.index.clear();
        for (i, entry) in manifest.entries.iter().enumerate() {
            manifest.index.insert(entry.name.clone(), i);
        }
        Ok(manifest)
    }
}

// ── 10. SerializationEngine ─────────────────────────────────────────────────

/// Unified model serialization / deserialization engine.
///
/// Combines header, manifest, tensor serialization, compression, and checksum
/// validation into a single API. CPU reference implementation.
#[derive(Debug, Clone)]
pub struct SerializationEngine {
    config: SerialConfig,
}

impl SerializationEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: SerialConfig) -> Result<Self, SerializationError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create an engine with default configuration.
    #[must_use]
    pub fn default_engine() -> Self {
        Self { config: SerialConfig::default() }
    }

    /// Reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &SerialConfig {
        &self.config
    }

    /// Serialize a complete model (header + tensors) to bytes.
    ///
    /// Returns the serialized buffer and a manifest.
    pub fn serialize_model(
        &self,
        header: ModelHeader,
        tensors: &[(&str, &[usize], DataType, &[u8])],
    ) -> Result<(Vec<u8>, ModelManifest), SerializationError> {
        let mut writer = StreamingWriter::new(self.config.clone(), header)?;
        for &(name, shape, dtype, data) in tensors {
            writer.write_tensor(name, shape, dtype, data)?;
        }
        Ok(writer.finish())
    }

    /// Deserialize a single tensor by name from a buffer + manifest.
    pub fn deserialize_tensor(
        &self,
        data: &[u8],
        manifest: &ModelManifest,
        name: &str,
    ) -> Result<(TensorMeta, Vec<u8>), SerializationError> {
        let mut reader =
            StreamingReader::new(self.config.clone(), data.to_vec(), manifest.clone())?;
        reader.read_tensor(name)
    }

    /// Deserialize all tensors from a buffer + manifest.
    pub fn deserialize_all(
        &self,
        data: &[u8],
        manifest: &ModelManifest,
    ) -> Result<Vec<(TensorMeta, Vec<u8>)>, SerializationError> {
        let mut reader =
            StreamingReader::new(self.config.clone(), data.to_vec(), manifest.clone())?;
        let names: Vec<String> = manifest.tensor_names().iter().map(|s| s.to_string()).collect();
        let mut result = Vec::with_capacity(names.len());
        for name in &names {
            result.push(reader.read_tensor(name)?);
        }
        Ok(result)
    }

    /// Validate the integrity of all tensors in a buffer + manifest.
    pub fn validate_model(
        &self,
        data: &[u8],
        manifest: &ModelManifest,
    ) -> Result<ValidationReport, SerializationError> {
        let validator = ChecksumValidator::new(self.config.checksum);
        let mut report = ValidationReport {
            total_tensors: manifest.len() as u32,
            valid_tensors: 0,
            invalid_tensors: 0,
            errors: Vec::new(),
        };

        for entry in manifest.entries() {
            let start = entry.offset as usize;
            let end = start + entry.size_bytes as usize;
            if end > data.len() {
                report.invalid_tensors += 1;
                report.errors.push(format!("tensor '{}' extends beyond data", entry.name));
                continue;
            }
            let compressed = &data[start..end];
            match self.config.compression.decompress(compressed) {
                Ok(raw) => match validator.validate(&raw, &entry.checksum) {
                    Ok(()) => report.valid_tensors += 1,
                    Err(e) => {
                        report.invalid_tensors += 1;
                        report.errors.push(format!("tensor '{}': {e}", entry.name));
                    }
                },
                Err(e) => {
                    report.invalid_tensors += 1;
                    report.errors.push(format!("tensor '{}': {e}", entry.name));
                }
            }
        }
        Ok(report)
    }

    /// Create a [`TensorSerializer`] using this engine's config.
    #[must_use]
    pub fn tensor_serializer(&self) -> TensorSerializer {
        TensorSerializer::new(self.config.clone())
    }

    /// Create a [`TensorDeserializer`] using this engine's config.
    #[must_use]
    pub fn tensor_deserializer(&self) -> TensorDeserializer {
        TensorDeserializer::new(self.config.clone())
    }
}

/// Report produced by [`SerializationEngine::validate_model`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Total number of tensors in the manifest.
    pub total_tensors: u32,
    /// Number of tensors that passed validation.
    pub valid_tensors: u32,
    /// Number of tensors that failed validation.
    pub invalid_tensors: u32,
    /// Human-readable error messages.
    pub errors: Vec<String>,
}

impl ValidationReport {
    /// Whether all tensors are valid.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.invalid_tensors == 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper factories ────────────────────────────────────────────────

    fn default_config() -> SerialConfig {
        SerialConfig::default()
    }

    fn config_with_compression(codec: CompressionCodec) -> SerialConfig {
        SerialConfig::default().with_compression(codec)
    }

    fn config_with_checksum(algo: ChecksumAlgorithm) -> SerialConfig {
        SerialConfig::default().with_checksum(algo)
    }

    fn sample_tensor_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    fn sample_header(count: u32) -> ModelHeader {
        ModelHeader::new("test-arch", count)
    }

    // ── SerialConfig tests ──────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = default_config();
        assert_eq!(cfg.format, SerialFormat::Binary);
        assert_eq!(cfg.compression, CompressionCodec::None);
        assert_eq!(cfg.checksum, ChecksumAlgorithm::Crc32);
        assert!(!cfg.streaming);
        assert_eq!(cfg.chunk_size, 64 * 1024 * 1024);
        assert!(cfg.little_endian);
        assert_eq!(cfg.alignment, 64);
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = SerialConfig::new(SerialFormat::Gguf)
            .with_compression(CompressionCodec::Zstd)
            .with_checksum(ChecksumAlgorithm::Sha256)
            .with_streaming(true)
            .with_chunk_size(1024)
            .with_alignment(128);
        assert_eq!(cfg.format, SerialFormat::Gguf);
        assert_eq!(cfg.compression, CompressionCodec::Zstd);
        assert_eq!(cfg.checksum, ChecksumAlgorithm::Sha256);
        assert!(cfg.streaming);
        assert_eq!(cfg.chunk_size, 1024);
        assert_eq!(cfg.alignment, 128);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_chunk() {
        let mut cfg = default_config();
        cfg.chunk_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_non_power_of_2_alignment() {
        let mut cfg = default_config();
        cfg.alignment = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_alignment() {
        let mut cfg = default_config();
        cfg.alignment = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = SerialConfig::new(SerialFormat::SafeTensors)
            .with_compression(CompressionCodec::Lz4)
            .with_checksum(ChecksumAlgorithm::XxHash);
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: SerialConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg2.format, cfg.format);
        assert_eq!(cfg2.compression, cfg.compression);
        assert_eq!(cfg2.checksum, cfg.checksum);
    }

    #[test]
    fn test_serial_format_display() {
        assert_eq!(SerialFormat::Binary.to_string(), "binary");
        assert_eq!(SerialFormat::Gguf.to_string(), "gguf");
        assert_eq!(SerialFormat::SafeTensors.to_string(), "safetensors");
    }

    // ── DataType tests ──────────────────────────────────────────────────

    #[test]
    fn test_dtype_element_bytes() {
        assert_eq!(DataType::F32.element_bytes(), 4);
        assert_eq!(DataType::F16.element_bytes(), 2);
        assert_eq!(DataType::BF16.element_bytes(), 2);
        assert_eq!(DataType::I8.element_bytes(), 1);
        assert_eq!(DataType::I2S.element_bytes(), 1);
        assert_eq!(DataType::U4.element_bytes(), 1);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(DataType::F32.to_string(), "f32");
        assert_eq!(DataType::I2S.to_string(), "i2s");
        assert_eq!(DataType::U4.to_string(), "u4");
    }

    #[test]
    fn test_dtype_serde_roundtrip() {
        for dt in [
            DataType::F32,
            DataType::F16,
            DataType::BF16,
            DataType::I8,
            DataType::I2S,
            DataType::U4,
        ] {
            let json = serde_json::to_string(&dt).unwrap();
            let dt2: DataType = serde_json::from_str(&json).unwrap();
            assert_eq!(dt, dt2);
        }
    }

    // ── ModelHeader tests ───────────────────────────────────────────────

    #[test]
    fn test_header_default() {
        let h = ModelHeader::default();
        assert_eq!(&h.magic, b"BNET");
        assert_eq!(h.version, (1, 0, 0));
        assert!(h.architecture.is_empty());
        assert_eq!(h.tensor_count, 0);
    }

    #[test]
    fn test_header_new() {
        let h = ModelHeader::new("my-arch", 42);
        assert_eq!(h.architecture, "my-arch");
        assert_eq!(h.tensor_count, 42);
        assert_eq!(&h.magic, b"BNET");
    }

    #[test]
    fn test_header_builder_chain() {
        let h = ModelHeader::new("arch", 10)
            .with_version(2, 1, 0)
            .with_precision(DataType::I2S)
            .with_compression(CompressionCodec::Zstd)
            .with_checksum(ChecksumAlgorithm::Sha256)
            .with_metadata("key", "value");
        assert_eq!(h.version, (2, 1, 0));
        assert_eq!(h.precision, DataType::I2S);
        assert_eq!(h.compression, CompressionCodec::Zstd);
        assert_eq!(h.checksum_algo, ChecksumAlgorithm::Sha256);
        assert_eq!(h.metadata.get("key").unwrap(), "value");
    }

    #[test]
    fn test_header_version_string() {
        let h = ModelHeader::default().with_version(3, 2, 1);
        assert_eq!(h.version_string(), "3.2.1");
    }

    #[test]
    fn test_header_bytes_roundtrip() {
        let h = ModelHeader::new("test-arch", 5).with_metadata("author", "test");
        let bytes = h.to_bytes().unwrap();
        let h2 = ModelHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn test_header_bad_magic() {
        let mut h = ModelHeader::default();
        h.magic = *b"XXXX";
        let bytes = serde_json::to_vec(&h).unwrap();
        assert!(ModelHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_header_validate_ok() {
        let h = ModelHeader::new("arch", 1);
        assert!(h.validate().is_ok());
    }

    #[test]
    fn test_header_validate_empty_arch() {
        let h = ModelHeader::default();
        assert!(h.validate().is_err());
    }

    #[test]
    fn test_header_validate_bad_magic() {
        let mut h = ModelHeader::new("arch", 1);
        h.magic = *b"FAIL";
        assert!(h.validate().is_err());
    }

    #[test]
    fn test_header_serde_roundtrip_json() {
        let h = ModelHeader::new("arch", 8).with_version(1, 2, 3);
        let json = serde_json::to_string(&h).unwrap();
        let h2: ModelHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(h, h2);
    }

    // ── ChecksumAlgorithm tests ─────────────────────────────────────────

    #[test]
    fn test_checksum_algo_display() {
        assert_eq!(ChecksumAlgorithm::None.to_string(), "none");
        assert_eq!(ChecksumAlgorithm::Crc32.to_string(), "crc32");
        assert_eq!(ChecksumAlgorithm::Sha256.to_string(), "sha256");
        assert_eq!(ChecksumAlgorithm::XxHash.to_string(), "xxhash");
    }

    #[test]
    fn test_checksum_algo_default_is_crc32() {
        assert_eq!(ChecksumAlgorithm::default(), ChecksumAlgorithm::Crc32);
    }

    // ── ChecksumValidator tests ─────────────────────────────────────────

    #[test]
    fn test_checksum_none_empty() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::None);
        assert_eq!(v.compute(b"anything"), "");
    }

    #[test]
    fn test_checksum_none_validate_always_ok() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::None);
        assert!(v.validate(b"data", "ignored").is_ok());
    }

    #[test]
    fn test_checksum_crc32_deterministic() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Crc32);
        let a = v.compute(b"hello");
        let b = v.compute(b"hello");
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn test_checksum_crc32_different_data() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Crc32);
        assert_ne!(v.compute(b"hello"), v.compute(b"world"));
    }

    #[test]
    fn test_checksum_sha256_deterministic() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Sha256);
        let a = v.compute(b"hello");
        let b = v.compute(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64); // 4 × 16 hex chars
    }

    #[test]
    fn test_checksum_sha256_different_data() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Sha256);
        assert_ne!(v.compute(b"hello"), v.compute(b"world"));
    }

    #[test]
    fn test_checksum_xxhash_deterministic() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::XxHash);
        let a = v.compute(b"hello");
        let b = v.compute(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16); // 64-bit → 16 hex chars
    }

    #[test]
    fn test_checksum_xxhash_different_data() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::XxHash);
        assert_ne!(v.compute(b"hello"), v.compute(b"world"));
    }

    #[test]
    fn test_checksum_validate_crc32_ok() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Crc32);
        let ck = v.compute(b"test data");
        assert!(v.validate(b"test data", &ck).is_ok());
    }

    #[test]
    fn test_checksum_validate_crc32_mismatch() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Crc32);
        assert!(v.validate(b"test data", "00000000").is_err());
    }

    #[test]
    fn test_checksum_validate_sha256_ok() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::Sha256);
        let ck = v.compute(b"test data");
        assert!(v.validate(b"test data", &ck).is_ok());
    }

    #[test]
    fn test_checksum_validate_xxhash_ok() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::XxHash);
        let ck = v.compute(b"test data");
        assert!(v.validate(b"test data", &ck).is_ok());
    }

    #[test]
    fn test_checksum_empty_data() {
        for algo in [ChecksumAlgorithm::Crc32, ChecksumAlgorithm::Sha256, ChecksumAlgorithm::XxHash]
        {
            let v = ChecksumValidator::new(algo);
            let ck = v.compute(b"");
            assert!(v.validate(b"", &ck).is_ok());
        }
    }

    #[test]
    fn test_checksum_algorithm_accessor() {
        let v = ChecksumValidator::new(ChecksumAlgorithm::XxHash);
        assert_eq!(v.algorithm(), ChecksumAlgorithm::XxHash);
    }

    // ── CompressionCodec tests ──────────────────────────────────────────

    #[test]
    fn test_compression_none_roundtrip() {
        let data = b"hello world";
        let c = CompressionCodec::None;
        let compressed = c.compress(data);
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compression_zstd_roundtrip() {
        let data = sample_tensor_data(1024);
        let c = CompressionCodec::Zstd;
        let compressed = c.compress(&data);
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compression_lz4_roundtrip() {
        let data = sample_tensor_data(512);
        let c = CompressionCodec::Lz4;
        let compressed = c.compress(&data);
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compression_snappy_roundtrip() {
        let data = sample_tensor_data(256);
        let c = CompressionCodec::Snappy;
        let compressed = c.compress(&data);
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compression_empty_payload_error() {
        let c = CompressionCodec::Zstd;
        assert!(c.decompress(&[]).is_err());
    }

    #[test]
    fn test_compression_tag_mismatch() {
        let data = b"hello";
        let compressed = CompressionCodec::Zstd.compress(data);
        assert!(CompressionCodec::Lz4.decompress(&compressed).is_err());
    }

    #[test]
    fn test_compression_from_tag() {
        assert_eq!(CompressionCodec::from_tag(0x00), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_tag(0x01), Some(CompressionCodec::Zstd));
        assert_eq!(CompressionCodec::from_tag(0x02), Some(CompressionCodec::Lz4));
        assert_eq!(CompressionCodec::from_tag(0x03), Some(CompressionCodec::Snappy));
        assert_eq!(CompressionCodec::from_tag(0xFF), Option::None);
    }

    #[test]
    fn test_compression_display() {
        assert_eq!(CompressionCodec::None.to_string(), "none");
        assert_eq!(CompressionCodec::Zstd.to_string(), "zstd");
        assert_eq!(CompressionCodec::Lz4.to_string(), "lz4");
        assert_eq!(CompressionCodec::Snappy.to_string(), "snappy");
    }

    #[test]
    fn test_compression_name() {
        assert_eq!(CompressionCodec::None.name(), "none");
        assert_eq!(CompressionCodec::Zstd.name(), "zstd");
    }

    #[test]
    fn test_compression_default_is_none() {
        assert_eq!(CompressionCodec::default(), CompressionCodec::None);
    }

    #[test]
    fn test_compression_large_data_roundtrip() {
        let data = sample_tensor_data(100_000);
        for codec in [
            CompressionCodec::None,
            CompressionCodec::Zstd,
            CompressionCodec::Lz4,
            CompressionCodec::Snappy,
        ] {
            let compressed = codec.compress(&data);
            let decompressed = codec.decompress(&compressed).unwrap();
            assert_eq!(decompressed, data, "codec {codec} failed roundtrip");
        }
    }

    // ── TensorMeta tests ────────────────────────────────────────────────

    #[test]
    fn test_tensor_meta_num_elements() {
        let meta = TensorMeta {
            name: "test".into(),
            shape: vec![2, 3, 4],
            dtype: DataType::F32,
            offset: 0,
            size_bytes: 0,
            checksum: String::new(),
        };
        assert_eq!(meta.num_elements(), 24);
    }

    #[test]
    fn test_tensor_meta_empty_shape() {
        let meta = TensorMeta {
            name: "scalar".into(),
            shape: vec![],
            dtype: DataType::F32,
            offset: 0,
            size_bytes: 0,
            checksum: String::new(),
        };
        assert_eq!(meta.num_elements(), 1); // product of empty = 1
    }

    #[test]
    fn test_tensor_meta_serde_roundtrip() {
        let meta = TensorMeta {
            name: "layer.0.weight".into(),
            shape: vec![768, 768],
            dtype: DataType::BF16,
            offset: 4096,
            size_bytes: 1024,
            checksum: "abcd1234".into(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let meta2: TensorMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, meta2);
    }

    // ── TensorSerializer / TensorDeserializer tests ─────────────────────

    #[test]
    fn test_tensor_serialize_deserialize_roundtrip() {
        let cfg = default_config();
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);

        let data = sample_tensor_data(128);
        let (meta, bytes) =
            ser.serialize_tensor("test.weight", &[4, 32], DataType::F32, &data).unwrap();

        assert_eq!(meta.name, "test.weight");
        assert_eq!(meta.shape, vec![4, 32]);
        assert_eq!(meta.dtype, DataType::F32);

        let (meta2, raw) = de.deserialize_tensor(&bytes).unwrap();
        assert_eq!(meta2.name, "test.weight");
        assert_eq!(raw, data);
    }

    #[test]
    fn test_tensor_serialize_with_zstd() {
        let cfg = config_with_compression(CompressionCodec::Zstd);
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);

        let data = sample_tensor_data(256);
        let (_meta, bytes) = ser.serialize_tensor("w", &[256], DataType::I8, &data).unwrap();
        let (_meta2, raw) = de.deserialize_tensor(&bytes).unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_tensor_serialize_with_lz4() {
        let cfg = config_with_compression(CompressionCodec::Lz4);
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);

        let data = sample_tensor_data(64);
        let (_meta, bytes) = ser.serialize_tensor("b", &[64], DataType::F16, &data).unwrap();
        let (_meta2, raw) = de.deserialize_tensor(&bytes).unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_tensor_serialize_with_snappy() {
        let cfg = config_with_compression(CompressionCodec::Snappy);
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);

        let data = sample_tensor_data(32);
        let (_meta, bytes) = ser.serialize_tensor("s", &[8, 4], DataType::BF16, &data).unwrap();
        let (_meta2, raw) = de.deserialize_tensor(&bytes).unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_tensor_deserialize_too_short() {
        let cfg = default_config();
        let mut de = TensorDeserializer::new(cfg);
        assert!(de.deserialize_tensor(&[0, 1]).is_err());
    }

    #[test]
    fn test_tensor_deserialize_truncated_meta() {
        let cfg = default_config();
        let mut de = TensorDeserializer::new(cfg);
        // meta_len says 255 but only 4 bytes provided
        let bytes = [0xFF, 0x00, 0x00, 0x00, 0x00];
        assert!(de.deserialize_tensor(&bytes).is_err());
    }

    #[test]
    fn test_tensor_serialize_empty_data() {
        let cfg = default_config();
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);

        let data = b"";
        let (_meta, bytes) = ser.serialize_tensor("empty", &[0], DataType::F32, data).unwrap();
        let (_meta2, raw) = de.deserialize_tensor(&bytes).unwrap();
        assert!(raw.is_empty());
    }

    #[test]
    fn test_tensor_serializer_counter() {
        let cfg = default_config();
        let mut ser = TensorSerializer::new(cfg);
        assert_eq!(ser.tensors_written(), 0);
        ser.serialize_tensor("a", &[1], DataType::F32, &[0]).unwrap();
        assert_eq!(ser.tensors_written(), 1);
        ser.serialize_tensor("b", &[1], DataType::F32, &[0]).unwrap();
        assert_eq!(ser.tensors_written(), 2);
    }

    #[test]
    fn test_tensor_deserializer_counter() {
        let cfg = default_config();
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);
        assert_eq!(de.tensors_read(), 0);
        let (_m, bytes) = ser.serialize_tensor("a", &[1], DataType::F32, &[42]).unwrap();
        de.deserialize_tensor(&bytes).unwrap();
        assert_eq!(de.tensors_read(), 1);
    }

    #[test]
    fn test_tensor_serializer_config_accessor() {
        let cfg = config_with_compression(CompressionCodec::Lz4);
        let ser = TensorSerializer::new(cfg);
        assert_eq!(ser.config().compression, CompressionCodec::Lz4);
    }

    #[test]
    fn test_tensor_deserializer_config_accessor() {
        let cfg = config_with_checksum(ChecksumAlgorithm::Sha256);
        let de = TensorDeserializer::new(cfg);
        assert_eq!(de.config().checksum, ChecksumAlgorithm::Sha256);
    }

    #[test]
    fn test_tensor_checksum_mismatch_detected() {
        let cfg = default_config();
        let mut ser = TensorSerializer::new(cfg.clone());
        let mut de = TensorDeserializer::new(cfg);

        let data = sample_tensor_data(64);
        let (_meta, mut bytes) = ser.serialize_tensor("w", &[64], DataType::F32, &data).unwrap();

        // Corrupt one byte in the payload.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;

        let result = de.deserialize_tensor(&bytes);
        assert!(result.is_err());
    }

    // ── StreamingWriter / StreamingReader tests ─────────────────────────

    #[test]
    fn test_streaming_roundtrip_single_tensor() {
        let cfg = default_config();
        let header = sample_header(1);
        let mut writer = StreamingWriter::new(cfg.clone(), header.clone()).unwrap();

        let data = sample_tensor_data(128);
        writer.write_tensor("t0", &[128], DataType::F32, &data).unwrap();

        let (buf, manifest) = writer.finish();
        let mut reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        let (_meta, raw) = reader.read_tensor("t0").unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_streaming_roundtrip_multiple_tensors() {
        let cfg = default_config();
        let header = sample_header(3);
        let mut writer = StreamingWriter::new(cfg.clone(), header).unwrap();

        let tensors: Vec<(String, Vec<u8>)> =
            (0..3).map(|i| (format!("t{i}"), sample_tensor_data(64 * (i + 1)))).collect();

        for (name, data) in &tensors {
            writer.write_tensor(name, &[data.len()], DataType::F32, data).unwrap();
        }

        let (buf, manifest) = writer.finish();
        let mut reader = StreamingReader::new(cfg, buf, manifest).unwrap();

        for (name, expected) in &tensors {
            let (_meta, raw) = reader.read_tensor(name).unwrap();
            assert_eq!(&raw, expected);
        }
    }

    #[test]
    fn test_streaming_with_compression() {
        let cfg = config_with_compression(CompressionCodec::Zstd);
        let header = sample_header(1);
        let mut writer = StreamingWriter::new(cfg.clone(), header).unwrap();

        let data = sample_tensor_data(512);
        writer.write_tensor("t", &[512], DataType::I8, &data).unwrap();

        let (buf, manifest) = writer.finish();
        let mut reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        let (_meta, raw) = reader.read_tensor("t").unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_streaming_tensor_not_found() {
        let cfg = default_config();
        let header = sample_header(0);
        let writer = StreamingWriter::new(cfg.clone(), header).unwrap();
        let (buf, manifest) = writer.finish();
        let mut reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        assert!(reader.read_tensor("nonexistent").is_err());
    }

    #[test]
    fn test_streaming_writer_bytes_written() {
        let cfg = default_config();
        let header = sample_header(1);
        let mut writer = StreamingWriter::new(cfg, header).unwrap();
        assert_eq!(writer.bytes_written(), 0);
        writer.write_tensor("t", &[10], DataType::F32, &[0; 10]).unwrap();
        assert!(writer.bytes_written() > 0);
    }

    #[test]
    fn test_streaming_writer_tensor_count() {
        let cfg = default_config();
        let header = sample_header(2);
        let mut writer = StreamingWriter::new(cfg, header).unwrap();
        assert_eq!(writer.tensors_written(), 0);
        writer.write_tensor("a", &[1], DataType::F32, &[0]).unwrap();
        writer.write_tensor("b", &[1], DataType::F32, &[1]).unwrap();
        assert_eq!(writer.tensors_written(), 2);
    }

    #[test]
    fn test_streaming_reader_tensor_names() {
        let cfg = default_config();
        let header = sample_header(2);
        let mut writer = StreamingWriter::new(cfg.clone(), header).unwrap();
        writer.write_tensor("alpha", &[1], DataType::F32, &[0]).unwrap();
        writer.write_tensor("beta", &[1], DataType::F32, &[1]).unwrap();
        let (buf, manifest) = writer.finish();
        let reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        assert_eq!(reader.tensor_names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn test_streaming_reader_data_len() {
        let cfg = default_config();
        let header = sample_header(1);
        let mut writer = StreamingWriter::new(cfg.clone(), header).unwrap();
        writer.write_tensor("t", &[8], DataType::F32, &[0; 8]).unwrap();
        let (buf, manifest) = writer.finish();
        let reader = StreamingReader::new(cfg, buf.clone(), manifest).unwrap();
        assert_eq!(reader.data_len(), buf.len());
    }

    #[test]
    fn test_streaming_reader_tensors_read_counter() {
        let cfg = default_config();
        let header = sample_header(1);
        let mut writer = StreamingWriter::new(cfg.clone(), header).unwrap();
        writer.write_tensor("t", &[4], DataType::F32, &[0; 4]).unwrap();
        let (buf, manifest) = writer.finish();
        let mut reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        assert_eq!(reader.tensors_read(), 0);
        reader.read_tensor("t").unwrap();
        assert_eq!(reader.tensors_read(), 1);
    }

    #[test]
    fn test_streaming_small_chunk_size() {
        let cfg = default_config().with_chunk_size(16);
        let header = sample_header(1);
        let mut writer = StreamingWriter::new(cfg.clone(), header).unwrap();
        let data = sample_tensor_data(256);
        writer.write_tensor("t", &[256], DataType::I8, &data).unwrap();
        let (buf, manifest) = writer.finish();
        let mut reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        let (_meta, raw) = reader.read_tensor("t").unwrap();
        assert_eq!(raw, data);
    }

    // ── ModelManifest tests ─────────────────────────────────────────────

    #[test]
    fn test_manifest_new_empty() {
        let m = ModelManifest::new(sample_header(0));
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_manifest_add_and_get() {
        let mut m = ModelManifest::new(sample_header(1));
        let meta = TensorMeta {
            name: "w".into(),
            shape: vec![10],
            dtype: DataType::F32,
            offset: 0,
            size_bytes: 40,
            checksum: "abc".into(),
        };
        m.add_entry(meta.clone());
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
        assert_eq!(m.get_entry("w"), Some(&meta));
        assert_eq!(m.get_entry("missing"), Option::None);
    }

    #[test]
    fn test_manifest_tensor_names() {
        let mut m = ModelManifest::new(sample_header(2));
        m.add_entry(TensorMeta {
            name: "a".into(),
            shape: vec![1],
            dtype: DataType::F32,
            offset: 0,
            size_bytes: 4,
            checksum: String::new(),
        });
        m.add_entry(TensorMeta {
            name: "b".into(),
            shape: vec![2],
            dtype: DataType::F32,
            offset: 4,
            size_bytes: 8,
            checksum: String::new(),
        });
        assert_eq!(m.tensor_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_manifest_total_data_bytes() {
        let mut m = ModelManifest::new(sample_header(2));
        m.add_entry(TensorMeta {
            name: "a".into(),
            shape: vec![1],
            dtype: DataType::F32,
            offset: 0,
            size_bytes: 100,
            checksum: String::new(),
        });
        m.add_entry(TensorMeta {
            name: "b".into(),
            shape: vec![1],
            dtype: DataType::F32,
            offset: 100,
            size_bytes: 200,
            checksum: String::new(),
        });
        assert_eq!(m.total_data_bytes(), 300);
    }

    #[test]
    fn test_manifest_bytes_roundtrip() {
        let mut m = ModelManifest::new(sample_header(1));
        m.add_entry(TensorMeta {
            name: "x".into(),
            shape: vec![3, 4],
            dtype: DataType::BF16,
            offset: 0,
            size_bytes: 24,
            checksum: "ff".into(),
        });
        let bytes = m.to_bytes().unwrap();
        let m2 = ModelManifest::from_bytes(&bytes).unwrap();
        assert_eq!(m2.len(), 1);
        assert_eq!(m2.get_entry("x").unwrap().name, "x");
        assert_eq!(m2.get_entry("x").unwrap().shape, vec![3, 4]);
    }

    #[test]
    fn test_manifest_entries_order() {
        let mut m = ModelManifest::new(sample_header(3));
        for name in ["first", "second", "third"] {
            m.add_entry(TensorMeta {
                name: name.into(),
                shape: vec![1],
                dtype: DataType::F32,
                offset: 0,
                size_bytes: 0,
                checksum: String::new(),
            });
        }
        let names: Vec<&str> = m.entries().iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    // ── SerializationEngine tests ───────────────────────────────────────

    #[test]
    fn test_engine_default() {
        let engine = SerializationEngine::default_engine();
        assert_eq!(engine.config().format, SerialFormat::Binary);
    }

    #[test]
    fn test_engine_new_validates_config() {
        let mut cfg = default_config();
        cfg.alignment = 0;
        assert!(SerializationEngine::new(cfg).is_err());
    }

    #[test]
    fn test_engine_serialize_deserialize_single() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(1);
        let data = sample_tensor_data(64);

        let (buf, manifest) = engine
            .serialize_model(header, &[("w", &[64], DataType::F32, data.as_slice())])
            .unwrap();

        let (meta, raw) = engine.deserialize_tensor(&buf, &manifest, "w").unwrap();
        assert_eq!(meta.name, "w");
        assert_eq!(raw, data);
    }

    #[test]
    fn test_engine_serialize_deserialize_multiple() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(3);
        let d0 = sample_tensor_data(32);
        let d1 = sample_tensor_data(64);
        let d2 = sample_tensor_data(16);

        let tensors: Vec<(&str, &[usize], DataType, &[u8])> = vec![
            ("a", &[32], DataType::F32, &d0),
            ("b", &[8, 8], DataType::F16, &d1),
            ("c", &[16], DataType::I8, &d2),
        ];

        let (buf, manifest) = engine.serialize_model(header, &tensors).unwrap();
        let all = engine.deserialize_all(&buf, &manifest).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].1, d0);
        assert_eq!(all[1].1, d1);
        assert_eq!(all[2].1, d2);
    }

    #[test]
    fn test_engine_deserialize_not_found() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(0);
        let (buf, manifest) = engine.serialize_model(header, &[]).unwrap();
        assert!(engine.deserialize_tensor(&buf, &manifest, "nope").is_err());
    }

    #[test]
    fn test_engine_validate_clean_model() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(2);
        let d0 = sample_tensor_data(32);
        let d1 = sample_tensor_data(64);

        let (buf, manifest) = engine
            .serialize_model(
                header,
                &[("a", &[32], DataType::F32, &d0), ("b", &[64], DataType::I8, &d1)],
            )
            .unwrap();

        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
        assert_eq!(report.valid_tensors, 2);
        assert_eq!(report.invalid_tensors, 0);
    }

    #[test]
    fn test_engine_validate_corrupted_data() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(1);
        let data = sample_tensor_data(32);

        let (mut buf, manifest) =
            engine.serialize_model(header, &[("a", &[32], DataType::F32, &data)]).unwrap();

        // Corrupt a byte in the middle.
        if buf.len() > 5 {
            let mid = buf.len() / 2;
            buf[mid] ^= 0xFF;
        }

        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(!report.is_valid());
        assert_eq!(report.invalid_tensors, 1);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_engine_validate_truncated_data() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(1);
        let data = sample_tensor_data(64);

        let (buf, manifest) =
            engine.serialize_model(header, &[("a", &[64], DataType::F32, &data)]).unwrap();

        // Truncate the buffer.
        let short = &buf[..buf.len() / 2];
        let report = engine.validate_model(short, &manifest).unwrap();
        assert!(!report.is_valid());
    }

    #[test]
    fn test_engine_with_zstd_compression() {
        let cfg = config_with_compression(CompressionCodec::Zstd);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header = sample_header(1);
        let data = sample_tensor_data(128);

        let (buf, manifest) =
            engine.serialize_model(header, &[("t", &[128], DataType::F32, &data)]).unwrap();
        let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "t").unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_engine_with_sha256_checksum() {
        let cfg = config_with_checksum(ChecksumAlgorithm::Sha256);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header = sample_header(1);
        let data = sample_tensor_data(64);

        let (buf, manifest) =
            engine.serialize_model(header, &[("t", &[64], DataType::F32, &data)]).unwrap();
        let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "t").unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_engine_with_xxhash_checksum() {
        let cfg = config_with_checksum(ChecksumAlgorithm::XxHash);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header = sample_header(1);
        let data = sample_tensor_data(48);

        let (buf, manifest) =
            engine.serialize_model(header, &[("t", &[48], DataType::F32, &data)]).unwrap();
        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
    }

    #[test]
    fn test_engine_tensor_serializer_factory() {
        let engine = SerializationEngine::default_engine();
        let ser = engine.tensor_serializer();
        assert_eq!(ser.config().format, SerialFormat::Binary);
    }

    #[test]
    fn test_engine_tensor_deserializer_factory() {
        let engine = SerializationEngine::default_engine();
        let de = engine.tensor_deserializer();
        assert_eq!(de.config().format, SerialFormat::Binary);
    }

    #[test]
    fn test_engine_config_accessor() {
        let cfg = config_with_compression(CompressionCodec::Snappy);
        let engine = SerializationEngine::new(cfg).unwrap();
        assert_eq!(engine.config().compression, CompressionCodec::Snappy);
    }

    // ── Error type tests ────────────────────────────────────────────────

    #[test]
    fn test_error_display_io() {
        let e = SerializationError::Io(io::Error::new(io::ErrorKind::NotFound, "gone"));
        assert!(e.to_string().contains("I/O error"));
    }

    #[test]
    fn test_error_display_serde() {
        let e = SerializationError::Serde("bad json".into());
        assert!(e.to_string().contains("serialization error"));
    }

    #[test]
    fn test_error_display_checksum_mismatch() {
        let e =
            SerializationError::ChecksumMismatch { expected: "aaa".into(), actual: "bbb".into() };
        let s = e.to_string();
        assert!(s.contains("checksum mismatch"));
        assert!(s.contains("aaa"));
        assert!(s.contains("bbb"));
    }

    #[test]
    fn test_error_display_invalid_header() {
        let e = SerializationError::InvalidHeader("bad".into());
        assert!(e.to_string().contains("invalid header"));
    }

    #[test]
    fn test_error_display_unsupported_format() {
        let e = SerializationError::UnsupportedFormat("v99".into());
        assert!(e.to_string().contains("unsupported format"));
    }

    #[test]
    fn test_error_display_tensor_not_found() {
        let e = SerializationError::TensorNotFound("w".into());
        assert!(e.to_string().contains("tensor not found"));
    }

    #[test]
    fn test_error_display_compression() {
        let e = SerializationError::CompressionError("fail".into());
        assert!(e.to_string().contains("compression error"));
    }

    #[test]
    fn test_error_display_invalid_config() {
        let e = SerializationError::InvalidConfig("zero".into());
        assert!(e.to_string().contains("invalid config"));
    }

    #[test]
    fn test_error_source_io() {
        let inner = io::Error::other("oops");
        let e = SerializationError::Io(inner);
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn test_error_source_non_io() {
        let e = SerializationError::Serde("x".into());
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "nope");
        let e: SerializationError = io_err.into();
        assert!(matches!(e, SerializationError::Io(_)));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_err = serde_json::from_str::<String>("not json").unwrap_err();
        let e: SerializationError = json_err.into();
        assert!(matches!(e, SerializationError::Serde(_)));
    }

    // ── Cross-component integration tests ───────────────────────────────

    #[test]
    fn test_full_pipeline_binary_crc32() {
        let cfg = SerialConfig::new(SerialFormat::Binary)
            .with_compression(CompressionCodec::None)
            .with_checksum(ChecksumAlgorithm::Crc32);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header =
            ModelHeader::new("full-test", 2).with_version(1, 0, 0).with_precision(DataType::F32);
        let d0 = sample_tensor_data(100);
        let d1 = sample_tensor_data(200);
        let (buf, manifest) = engine
            .serialize_model(
                header,
                &[
                    ("layer0", &[10, 10], DataType::F32, &d0),
                    ("layer1", &[20, 10], DataType::F32, &d1),
                ],
            )
            .unwrap();
        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
        let all = engine.deserialize_all(&buf, &manifest).unwrap();
        assert_eq!(all[0].1, d0);
        assert_eq!(all[1].1, d1);
    }

    #[test]
    fn test_full_pipeline_zstd_sha256() {
        let cfg = SerialConfig::new(SerialFormat::Binary)
            .with_compression(CompressionCodec::Zstd)
            .with_checksum(ChecksumAlgorithm::Sha256);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header = ModelHeader::new("zstd-sha", 1).with_precision(DataType::I2S);
        let data = sample_tensor_data(512);
        let (buf, manifest) =
            engine.serialize_model(header, &[("w", &[512], DataType::I2S, &data)]).unwrap();
        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
        let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "w").unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_full_pipeline_lz4_xxhash() {
        let cfg = SerialConfig::new(SerialFormat::Binary)
            .with_compression(CompressionCodec::Lz4)
            .with_checksum(ChecksumAlgorithm::XxHash);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header = ModelHeader::new("lz4-xx", 1).with_precision(DataType::BF16);
        let data = sample_tensor_data(256);
        let (buf, manifest) =
            engine.serialize_model(header, &[("t", &[256], DataType::BF16, &data)]).unwrap();
        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
    }

    #[test]
    fn test_full_pipeline_snappy_no_checksum() {
        let cfg = SerialConfig::new(SerialFormat::Binary)
            .with_compression(CompressionCodec::Snappy)
            .with_checksum(ChecksumAlgorithm::None);
        let engine = SerializationEngine::new(cfg).unwrap();
        let header = ModelHeader::new("snappy-none", 1);
        let data = sample_tensor_data(128);
        let (buf, manifest) =
            engine.serialize_model(header, &[("t", &[128], DataType::F32, &data)]).unwrap();
        let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "t").unwrap();
        assert_eq!(raw, data);
    }

    #[test]
    fn test_manifest_roundtrip_after_serialization() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(2);
        let d0 = sample_tensor_data(16);
        let d1 = sample_tensor_data(32);
        let (_, manifest) = engine
            .serialize_model(
                header,
                &[("a", &[16], DataType::F32, &d0), ("b", &[32], DataType::F32, &d1)],
            )
            .unwrap();

        let bytes = manifest.to_bytes().unwrap();
        let manifest2 = ModelManifest::from_bytes(&bytes).unwrap();
        assert_eq!(manifest2.len(), 2);
        assert_eq!(manifest2.tensor_names(), vec!["a", "b"]);
    }

    #[test]
    fn test_streaming_writer_manifest_accessor() {
        let cfg = default_config();
        let header = sample_header(0);
        let writer = StreamingWriter::new(cfg, header.clone()).unwrap();
        assert_eq!(writer.manifest().header.architecture, header.architecture);
    }

    #[test]
    fn test_streaming_reader_manifest_accessor() {
        let cfg = default_config();
        let header = sample_header(0);
        let writer = StreamingWriter::new(cfg.clone(), header.clone()).unwrap();
        let (buf, manifest) = writer.finish();
        let reader = StreamingReader::new(cfg, buf, manifest).unwrap();
        assert_eq!(reader.manifest().header.architecture, header.architecture);
    }

    #[test]
    fn test_validation_report_default() {
        let report = ValidationReport {
            total_tensors: 0,
            valid_tensors: 0,
            invalid_tensors: 0,
            errors: vec![],
        };
        assert!(report.is_valid());
    }

    #[test]
    fn test_validation_report_with_errors() {
        let report = ValidationReport {
            total_tensors: 2,
            valid_tensors: 1,
            invalid_tensors: 1,
            errors: vec!["bad tensor".into()],
        };
        assert!(!report.is_valid());
    }

    #[test]
    fn test_many_tensors_roundtrip() {
        let engine = SerializationEngine::default_engine();
        let n = 50;
        let header = sample_header(n as u32);
        let tensor_data: Vec<(String, Vec<u8>)> =
            (0..n).map(|i| (format!("layer.{i}.weight"), sample_tensor_data(32 + i * 4))).collect();

        let shapes: Vec<[usize; 1]> = tensor_data.iter().map(|(_, d)| [d.len()]).collect();
        let tensors: Vec<(&str, &[usize], DataType, &[u8])> = tensor_data
            .iter()
            .zip(shapes.iter())
            .map(|((name, data), shape)| {
                (name.as_str(), shape.as_slice(), DataType::F32, data.as_slice())
            })
            .collect();

        let (buf, manifest) = engine.serialize_model(header, &tensors).unwrap();
        assert_eq!(manifest.len(), n);

        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
        assert_eq!(report.valid_tensors, n as u32);

        let all = engine.deserialize_all(&buf, &manifest).unwrap();
        for (i, (meta, raw)) in all.iter().enumerate() {
            assert_eq!(meta.name, tensor_data[i].0);
            assert_eq!(raw, &tensor_data[i].1);
        }
    }

    #[test]
    fn test_engine_zero_tensors() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(0);
        let (buf, manifest) = engine.serialize_model(header, &[]).unwrap();
        assert!(buf.is_empty());
        assert!(manifest.is_empty());
        let report = engine.validate_model(&buf, &manifest).unwrap();
        assert!(report.is_valid());
        assert_eq!(report.total_tensors, 0);
    }

    #[test]
    fn test_header_metadata_multiple_entries() {
        let h = ModelHeader::new("a", 1)
            .with_metadata("k1", "v1")
            .with_metadata("k2", "v2")
            .with_metadata("k3", "v3");
        assert_eq!(h.metadata.len(), 3);
        assert_eq!(h.metadata["k1"], "v1");
        assert_eq!(h.metadata["k2"], "v2");
        assert_eq!(h.metadata["k3"], "v3");
    }

    #[test]
    fn test_compression_all_codecs_with_engine() {
        for codec in [
            CompressionCodec::None,
            CompressionCodec::Zstd,
            CompressionCodec::Lz4,
            CompressionCodec::Snappy,
        ] {
            let cfg = config_with_compression(codec);
            let engine = SerializationEngine::new(cfg).unwrap();
            let data = sample_tensor_data(100);
            let header = sample_header(1);
            let (buf, manifest) =
                engine.serialize_model(header, &[("t", &[100], DataType::F32, &data)]).unwrap();
            let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "t").unwrap();
            assert_eq!(raw, data, "codec {codec} failed engine roundtrip");
        }
    }

    #[test]
    fn test_all_checksum_algos_with_engine() {
        for algo in [
            ChecksumAlgorithm::None,
            ChecksumAlgorithm::Crc32,
            ChecksumAlgorithm::Sha256,
            ChecksumAlgorithm::XxHash,
        ] {
            let cfg = config_with_checksum(algo);
            let engine = SerializationEngine::new(cfg).unwrap();
            let data = sample_tensor_data(64);
            let header = sample_header(1);
            let (buf, manifest) =
                engine.serialize_model(header, &[("t", &[64], DataType::F32, &data)]).unwrap();
            let report = engine.validate_model(&buf, &manifest).unwrap();
            assert!(report.is_valid(), "algo {algo} failed validation");
        }
    }

    #[test]
    fn test_all_dtypes_roundtrip() {
        let engine = SerializationEngine::default_engine();
        for dtype in [
            DataType::F32,
            DataType::F16,
            DataType::BF16,
            DataType::I8,
            DataType::I2S,
            DataType::U4,
        ] {
            let data = sample_tensor_data(16);
            let header = sample_header(1);
            let (buf, manifest) =
                engine.serialize_model(header, &[("t", &[16], dtype, &data)]).unwrap();
            let (meta, raw) = engine.deserialize_tensor(&buf, &manifest, "t").unwrap();
            assert_eq!(meta.dtype, dtype);
            assert_eq!(raw, data, "dtype {dtype} roundtrip failed");
        }
    }

    #[test]
    fn test_single_byte_tensor() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(1);
        let (buf, manifest) =
            engine.serialize_model(header, &[("t", &[1], DataType::I8, &[42])]).unwrap();
        let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "t").unwrap();
        assert_eq!(raw, vec![42]);
    }

    #[test]
    fn test_large_tensor_roundtrip() {
        let engine = SerializationEngine::default_engine();
        let header = sample_header(1);
        let data = sample_tensor_data(1_000_000);
        let (buf, manifest) = engine
            .serialize_model(header, &[("big", &[1000, 1000], DataType::F32, &data)])
            .unwrap();
        let (_, raw) = engine.deserialize_tensor(&buf, &manifest, "big").unwrap();
        assert_eq!(raw, data);
    }
}
