//! Dense GGUF linear fixture extraction.
//!
//! This module bridges descriptor-only dense GGUF inspection toward a future
//! dense CUDA parity lane. It extracts one recognized dense linear tensor,
//! materializes it as F32, and computes a deterministic CPU reference matvec.
//! It does not execute CUDA, load a full dense model, or claim dense GGUF
//! inference support.

use crate::dense_gguf_descriptors::{DenseGgufTensorRole, inspect_dense_gguf_tensor_descriptors};
use crate::formats::gguf::{GgufReader, GgufTensorType};
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Receipt artifact kind for descriptor-driven dense GGUF linear fixture extraction.
pub const DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND: &str = "dense_gguf_linear_fixture_extraction";
pub const DENSE_GGUF_Q8_LINEAR_SIDECAR_ARTIFACT_KIND: &str =
    "dense_gguf_q8_linear_sidecar_prototype";

/// Logical matrix layout used by the extracted fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseGgufLinearLayout {
    /// GGUF dense projection dims are commonly `[in, out]`; the runtime
    /// consumes them as row-major `[out, in]`.
    GgufInOutReinterpretedAsOutIn,
}

/// Hash-only receipt summary for an extracted dense GGUF linear fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufLinearFixtureSummary {
    pub schema: u64,
    pub artifact_kind: String,
    pub architecture: String,
    pub model_family: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub tensor_type: String,
    pub source_shape: Vec<usize>,
    pub source_offset: u64,
    pub source_size_bytes: u64,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub value_count: usize,
    pub logical_layout: DenseGgufLinearLayout,
    pub values_materialized_as_f32: bool,
    pub weight_values_sha256: String,
    pub cpu_reference_input_len: usize,
    pub cpu_reference_output_len: usize,
    pub cpu_reference_input_sha256: String,
    pub cpu_reference_output_sha256: String,
    pub cpu_reference_computed: bool,
    pub dense_gguf_inference_claimed: bool,
    pub dense_regular_llm_cuda_claimed: bool,
    pub cpu_cuda_parity_claimed: bool,
    pub bitnet_packed_i2s_qk256_proof: bool,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
}

/// Extracted fixture data plus its receipt-ready summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufLinearFixture {
    pub summary: DenseGgufLinearFixtureSummary,
    pub weight_values_f32: Vec<f32>,
    pub cpu_reference_input: Vec<f32>,
    pub cpu_reference_output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufQ8LinearSidecarSummary {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub tensor_type: String,
    pub source_shape: Vec<usize>,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub value_count: usize,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub packed_q8_bytes_sha256: String,
    pub cpu_reference_input_sha256: String,
    pub fused_output_sha256: String,
    pub eager_output_sha256: String,
    pub max_abs_diff_vs_eager_f32: f32,
    pub dequantizes_inside_matvec: bool,
    pub materializes_full_f32_weights: bool,
    pub compares_against_eager_f32_reference: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub speedup_claim: bool,
    pub dense_runtime_replaced: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseGgufQ8LinearSidecar {
    pub summary: DenseGgufQ8LinearSidecarSummary,
    pub cpu_reference_input: Vec<f32>,
    pub fused_output: Vec<f32>,
    pub eager_output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct Q8Block {
    scale: f32,
    qs: [i8; 32],
}

/// Extract the first tensor for `role` as a dense linear CPU-reference fixture.
///
/// The extractor fails closed for non-linear roles, BitNet packed markers,
/// unsupported dense quantizations, and non-matrix tensors. Q8_0 is supported
/// here only as a F32 materialization bridge for CPU reference fixtures; it is
/// not a dense CUDA execution claim.
pub fn extract_dense_gguf_linear_fixture(
    reader: &GgufReader<'_>,
    role: DenseGgufTensorRole,
) -> Result<DenseGgufLinearFixture> {
    if !is_extractable_linear_role(role) {
        return Err(BitNetError::Validation(format!(
            "dense GGUF linear fixture extraction requires an extractable linear role, got {role:?}"
        )));
    }

    let inspection = inspect_dense_gguf_tensor_descriptors(reader)?;
    let descriptor =
        inspection.descriptors.iter().find(|descriptor| descriptor.role == role).ok_or_else(
            || {
                BitNetError::Validation(format!(
                    "dense GGUF linear fixture extraction could not find role {role:?}"
                ))
            },
        )?;
    let info = reader.get_tensor_info_by_name(&descriptor.name).ok_or_else(|| {
        BitNetError::Validation(format!(
            "dense GGUF linear fixture extraction descriptor '{}' is missing tensor info",
            descriptor.name
        ))
    })?;
    if info.shape.len() != 2 {
        return Err(BitNetError::Validation(format!(
            "dense GGUF linear fixture '{}' requires a 2D tensor, got {:?}",
            info.name, info.shape
        )));
    }

    let data = reader.get_tensor_data_by_info(info)?;
    let weight_values_f32 = tensor_values_as_f32(data, info.tensor_type, &info.shape, &info.name)?;
    let (matrix_rows, matrix_cols) = (info.shape[1], info.shape[0]);
    let value_count = matrix_rows.checked_mul(matrix_cols).ok_or_else(|| {
        BitNetError::Validation(format!(
            "dense GGUF linear fixture '{}' matrix shape overflows: {}x{}",
            info.name, matrix_rows, matrix_cols
        ))
    })?;
    if weight_values_f32.len() != value_count {
        return Err(BitNetError::Validation(format!(
            "dense GGUF linear fixture '{}' materialized {} values, expected {}",
            info.name,
            weight_values_f32.len(),
            value_count
        )));
    }

    let cpu_reference_input = deterministic_reference_input(matrix_cols);
    let cpu_reference_output =
        dense_matvec_row_major(&weight_values_f32, matrix_rows, matrix_cols, &cpu_reference_input)?;

    let summary = DenseGgufLinearFixtureSummary {
        schema: 1,
        artifact_kind: DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND.to_string(),
        architecture: inspection.architecture,
        model_family: inspection.model_family,
        tensor_name: info.name.clone(),
        role,
        tensor_type: tensor_type_label(info.tensor_type).to_string(),
        source_shape: info.shape.clone(),
        source_offset: info.offset,
        source_size_bytes: info.size,
        matrix_rows,
        matrix_cols,
        value_count,
        logical_layout: DenseGgufLinearLayout::GgufInOutReinterpretedAsOutIn,
        values_materialized_as_f32: true,
        weight_values_sha256: f32_values_sha256(&weight_values_f32),
        cpu_reference_input_len: cpu_reference_input.len(),
        cpu_reference_output_len: cpu_reference_output.len(),
        cpu_reference_input_sha256: f32_values_sha256(&cpu_reference_input),
        cpu_reference_output_sha256: f32_values_sha256(&cpu_reference_output),
        cpu_reference_computed: true,
        dense_gguf_inference_claimed: false,
        dense_regular_llm_cuda_claimed: false,
        cpu_cuda_parity_claimed: false,
        bitnet_packed_i2s_qk256_proof: false,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    };

    Ok(DenseGgufLinearFixture {
        summary,
        weight_values_f32,
        cpu_reference_input,
        cpu_reference_output,
    })
}

/// Extract a Q8_0 dense linear sidecar and compute a dequant-fused CPU matvec.
///
/// This is a locality prototype only. It keeps packed Q8_0 scales/codes and
/// dequantizes inside the dot product for one extracted dense linear fixture,
/// then compares against the existing eager F32 fixture. It does not replace the
/// production dense runtime path and it does not make a speed claim.
pub fn extract_dense_gguf_q8_linear_sidecar(
    reader: &GgufReader<'_>,
    role: DenseGgufTensorRole,
) -> Result<DenseGgufQ8LinearSidecar> {
    if !is_extractable_linear_role(role) {
        return Err(BitNetError::Validation(format!(
            "Q8_0 dense linear sidecar extraction requires an extractable linear role, got {role:?}"
        )));
    }

    let eager = extract_dense_gguf_linear_fixture(reader, role)?;
    if eager.summary.tensor_type != "q8_0" {
        return Err(BitNetError::Validation(format!(
            "Q8_0 dense linear sidecar extraction requires tensor type q8_0, got {} for '{}'",
            eager.summary.tensor_type, eager.summary.tensor_name
        )));
    }

    let info = reader.get_tensor_info_by_name(&eager.summary.tensor_name).ok_or_else(|| {
        BitNetError::Validation(format!(
            "Q8_0 dense linear sidecar descriptor '{}' is missing tensor info",
            eager.summary.tensor_name
        ))
    })?;
    let data = reader.get_tensor_data_by_info(info)?;
    let elements = eager.summary.value_count;
    let blocks = q8_0_blocks(data, elements, &info.name)?;
    let fused_output = q8_0_matvec_dequant_fused_row_major(
        &blocks,
        elements,
        eager.summary.matrix_rows,
        eager.summary.matrix_cols,
        &eager.cpu_reference_input,
    )?;
    let max_abs_diff_vs_eager_f32 = max_abs_diff(&fused_output, &eager.cpu_reference_output)?;

    let summary = DenseGgufQ8LinearSidecarSummary {
        schema: 1,
        artifact_kind: DENSE_GGUF_Q8_LINEAR_SIDECAR_ARTIFACT_KIND.to_string(),
        tensor_name: eager.summary.tensor_name,
        role,
        tensor_type: eager.summary.tensor_type,
        source_shape: eager.summary.source_shape,
        matrix_rows: eager.summary.matrix_rows,
        matrix_cols: eager.summary.matrix_cols,
        value_count: eager.summary.value_count,
        q8_block_size: 32,
        q8_block_count: blocks.len(),
        packed_q8_bytes_sha256: bytes_sha256(data),
        cpu_reference_input_sha256: f32_values_sha256(&eager.cpu_reference_input),
        fused_output_sha256: f32_values_sha256(&fused_output),
        eager_output_sha256: f32_values_sha256(&eager.cpu_reference_output),
        max_abs_diff_vs_eager_f32,
        dequantizes_inside_matvec: true,
        materializes_full_f32_weights: false,
        compares_against_eager_f32_reference: true,
        generated_id_preservation_required_before_runtime_use: true,
        speedup_claim: false,
        dense_runtime_replaced: false,
    };

    Ok(DenseGgufQ8LinearSidecar {
        summary,
        cpu_reference_input: eager.cpu_reference_input,
        fused_output,
        eager_output: eager.cpu_reference_output,
    })
}

fn is_extractable_linear_role(role: DenseGgufTensorRole) -> bool {
    matches!(
        role,
        DenseGgufTensorRole::Output
            | DenseGgufTensorRole::AttentionQ
            | DenseGgufTensorRole::AttentionK
            | DenseGgufTensorRole::AttentionV
            | DenseGgufTensorRole::AttentionOutput
            | DenseGgufTensorRole::MlpGate
            | DenseGgufTensorRole::MlpUp
            | DenseGgufTensorRole::MlpDown
    )
}

fn tensor_values_as_f32(
    bytes: &[u8],
    tensor_type: GgufTensorType,
    shape: &[usize],
    tensor_name: &str,
) -> Result<Vec<f32>> {
    match tensor_type {
        GgufTensorType::F32 => f32_tensor_values(bytes, shape, tensor_name),
        GgufTensorType::F16 => f16_tensor_values(bytes, shape, tensor_name),
        GgufTensorType::Q8_0 => q8_0_tensor_values(bytes, shape, tensor_name),
        other => Err(BitNetError::Validation(format!(
            "dense GGUF linear fixture extraction for '{}' does not support tensor type {} yet",
            tensor_name,
            tensor_type_label(other)
        ))),
    }
}

fn f32_tensor_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = checked_element_count(shape, tensor_name, "F32")?;
    let expected_bytes = elements.checked_mul(4).ok_or_else(|| {
        BitNetError::Validation(format!(
            "F32 tensor '{tensor_name}' byte count overflows for shape {shape:?}"
        ))
    })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "F32 tensor '{tensor_name}' has {} bytes, expected at least {}",
            bytes.len(),
            expected_bytes
        )));
    }

    Ok(bytes[..expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn f16_tensor_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = checked_element_count(shape, tensor_name, "F16")?;
    let expected_bytes = elements.checked_mul(2).ok_or_else(|| {
        BitNetError::Validation(format!(
            "F16 tensor '{tensor_name}' byte count overflows for shape {shape:?}"
        ))
    })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "F16 tensor '{tensor_name}' has {} bytes, expected at least {}",
            bytes.len(),
            expected_bytes
        )));
    }

    Ok(bytes[..expected_bytes]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn q8_0_tensor_values(bytes: &[u8], shape: &[usize], tensor_name: &str) -> Result<Vec<f32>> {
    let elements = checked_element_count(shape, tensor_name, "Q8_0")?;
    let blocks = elements.div_ceil(32);
    let expected_bytes =
        blocks.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
            BitNetError::Validation(format!(
                "Q8_0 tensor '{tensor_name}' byte count overflows for {blocks} blocks"
            ))
        })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "Q8_0 tensor '{tensor_name}' has {} bytes, expected at least {}",
            bytes.len(),
            expected_bytes
        )));
    }

    let mut values = Vec::with_capacity(elements);
    for block_idx in 0..blocks {
        let offset = block_idx * GgufTensorType::Q8_0.element_size();
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        for code_idx in 0..32 {
            if values.len() == elements {
                break;
            }
            let q = bytes[offset + 2 + code_idx] as i8;
            values.push(scale * f32::from(q));
        }
    }

    Ok(values)
}

fn q8_0_blocks(bytes: &[u8], elements: usize, tensor_name: &str) -> Result<Vec<Q8Block>> {
    let blocks = elements.div_ceil(32);
    let expected_bytes =
        blocks.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
            BitNetError::Validation(format!(
                "Q8_0 tensor '{tensor_name}' byte count overflows for {blocks} blocks"
            ))
        })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "Q8_0 tensor '{tensor_name}' has {} bytes, expected at least {}",
            bytes.len(),
            expected_bytes
        )));
    }

    let mut parsed = Vec::with_capacity(blocks);
    for block_idx in 0..blocks {
        let offset = block_idx * GgufTensorType::Q8_0.element_size();
        let scale_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let mut qs = [0i8; 32];
        for code_idx in 0..32 {
            qs[code_idx] = bytes[offset + 2 + code_idx] as i8;
        }
        parsed.push(Q8Block { scale: half::f16::from_bits(scale_bits).to_f32(), qs });
    }
    Ok(parsed)
}

fn q8_0_matvec_dequant_fused_row_major(
    blocks: &[Q8Block],
    elements: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != cols {
        return Err(BitNetError::Validation(format!(
            "Q8_0 fused dense linear input length {} does not match matrix cols {}",
            input.len(),
            cols
        )));
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        BitNetError::Validation(format!(
            "Q8_0 fused dense linear matrix shape overflows: {rows}x{cols}"
        ))
    })?;
    if expected != elements {
        return Err(BitNetError::Validation(format!(
            "Q8_0 fused dense linear element count {elements} does not match matrix shape {rows}x{cols}"
        )));
    }

    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_start = row * cols;
        let mut sum = 0.0f32;
        for (col, value) in input.iter().enumerate().take(cols) {
            let weight_idx = row_start + col;
            let block = &blocks[weight_idx / 32];
            let q = f32::from(block.qs[weight_idx % 32]);
            sum += block.scale * q * *value;
        }
        output.push(sum);
    }
    Ok(output)
}

fn checked_element_count(shape: &[usize], tensor_name: &str, dtype: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            BitNetError::Validation(format!(
                "{dtype} tensor '{tensor_name}' shape {shape:?} overflows element count"
            ))
        })
    })
}

fn deterministic_reference_input(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|idx| {
            let centered = (idx % 17) as f32 - 8.0;
            centered / 16.0
        })
        .collect()
}

fn dense_matvec_row_major(
    weights: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != cols {
        return Err(BitNetError::Validation(format!(
            "dense GGUF linear fixture input length {} does not match matrix cols {}",
            input.len(),
            cols
        )));
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        BitNetError::Validation(format!(
            "dense GGUF linear fixture matrix shape overflows: {rows}x{cols}"
        ))
    })?;
    if weights.len() != expected {
        return Err(BitNetError::Validation(format!(
            "dense GGUF linear fixture has {} weights, expected {}",
            weights.len(),
            expected
        )));
    }

    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let mut sum = 0.0f32;
        for col in 0..cols {
            sum += weights[start + col] * input[col];
        }
        output.push(sum);
    }
    Ok(output)
}

fn f32_values_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(BitNetError::Validation(format!(
            "dense linear output lengths differ: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left.iter().zip(right).map(|(l, r)| (l - r).abs()).fold(0.0f32, f32::max))
}

fn tensor_type_label(tensor_type: GgufTensorType) -> &'static str {
    match tensor_type {
        GgufTensorType::F32 => "f32",
        GgufTensorType::F16 => "f16",
        GgufTensorType::F64 => "f64",
        GgufTensorType::Q4_0 => "q4_0",
        GgufTensorType::Q4_1 => "q4_1",
        GgufTensorType::Q5_0 => "q5_0",
        GgufTensorType::Q5_1 => "q5_1",
        GgufTensorType::Q8_0 => "q8_0",
        GgufTensorType::Q8_1 => "q8_1",
        GgufTensorType::Q2_K => "q2_k",
        GgufTensorType::Q3_K => "q3_k",
        GgufTensorType::Q4_K => "q4_k",
        GgufTensorType::Q5_K => "q5_k",
        GgufTensorType::Q6_K => "q6_k",
        GgufTensorType::Q8_K => "q8_k",
        GgufTensorType::IQ2_S => "iq2_s",
        GgufTensorType::I2_S => "i2_s",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::gguf::GgufValue;

    #[test]
    fn qwen_q8_attention_q_fixture_dequantizes_and_computes_cpu_reference() -> Result<()> {
        let data = build_qwen_gguf(vec![(
            "blk.0.attn_q.weight",
            vec![4, 3],
            GgufTensorType::Q8_0,
            q8_0_blob(0.5, &(1..=12).collect::<Vec<_>>()),
        )]);
        let reader = GgufReader::new(&data)?;

        let fixture = extract_dense_gguf_linear_fixture(&reader, DenseGgufTensorRole::AttentionQ)?;

        assert_eq!(fixture.summary.artifact_kind, DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND);
        assert_eq!(fixture.summary.model_family, "qwen");
        assert_eq!(fixture.summary.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(fixture.summary.tensor_type, "q8_0");
        assert_eq!(fixture.summary.source_shape, vec![4, 3]);
        assert_eq!(fixture.summary.matrix_rows, 3);
        assert_eq!(fixture.summary.matrix_cols, 4);
        assert_eq!(
            fixture.weight_values_f32[0..12],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        );
        assert_eq!(fixture.cpu_reference_input.len(), 4);
        assert_eq!(fixture.cpu_reference_output.len(), 3);
        assert_eq!(
            fixture.cpu_reference_output,
            dense_matvec_row_major(
                &fixture.weight_values_f32,
                fixture.summary.matrix_rows,
                fixture.summary.matrix_cols,
                &fixture.cpu_reference_input
            )?
        );
        assert!(fixture.summary.values_materialized_as_f32);
        assert!(fixture.summary.cpu_reference_computed);
        assert!(!fixture.summary.cpu_cuda_parity_claimed);
        assert!(!fixture.summary.dense_gguf_inference_claimed);
        assert!(!fixture.summary.bitnet_packed_i2s_qk256_proof);
        Ok(())
    }

    #[test]
    fn qwen_q8_sidecar_matvec_matches_eager_f32_fixture() -> Result<()> {
        let data = build_qwen_gguf(vec![(
            "blk.0.ffn_down.weight",
            vec![4, 3],
            GgufTensorType::Q8_0,
            q8_0_blob(0.25, &(1..=12).collect::<Vec<_>>()),
        )]);
        let reader = GgufReader::new(&data)?;

        let sidecar = extract_dense_gguf_q8_linear_sidecar(&reader, DenseGgufTensorRole::MlpDown)?;

        assert_eq!(sidecar.summary.artifact_kind, DENSE_GGUF_Q8_LINEAR_SIDECAR_ARTIFACT_KIND);
        assert_eq!(sidecar.summary.tensor_name, "blk.0.ffn_down.weight");
        assert_eq!(sidecar.summary.tensor_type, "q8_0");
        assert_eq!(sidecar.summary.matrix_rows, 3);
        assert_eq!(sidecar.summary.matrix_cols, 4);
        assert_eq!(sidecar.summary.q8_block_count, 1);
        assert!(sidecar.summary.dequantizes_inside_matvec);
        assert!(!sidecar.summary.materializes_full_f32_weights);
        assert!(sidecar.summary.compares_against_eager_f32_reference);
        assert!(sidecar.summary.generated_id_preservation_required_before_runtime_use);
        assert!(!sidecar.summary.speedup_claim);
        assert!(!sidecar.summary.dense_runtime_replaced);
        assert_eq!(sidecar.summary.max_abs_diff_vs_eager_f32, 0.0);
        assert_eq!(sidecar.fused_output, sidecar.eager_output);
        Ok(())
    }

    #[test]
    fn qwen_q8_sidecar_rejects_non_q8_fixture() -> Result<()> {
        let values: Vec<f32> = (0..12).map(|idx| idx as f32 / 8.0).collect();
        let data = build_qwen_gguf(vec![(
            "blk.0.ffn_up.weight",
            vec![4, 3],
            GgufTensorType::F16,
            f16_blob(&values),
        )]);
        let reader = GgufReader::new(&data)?;

        let err = match extract_dense_gguf_q8_linear_sidecar(&reader, DenseGgufTensorRole::MlpUp) {
            Ok(_) => {
                return Err(BitNetError::Validation(
                    "expected non-Q8 fixture to be rejected".to_string(),
                ));
            }
            Err(err) => err.to_string(),
        };

        assert!(err.contains("requires tensor type q8_0"), "unexpected error: {err}");
        Ok(())
    }

    #[test]
    fn qwen_f16_mlp_up_fixture_materializes_cpu_reference() -> Result<()> {
        let values: Vec<f32> = (0..12).map(|idx| idx as f32 / 8.0).collect();
        let data = build_qwen_gguf(vec![(
            "blk.0.ffn_up.weight",
            vec![4, 3],
            GgufTensorType::F16,
            f16_blob(&values),
        )]);
        let reader = GgufReader::new(&data)?;

        let fixture = extract_dense_gguf_linear_fixture(&reader, DenseGgufTensorRole::MlpUp)?;

        assert_eq!(fixture.summary.tensor_type, "f16");
        assert_eq!(fixture.summary.matrix_rows, 3);
        assert_eq!(fixture.summary.matrix_cols, 4);
        assert_eq!(fixture.summary.value_count, 12);
        assert_eq!(fixture.cpu_reference_output.len(), 3);
        Ok(())
    }

    #[test]
    fn non_linear_roles_are_rejected() {
        let data = build_qwen_gguf(vec![(
            "blk.0.attn_norm.weight",
            vec![4],
            GgufTensorType::F32,
            f32_blob(&[1.0, 1.0, 1.0, 1.0]),
        )]);
        let reader = GgufReader::new(&data).expect("parse qwen norm fixture");

        let err = extract_dense_gguf_linear_fixture(&reader, DenseGgufTensorRole::AttentionNorm)
            .unwrap_err()
            .to_string();

        assert!(err.contains("extractable linear role"), "unexpected error: {err}");
    }

    #[test]
    fn bitnet_packed_tensor_markers_are_rejected() {
        let data = build_qwen_gguf(vec![(
            "blk.0.attn_q.weight",
            vec![4, 3],
            GgufTensorType::I2_S,
            vec![0u8; GgufTensorType::I2_S.element_size()],
        )]);
        let reader = GgufReader::new(&data).expect("parse bitnet marker fixture");

        let err = extract_dense_gguf_linear_fixture(&reader, DenseGgufTensorRole::AttentionQ)
            .unwrap_err()
            .to_string();

        assert!(err.contains("BitNet packed tensor markers"), "unexpected error: {err}");
    }

    fn build_qwen_gguf(
        tensors: Vec<(&'static str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        build_gguf_for_test(
            vec![
                ("general.architecture", GgufValue::String("qwen3".to_string())),
                ("general.name", GgufValue::String("qwen3-linear-fixture".to_string())),
                ("qwen3.embedding_length", GgufValue::U32(4)),
                ("qwen3.feed_forward_length", GgufValue::U32(3)),
            ],
            tensors,
        )
    }

    fn build_gguf_for_test(
        metadata: Vec<(&str, GgufValue)>,
        tensors: Vec<(&str, Vec<usize>, GgufTensorType, Vec<u8>)>,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        const GGUF_VERSION: u32 = 2;
        const ALIGN: usize = 32;

        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for (key, value) in metadata {
            write_string(&mut data, key);
            write_gguf_value(&mut data, value);
        }

        let mut running_offset = 0usize;
        let mut offsets = Vec::with_capacity(tensors.len());
        for (_, _, _, blob) in &tensors {
            offsets.push(running_offset);
            running_offset += blob.len();
        }

        for (index, (name, shape, tensor_type, _blob)) in tensors.iter().enumerate() {
            write_string(&mut data, name);
            data.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                data.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            data.extend_from_slice(&tensor_type_id(*tensor_type).to_le_bytes());
            data.extend_from_slice(&(offsets[index] as u64).to_le_bytes());
        }

        let pad = (ALIGN - (data.len() % ALIGN)) % ALIGN;
        data.resize(data.len() + pad, 0);

        for (_, _, _, blob) in tensors {
            data.extend_from_slice(&blob);
        }

        data
    }

    fn q8_0_blob(scale: f32, values: &[i8]) -> Vec<u8> {
        let mut blob = Vec::new();
        let scale_bits = half::f16::from_f32(scale).to_bits();
        blob.extend_from_slice(&scale_bits.to_le_bytes());
        for idx in 0..32 {
            blob.push(values.get(idx).copied().unwrap_or(0) as u8);
        }
        blob
    }

    fn f16_blob(values: &[f32]) -> Vec<u8> {
        let mut blob = Vec::with_capacity(values.len() * 2);
        for value in values {
            blob.extend_from_slice(&half::f16::from_f32(*value).to_bits().to_le_bytes());
        }
        blob
    }

    fn f32_blob(values: &[f32]) -> Vec<u8> {
        let mut blob = Vec::with_capacity(values.len() * 4);
        for value in values {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        blob
    }

    fn write_gguf_value(data: &mut Vec<u8>, value: GgufValue) {
        match value {
            GgufValue::U32(value) => {
                data.extend_from_slice(&4u32.to_le_bytes());
                data.extend_from_slice(&value.to_le_bytes());
            }
            GgufValue::String(value) => {
                data.extend_from_slice(&8u32.to_le_bytes());
                write_string(data, &value);
            }
            other => panic!("unsupported test GGUF value: {other:?}"),
        }
    }

    fn write_string(data: &mut Vec<u8>, value: &str) {
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }

    fn tensor_type_id(tensor_type: GgufTensorType) -> u32 {
        match tensor_type {
            GgufTensorType::F32 => 0,
            GgufTensorType::F16 => 1,
            GgufTensorType::F64 => 4,
            GgufTensorType::Q4_0 => 2,
            GgufTensorType::Q4_1 => 3,
            GgufTensorType::Q5_0 => 6,
            GgufTensorType::Q5_1 => 7,
            GgufTensorType::Q8_0 => 8,
            GgufTensorType::Q8_1 => 9,
            GgufTensorType::Q2_K => 10,
            GgufTensorType::Q3_K => 11,
            GgufTensorType::Q4_K => 12,
            GgufTensorType::Q5_K => 13,
            GgufTensorType::Q6_K => 14,
            GgufTensorType::Q8_K => 15,
            GgufTensorType::IQ2_S => 24,
            GgufTensorType::I2_S => 36,
        }
    }
}
