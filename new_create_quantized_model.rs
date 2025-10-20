    /// Create GGUF file with specific quantization types
    /// Note: Creates FP16 tensors that will be quantized by validation helpers
    pub fn create_quantized_model(&self, quantization_types: Vec<&str>) -> Result<PathBuf> {
        let model_path = self.temp_dir.path().join(format!(
            "test_quantized_{}_model.gguf",
            quantization_types.join("_")
        ));

        // Create a real GGUF file with F16 tensors
        let mut writer = bitnet_st2gguf::writer::GgufWriter::new();

        // Add metadata
        writer.add_metadata(
            "llama.embedding_length",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.hidden_size as u32),
        );
        writer.add_metadata(
            "llama.block_count",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.test_model_layers as u32),
        );
        writer.add_metadata(
            "llama.feed_forward_length",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.intermediate_size as u32),
        );

        use half::f16;

        // Create F16 tensors for transformer layers based on requested quantization type
        for layer_idx in 0..self.config.test_model_layers {
            let layer_prefix = format!("blk.{}", layer_idx);
            let layer_offset = layer_idx as f32 * 0.1;

            for quant_type in &quantization_types {
                let (weight_names, tensor_size) = match *quant_type {
                    "I2_S" => (vec!["attn_q.weight"], self.config.hidden_size * self.config.hidden_size),
                    "TL1" => (vec!["ffn_gate.weight"], self.config.hidden_size * self.config.intermediate_size),
                    "TL2" => (vec!["ffn_up.weight"], self.config.hidden_size * self.config.intermediate_size),
                    _ => continue,
                };

                for weight_name in weight_names {
                    let data: Vec<f32> = (0..tensor_size)
                        .map(|i| (i as f32 * 0.003 + layer_offset).sin() * 0.1)
                        .collect();
                    let f16_data: Vec<f16> = data.iter().map(|&f| f16::from_f32(f)).collect();
                    let data_bytes = bytemuck::cast_slice(&f16_data).to_vec();

                    let shape = if weight_name.contains("ffn") {
                        vec![self.config.hidden_size as u64, self.config.intermediate_size as u64]
                    } else {
                        vec![self.config.hidden_size as u64, self.config.hidden_size as u64]
                    };

                    writer.add_tensor(bitnet_st2gguf::writer::TensorEntry::new(
                        format!("{}.{}", layer_prefix, weight_name),
                        shape,
                        bitnet_st2gguf::writer::TensorDType::F16,
                        data_bytes,
                    ));
                }
            }
        }

        // Write GGUF file to disk
        writer.write_to_file(&model_path).context("Failed to write quantized GGUF file")?;

        Ok(model_path)
    }
