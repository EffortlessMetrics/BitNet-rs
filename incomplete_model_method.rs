    /// Create an incomplete GGUF file missing required tensors for error handling tests
    /// This creates a valid GGUF file but omits critical transformer layer tensors
    pub fn create_incomplete_model(&self) -> Result<PathBuf> {
        let model_path = self.temp_dir.path().join("test_incomplete_model.gguf");

        // Use bitnet-st2gguf writer to create a real but incomplete GGUF file
        let mut writer = bitnet_st2gguf::writer::GgufWriter::new();

        // Add metadata with model configuration matching test config
        writer.add_metadata(
            "llama.embedding_length",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.hidden_size as u32),
        );
        writer.add_metadata(
            "llama.block_count",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.test_model_layers as u32),
        );
        writer.add_metadata(
            "llama.attention.head_count",
            bitnet_st2gguf::writer::MetadataValue::U32(32),
        );
        writer.add_metadata(
            "llama.attention.head_count_kv",
            bitnet_st2gguf::writer::MetadataValue::U32(8),
        );
        writer.add_metadata(
            "llama.feed_forward_length",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.intermediate_size as u32),
        );
        writer.add_metadata(
            "llama.vocab_size",
            bitnet_st2gguf::writer::MetadataValue::U32(self.config.vocab_size as u32),
        );

        // Create F16 tensors with non-zero deterministic data
        use half::f16;

        // Token embeddings: [vocab_size, hidden_size] - INCLUDE
        let tok_emb_data: Vec<f32> = (0..(self.config.vocab_size * self.config.hidden_size))
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let tok_emb_f16: Vec<f16> = tok_emb_data.iter().map(|&f| f16::from_f32(f)).collect();
        let tok_emb_bytes = bytemuck::cast_slice(&tok_emb_f16).to_vec();
        writer.add_tensor(bitnet_st2gguf::writer::TensorEntry::new(
            "token_embd.weight".to_string(),
            vec![self.config.vocab_size as u64, self.config.hidden_size as u64],
            bitnet_st2gguf::writer::TensorDType::F16,
            tok_emb_bytes,
        ));

        // Output projection: [hidden_size, vocab_size] - INCLUDE
        let output_data: Vec<f32> = (0..(self.config.hidden_size * self.config.vocab_size))
            .map(|i| (i as f32 * 0.002).cos() * 0.1)
            .collect();
        let output_f16: Vec<f16> = output_data.iter().map(|&f| f16::from_f32(f)).collect();
        let output_bytes = bytemuck::cast_slice(&output_f16).to_vec();
        writer.add_tensor(bitnet_st2gguf::writer::TensorEntry::new(
            "output.weight".to_string(),
            vec![self.config.hidden_size as u64, self.config.vocab_size as u64],
            bitnet_st2gguf::writer::TensorDType::F16,
            output_bytes,
        ));

        // Add ONLY layer 0 attention Q weight (skip K, V, Output to create incompleteness)
        // This will trigger missing tensor detection for blk.0.attn_k.weight
        let layer_prefix = "blk.0";
        let data: Vec<f32> = (0..(self.config.hidden_size * self.config.hidden_size))
            .map(|i| (i as f32 * 0.003).sin() * 0.1)
            .collect();
        let data_f16: Vec<f16> = data.iter().map(|&f| f16::from_f32(f)).collect();
        let data_bytes = bytemuck::cast_slice(&data_f16).to_vec();
        writer.add_tensor(bitnet_st2gguf::writer::TensorEntry::new(
            format!("{}.attn_q.weight", layer_prefix),
            vec![self.config.hidden_size as u64, self.config.hidden_size as u64],
            bitnet_st2gguf::writer::TensorDType::F16,
            data_bytes,
        ));

        // Deliberately SKIP blk.0.attn_k.weight, blk.0.attn_v.weight, blk.0.attn_output.weight
        // Deliberately SKIP all FFN and normalization tensors for layer 0
        // Deliberately SKIP all tensors for layers 1+

        // Output normalization: [hidden_size] - INCLUDE
        let out_norm_data: Vec<f32> =
            (0..self.config.hidden_size).map(|i| 1.0 + (i as f32 * 0.001).sin() * 0.05).collect();
        let out_norm_f16: Vec<f16> = out_norm_data.iter().map(|&f| f16::from_f32(f)).collect();
        let out_norm_bytes = bytemuck::cast_slice(&out_norm_f16).to_vec();
        writer.add_tensor(bitnet_st2gguf::writer::TensorEntry::new(
            "output_norm.weight".to_string(),
            vec![self.config.hidden_size as u64],
            bitnet_st2gguf::writer::TensorDType::F16,
            out_norm_bytes,
        ));

        // Write GGUF file to disk
        writer.write_to_file(&model_path).context("Failed to write incomplete GGUF file")?;

        Ok(model_path)
    }
