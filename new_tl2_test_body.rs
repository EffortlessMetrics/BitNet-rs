    let config = GgufWeightLoadingTestConfig::default();
    let mock_builder = MockGgufFileBuilder::new()?.with_config(config.clone());
    let model_path = mock_builder.create_quantized_model(vec!["TL2"])?;

    let (_, tensor_map) = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu)
        .context("Failed to load TL2 test GGUF model")?;

    // Validate TL2 quantization accuracy for all FFN up weights
    for layer_idx in 0..config.test_model_layers {
        let layer_prefix = format!("blk.{}", layer_idx);
        let tensor_name = format!("{}.ffn_up.weight", layer_prefix);

        let tensor = tensor_map
            .get(&tensor_name)
            .context(format!("Missing expected tensor: {}", tensor_name))?;

        // Validate TL2 quantization maintains ≥99% accuracy
        let accuracy = validate_quantization_accuracy_tl2(tensor)
            .context(format!("TL2 quantization validation failed for {}", tensor_name))?;

        assert!(
            accuracy >= config.accuracy_threshold,
            "TL2 quantization accuracy {:.4} below threshold {:.4} for tensor {}",
            accuracy,
            config.accuracy_threshold,
            tensor_name
        );
    }

    Ok(())
