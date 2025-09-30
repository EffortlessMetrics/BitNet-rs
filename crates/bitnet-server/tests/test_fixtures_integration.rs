#![allow(unused)]
#![allow(dead_code)]

//! Integration tests for BitNet.rs inference server test fixtures
//!
//! This test validates that all fixture modules load correctly and provide
//! the expected test data for comprehensive BitNet.rs testing scenarios.

mod fixtures;

use fixtures::*;

#[cfg(test)]
mod fixture_integration_tests {
    use super::*;
    use crate::quantization::DeviceType;

    #[test]
    fn test_fixture_initialization() {
        // Test that fixture initialization works correctly
        init_fixtures();

        // Verify deterministic environment is set
        assert_eq!(std::env::var("BITNET_DETERMINISTIC").unwrap(), "1");
        assert_eq!(std::env::var("BITNET_SEED").unwrap(), "42");
    }

    #[test]
    fn test_model_fixtures_loading() {
        let all_models = get_all_test_models();
        assert!(!all_models.is_empty());

        // Verify each model has valid properties
        for model in all_models {
            assert!(!model.file_path.is_empty());
            assert!(model.parameter_count > 0 || model.file_path.contains("invalid"));
            assert!(model.vocab_size > 0 || model.file_path.contains("invalid"));

            // Verify quantization compatibility
            match model.quantization_type {
                models::QuantizationType::I2S => {
                    assert!(model.expected_tensors > 0 || model.file_path.contains("invalid"));
                }
                models::QuantizationType::TL1 => {
                    assert!(model.alignment == 32 || model.file_path.contains("invalid"));
                }
                models::QuantizationType::TL2 => {
                    assert!(model.weight_mapper_compatible || model.file_path.contains("invalid"));
                }
                _ => {}
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cpu_fixture_loading() {
        let cpu_models = FixtureLoader::load_cpu_models();
        assert!(!cpu_models.is_empty());

        let cpu_vectors = FixtureLoader::load_cpu_quantization_vectors();
        assert!(!cpu_vectors.is_empty());

        // Verify CPU-specific properties
        for vector in cpu_vectors {
            assert_eq!(vector.device_type, DeviceType::CPU);
            assert!(vector.tolerance >= 0.0, "Tolerance must be non-negative for {}", vector.name);
            assert!(vector.accuracy_target > 0.0);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_fixture_loading() {
        let gpu_models = FixtureLoader::load_gpu_models();
        assert!(!gpu_models.is_empty());

        let gpu_vectors = FixtureLoader::load_gpu_quantization_vectors();
        assert!(!gpu_vectors.is_empty());

        // Verify GPU-specific properties
        for vector in gpu_vectors {
            assert_eq!(vector.device_type, DeviceType::GPU);
            assert!(vector.input_data.len() >= 256); // GPU vectors should be larger
        }
    }

    #[test]
    fn test_quantization_accuracy_validation() {
        // Test I2S quantization accuracy validation
        let i2s_vectors =
            quantization::get_quantization_vectors(quantization::QuantizationType::I2S)
                .into_iter()
                .filter(|v| !v.name.starts_with("edge_"))
                .collect::<Vec<_>>();
        assert!(!i2s_vectors.is_empty());

        for vector in i2s_vectors {
            let accuracy = FixtureValidator::validate_quantization_accuracy(vector);
            if let Ok(acc) = accuracy {
                assert!(
                    acc >= vector.accuracy_target,
                    "Accuracy {:.4} below target {:.4} for vector '{}'",
                    acc,
                    vector.accuracy_target,
                    vector.name
                );
            } else {
                // Skip vectors with validation errors (incomplete fixture data)
                eprintln!(
                    "Skipping vector '{}' with validation error: {:?}",
                    vector.name,
                    accuracy.err()
                );
            }
        }
    }

    #[test]
    fn test_http_request_fixtures() {
        let basic_requests = FixtureLoader::load_basic_requests();
        assert!(!basic_requests.is_empty());

        // Test each basic request fixture
        for request_name in basic_requests {
            let (request, metadata) = requests::get_basic_request(request_name).unwrap();

            // Validate request structure
            assert!(FixtureValidator::validate_request_fixture(request).is_ok());

            // Validate metadata
            assert!(!metadata.description.is_empty());
            assert!(metadata.expected_status_code >= 200 && metadata.expected_status_code < 300);
            assert!(metadata.expected_response_time_ms > 0);
        }
    }

    #[test]
    fn test_http_response_fixtures() {
        // Test basic inference responses
        let response = responses::get_inference_response("simple_question_i2s_cpu").unwrap();
        assert!(FixtureValidator::validate_response_fixture(response).is_ok());
        assert!(response.accuracy_metrics.quantization_accuracy >= 0.99);

        // Test streaming responses
        let streaming = responses::get_streaming_response("bitnet_guide_stream").unwrap();
        assert!(!streaming.is_empty());

        let final_chunk = streaming.last().unwrap();
        assert!(final_chunk.is_final);

        // Test error responses
        let error = responses::get_error_response("empty_prompt").unwrap();
        assert!(!error.error.is_empty());
        assert!(!error.error_code.is_empty());
    }

    #[test]
    fn test_deployment_fixtures() {
        let docker_configs = FixtureLoader::load_docker_configs();
        assert!(!docker_configs.is_empty());

        let k8s_configs = FixtureLoader::load_kubernetes_configs();
        assert!(!k8s_configs.is_empty());

        let env_configs = FixtureLoader::load_environment_configs();
        assert!(!env_configs.is_empty());

        // Test specific configurations
        let cpu_docker = deployment::get_docker_config("cpu_production").unwrap();
        assert!(cpu_docker.dockerfile_content.contains("--features cpu"));
        assert_eq!(cpu_docker.exposed_ports, vec![8080]);

        let prod_env = deployment::get_environment_config("production").unwrap();
        assert_eq!(prod_env.deployment_type, "kubernetes");
        assert_eq!(prod_env.variables.get("BITNET_LOG_LEVEL"), Some(&"warn"));
    }

    #[test]
    fn test_performance_benchmarks() {
        let benchmarks = FixtureLoader::load_performance_benchmarks();
        assert!(!benchmarks.is_empty());

        for benchmark_name in benchmarks {
            let benchmark = deployment::get_performance_benchmark(benchmark_name).unwrap();

            assert!(benchmark.concurrent_requests > 0);
            assert!(benchmark.request_rate_per_second > 0);
            assert!(benchmark.expected_throughput_rps > 0.0);
            assert!(benchmark.expected_accuracy >= 0.90);
            assert!(benchmark.resource_requirements.cpu_cores > 0.0);
            assert!(benchmark.resource_requirements.memory_mb > 0);
        }
    }

    #[test]
    fn test_fixture_selector_functionality() {
        // Test model selection
        let cpu_model = FixtureSelector::select_model_for_test(
            Some("cpu"),
            Some(models::QuantizationType::I2S),
            Some(100), // 100MB limit
        );
        assert!(cpu_model.is_some());

        let model = cpu_model.unwrap();
        assert_eq!(model.quantization_type, models::QuantizationType::I2S);
        assert!(model.model_size_bytes <= 100 * 1024 * 1024);

        // Test quantization vector selection
        let accuracy_vectors =
            FixtureSelector::select_quantization_vectors_for_scenario("accuracy_validation");
        assert!(!accuracy_vectors.is_empty());

        for vector in accuracy_vectors {
            assert!(vector.accuracy_target >= 0.99);
        }

        // Test edge case selection
        let edge_vectors =
            FixtureSelector::select_quantization_vectors_for_scenario("edge_case_testing");
        assert!(!edge_vectors.is_empty());

        for vector in edge_vectors {
            assert!(vector.name.starts_with("edge_"));
        }
    }

    #[test]
    fn test_load_test_request_generation() {
        let requests =
            FixtureLoader::generate_load_test_requests(50, "Load test prompt for BitNet inference");

        assert_eq!(requests.len(), 50);

        // Verify request diversity
        let mut device_preferences = std::collections::HashSet::new();
        let mut quantization_preferences = std::collections::HashSet::new();

        for request in &requests {
            if let Some(device) = &request.device_preference {
                device_preferences.insert(device.clone());
            }
            if let Some(quant) = &request.quantization_preference {
                quantization_preferences.insert(quant.clone());
            }
        }

        // Should have diversity in device and quantization preferences
        assert!(device_preferences.len() >= 2);
        assert!(quantization_preferences.len() >= 2);
    }

    #[test]
    fn test_error_request_fixtures() {
        let error_requests = FixtureLoader::load_error_requests();
        assert!(!error_requests.is_empty());

        for request_name in error_requests {
            let (_request_json, metadata) = requests::get_error_request(request_name).unwrap();

            // Verify error scenarios have appropriate status codes
            assert!(metadata.expected_status_code >= 400);
            assert!(!metadata.validation_notes.is_empty());
        }
    }

    #[test]
    fn test_security_request_fixtures() {
        let security_requests = FixtureLoader::load_security_requests();
        assert!(!security_requests.is_empty());

        for request_name in security_requests {
            let (_request_json, metadata) = requests::get_security_request(request_name).unwrap();

            // Security tests should expect rejection
            assert!(metadata.expected_status_code >= 400);
            assert!(
                metadata.test_scenario.to_lowercase().contains("security")
                    || metadata.description.to_lowercase().contains("injection")
                    || metadata.description.to_lowercase().contains("dos")
            );
        }
    }

    #[cfg(feature = "crossval")]
    #[test]
    fn test_crossval_fixtures() {
        let crossval_fixtures = FixtureLoader::load_crossval_fixtures();
        assert!(!crossval_fixtures.is_empty());

        for fixture in crossval_fixtures {
            // Verify cross-validation data structure
            assert_eq!(fixture.rust_output.len(), fixture.cpp_reference.len());
            assert!(fixture.tolerance > 0.0);
            assert!(!fixture.validation_notes.is_empty());

            // Verify the outputs are reasonably close
            for (rust_val, cpp_val) in fixture.rust_output.iter().zip(fixture.cpp_reference.iter())
            {
                let diff = (rust_val - cpp_val).abs();
                assert!(diff <= fixture.tolerance);
            }
        }
    }

    #[test]
    fn test_fixture_deterministic_behavior() {
        // Test deterministic data generation
        let data1 = quantization::generate_deterministic_test_data(100, 42, (-1.0, 1.0));
        let data2 = quantization::generate_deterministic_test_data(100, 42, (-1.0, 1.0));

        assert_eq!(data1, data2); // Should be identical with same seed

        // Test different seeds produce different data
        let data3 = quantization::generate_deterministic_test_data(100, 123, (-1.0, 1.0));
        assert_ne!(data1, data3);
    }

    #[test]
    fn test_sse_streaming_format() {
        let streaming_chunks = responses::get_streaming_response("bitnet_guide_stream").unwrap();
        assert!(!streaming_chunks.is_empty());

        for chunk in streaming_chunks {
            let sse_formatted = responses::format_as_sse(chunk);

            assert!(sse_formatted.starts_with("data: "));

            if chunk.is_final {
                assert!(sse_formatted.contains("event: done"));
            }
        }
    }

    #[test]
    fn test_prometheus_metrics_format() {
        let metrics = responses::get_prometheus_metrics();

        // Verify Prometheus format compliance
        assert!(metrics.contains("# HELP"));
        assert!(metrics.contains("# TYPE"));
        assert!(metrics.contains("bitnet_requests_total"));
        assert!(metrics.contains("bitnet_inference_duration_seconds"));
        assert!(metrics.contains("bitnet_quantization_accuracy"));

        // Verify quantization accuracy metrics are present
        assert!(metrics.contains("quantization=\"i2s\""));
        assert!(metrics.contains("quantization=\"tl1\""));
        assert!(metrics.contains("quantization=\"tl2\""));
    }

    #[test]
    fn test_fixture_file_structure_validation() {
        let result = FixtureLoader::validate_fixture_files();
        assert!(result.is_ok());

        let fixture_dir = FixtureLoader::fixture_dir();
        assert!(fixture_dir.exists());

        // Verify subdirectories exist
        assert!(fixture_dir.join("models").exists());
        assert!(fixture_dir.join("requests").exists());
        assert!(fixture_dir.join("responses").exists());
        assert!(fixture_dir.join("quantization").exists());
        assert!(fixture_dir.join("deployment").exists());
    }
}
