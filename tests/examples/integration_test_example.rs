//! Example integration test demonstrating workflow testing
//!
//! This example shows how to write integration tests that validate
//! complete workflows and component interactions in BitNet.rs.

use bitnet_tests::common::{TestError, TestUtilities};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs;

/// Example system that processes data through multiple components
pub struct DataProcessingSystem {
    loader: DataLoader,
    processor: DataProcessor,
    writer: DataWriter,
    config: SystemConfig,
}

#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub input_format: String,
    pub output_format: String,
    pub processing_mode: ProcessingMode,
    pub max_items: usize,
}

#[derive(Debug, Clone)]
pub enum ProcessingMode {
    Fast,
    Accurate,
    Balanced,
}

pub struct DataLoader {
    supported_formats: Vec<String>,
}

pub struct DataProcessor {
    mode: ProcessingMode,
}

pub struct DataWriter {
    output_format: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataItem {
    pub id: u64,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub items: Vec<DataItem>,
    pub statistics: ProcessingStatistics,
}

#[derive(Debug, Clone)]
pub struct ProcessingStatistics {
    pub items_processed: usize,
    pub processing_time: Duration,
    pub errors_encountered: usize,
}

#[derive(Debug)]
pub enum SystemError {
    LoadError(String),
    ProcessError(String),
    WriteError(String),
    ConfigError(String),
}

impl DataProcessingSystem {
    pub async fn new(config: SystemConfig) -> Result<Self, SystemError> {
        // Validate configuration
        if config.max_items == 0 {
            return Err(SystemError::ConfigError("max_items cannot be zero".to_string()));
        }

        let loader = DataLoader {
            supported_formats: vec!["json".to_string(), "csv".to_string(), "xml".to_string()],
        };

        let processor = DataProcessor { mode: config.processing_mode.clone() };

        let writer = DataWriter { output_format: config.output_format.clone() };

        Ok(Self { loader, processor, writer, config })
    }

    pub async fn process_file(
        &mut self,
        input_path: &Path,
        output_path: &Path,
    ) -> Result<ProcessingResult, SystemError> {
        let start_time = std::time::Instant::now();

        // Step 1: Load data
        let data = self.loader.load_data(input_path, &self.config.input_format).await?;

        // Step 2: Process data
        let processed_data = self.processor.process_data(data, self.config.max_items).await?;

        // Step 3: Write results
        self.writer.write_data(&processed_data, output_path).await?;

        let processing_time = start_time.elapsed();

        Ok(ProcessingResult {
            items: processed_data,
            statistics: ProcessingStatistics {
                items_processed: processed_data.len(),
                processing_time,
                errors_encountered: 0,
            },
        })
    }

    pub async fn get_supported_formats(&self) -> Vec<String> {
        self.loader.supported_formats.clone()
    }

    pub async fn validate_config(&self) -> Result<(), SystemError> {
        if !self.loader.supported_formats.contains(&self.config.input_format) {
            return Err(SystemError::ConfigError(format!(
                "Unsupported input format: {}",
                self.config.input_format
            )));
        }

        Ok(())
    }
}

impl DataLoader {
    pub async fn load_data(&self, path: &Path, format: &str) -> Result<Vec<DataItem>, SystemError> {
        if !self.supported_formats.contains(&format.to_string()) {
            return Err(SystemError::LoadError(format!("Unsupported format: {}", format)));
        }

        if !path.exists() {
            return Err(SystemError::LoadError(format!("File not found: {:?}", path)));
        }

        // Simulate loading data based on format
        let content = fs::read_to_string(path)
            .await
            .map_err(|e| SystemError::LoadError(format!("Failed to read file: {}", e)))?;

        match format {
            "json" => self.load_json(&content).await,
            "csv" => self.load_csv(&content).await,
            _ => Err(SystemError::LoadError(format!("Format not implemented: {}", format))),
        }
    }

    async fn load_json(&self, content: &str) -> Result<Vec<DataItem>, SystemError> {
        // Simulate JSON parsing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Simple JSON-like parsing for demo
        if content.trim().is_empty() {
            return Ok(vec![]);
        }

        let items = content
            .lines()
            .enumerate()
            .map(|(i, line)| DataItem {
                id: i as u64,
                content: line.trim().to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("format".to_string(), "json".to_string());
                    meta.insert("line".to_string(), i.to_string());
                    meta
                },
            })
            .collect();

        Ok(items)
    }

    async fn load_csv(&self, content: &str) -> Result<Vec<DataItem>, SystemError> {
        // Simulate CSV parsing
        tokio::time::sleep(Duration::from_millis(15)).await;

        let items = content
            .lines()
            .enumerate()
            .map(|(i, line)| DataItem {
                id: i as u64,
                content: line.replace(',', " | "), // Simple CSV transformation
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("format".to_string(), "csv".to_string());
                    meta.insert("columns".to_string(), line.split(',').count().to_string());
                    meta
                },
            })
            .collect();

        Ok(items)
    }
}

impl DataProcessor {
    pub async fn process_data(
        &self,
        mut data: Vec<DataItem>,
        max_items: usize,
    ) -> Result<Vec<DataItem>, SystemError> {
        // Limit items if necessary
        if data.len() > max_items {
            data.truncate(max_items);
        }

        // Process based on mode
        match self.mode {
            ProcessingMode::Fast => self.fast_process(data).await,
            ProcessingMode::Accurate => self.accurate_process(data).await,
            ProcessingMode::Balanced => self.balanced_process(data).await,
        }
    }

    async fn fast_process(&self, mut data: Vec<DataItem>) -> Result<Vec<DataItem>, SystemError> {
        tokio::time::sleep(Duration::from_millis(5)).await;

        for item in &mut data {
            item.content = format!("FAST: {}", item.content);
            item.metadata.insert("processing_mode".to_string(), "fast".to_string());
        }

        Ok(data)
    }

    async fn accurate_process(
        &self,
        mut data: Vec<DataItem>,
    ) -> Result<Vec<DataItem>, SystemError> {
        tokio::time::sleep(Duration::from_millis(20)).await;

        for item in &mut data {
            item.content = format!("ACCURATE: {}", item.content.to_uppercase());
            item.metadata.insert("processing_mode".to_string(), "accurate".to_string());
            item.metadata.insert(
                "processed_at".to_string(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    .to_string(),
            );
        }

        Ok(data)
    }

    async fn balanced_process(
        &self,
        mut data: Vec<DataItem>,
    ) -> Result<Vec<DataItem>, SystemError> {
        tokio::time::sleep(Duration::from_millis(10)).await;

        for item in &mut data {
            item.content = format!("BALANCED: {}", item.content);
            item.metadata.insert("processing_mode".to_string(), "balanced".to_string());
        }

        Ok(data)
    }
}

impl DataWriter {
    pub async fn write_data(
        &self,
        data: &[DataItem],
        output_path: &Path,
    ) -> Result<(), SystemError> {
        // Create parent directories if they don't exist
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                SystemError::WriteError(format!("Failed to create directories: {}", e))
            })?;
        }

        match self.output_format.as_str() {
            "json" => self.write_json(data, output_path).await,
            "txt" => self.write_text(data, output_path).await,
            _ => Err(SystemError::WriteError(format!(
                "Unsupported output format: {}",
                self.output_format
            ))),
        }
    }

    async fn write_json(&self, data: &[DataItem], output_path: &Path) -> Result<(), SystemError> {
        let mut content = String::new();
        content.push_str("[\n");

        for (i, item) in data.iter().enumerate() {
            content.push_str(&format!(
                "  {{\"id\": {}, \"content\": \"{}\", \"metadata\": {{",
                item.id, item.content
            ));

            for (j, (key, value)) in item.metadata.iter().enumerate() {
                content.push_str(&format!("\"{}\": \"{}\"", key, value));
                if j < item.metadata.len() - 1 {
                    content.push_str(", ");
                }
            }

            content.push_str("}}");
            if i < data.len() - 1 {
                content.push_str(",");
            }
            content.push_str("\n");
        }

        content.push_str("]\n");

        fs::write(output_path, content)
            .await
            .map_err(|e| SystemError::WriteError(format!("Failed to write JSON: {}", e)))?;

        Ok(())
    }

    async fn write_text(&self, data: &[DataItem], output_path: &Path) -> Result<(), SystemError> {
        let mut content = String::new();

        for item in data {
            content.push_str(&format!("ID: {} | Content: {}\n", item.id, item.content));
        }

        fs::write(output_path, content)
            .await
            .map_err(|e| SystemError::WriteError(format!("Failed to write text: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config() -> SystemConfig {
        SystemConfig {
            input_format: "json".to_string(),
            output_format: "json".to_string(),
            processing_mode: ProcessingMode::Balanced,
            max_items: 100,
        }
    }

    #[tokio::test]
    async fn test_complete_processing_workflow() {
        // Setup temporary directories and files
        let temp_dir = TempDir::new().unwrap();
        let input_file = temp_dir.path().join("input.json");
        let output_file = temp_dir.path().join("output.json");

        // Create test input data
        let test_data = "line 1\nline 2\nline 3\n";
        TestUtilities::write_test_file(&input_file, test_data.as_bytes()).await.unwrap();

        // Create and configure system
        let config = create_test_config();
        let mut system = DataProcessingSystem::new(config).await.unwrap();

        // Execute complete workflow
        let result = system.process_file(&input_file, &output_file).await;

        // Verify workflow succeeded
        assert!(result.is_ok(), "Workflow should complete successfully");
        let processing_result = result.unwrap();

        // Verify processing results
        assert_eq!(processing_result.items.len(), 3, "Should process 3 items");
        assert_eq!(processing_result.statistics.items_processed, 3);
        assert!(processing_result.statistics.processing_time > Duration::from_millis(0));

        // Verify output file was created
        assert!(output_file.exists(), "Output file should be created");

        // Verify output content
        let output_content = TestUtilities::read_test_file(&output_file).await.unwrap();
        let output_str = String::from_utf8(output_content).unwrap();

        assert!(output_str.contains("BALANCED: line 1"), "Output should contain processed content");
        assert!(
            output_str.contains("\"processing_mode\": \"balanced\""),
            "Output should contain metadata"
        );
    }

    #[tokio::test]
    async fn test_component_interaction_data_flow() {
        let temp_dir = TempDir::new().unwrap();
        let input_file = temp_dir.path().join("test_input.json");
        let output_file = temp_dir.path().join("test_output.json");

        // Create test data with specific content
        let test_content = "hello world\ntest data\nfinal line";
        TestUtilities::write_test_file(&input_file, test_content.as_bytes()).await.unwrap();

        let config = SystemConfig {
            input_format: "json".to_string(),
            output_format: "json".to_string(),
            processing_mode: ProcessingMode::Accurate,
            max_items: 10,
        };

        let mut system = DataProcessingSystem::new(config).await.unwrap();

        // Test the complete data flow
        let result = system.process_file(&input_file, &output_file).await.unwrap();

        // Verify data transformations at each step
        assert_eq!(result.items.len(), 3);

        // Check that data was processed correctly by each component
        for item in &result.items {
            // Loader should have added format metadata
            assert_eq!(item.metadata.get("format").unwrap(), "json");

            // Processor should have transformed content and added processing metadata
            assert!(item.content.starts_with("ACCURATE: "));
            assert!(item.content.contains(&item.content.to_uppercase()));
            assert_eq!(item.metadata.get("processing_mode").unwrap(), "accurate");
            assert!(item.metadata.contains_key("processed_at"));
        }

        // Writer should have created valid output
        let output_content = fs::read_to_string(&output_file).await.unwrap();
        assert!(output_content.starts_with("["));
        assert!(output_content.ends_with("]\n"));
    }

    #[tokio::test]
    async fn test_different_processing_modes() {
        let temp_dir = TempDir::new().unwrap();
        let input_file = temp_dir.path().join("mode_test.json");
        let test_data = "test content";
        TestUtilities::write_test_file(&input_file, test_data.as_bytes()).await.unwrap();

        let modes = vec![
            (ProcessingMode::Fast, "FAST: test content"),
            (ProcessingMode::Accurate, "ACCURATE: TEST CONTENT"),
            (ProcessingMode::Balanced, "BALANCED: test content"),
        ];

        for (mode, expected_prefix) in modes {
            let output_file = temp_dir.path().join(format!("output_{:?}.json", mode));

            let config = SystemConfig {
                input_format: "json".to_string(),
                output_format: "json".to_string(),
                processing_mode: mode,
                max_items: 100,
            };

            let mut system = DataProcessingSystem::new(config).await.unwrap();
            let result = system.process_file(&input_file, &output_file).await.unwrap();

            assert_eq!(result.items.len(), 1);
            assert_eq!(result.items[0].content, expected_prefix);
        }
    }

    #[tokio::test]
    async fn test_error_handling_file_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let nonexistent_file = temp_dir.path().join("does_not_exist.json");
        let output_file = temp_dir.path().join("output.json");

        let config = create_test_config();
        let mut system = DataProcessingSystem::new(config).await.unwrap();

        let result = system.process_file(&nonexistent_file, &output_file).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            SystemError::LoadError(msg) => {
                assert!(msg.contains("File not found"));
            }
            other => panic!("Expected LoadError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_error_handling_unsupported_format() {
        let temp_dir = TempDir::new().unwrap();
        let input_file = temp_dir.path().join("test.json");
        let output_file = temp_dir.path().join("output.json");

        TestUtilities::write_test_file(&input_file, b"test data").await.unwrap();

        let config = SystemConfig {
            input_format: "unsupported_format".to_string(),
            output_format: "json".to_string(),
            processing_mode: ProcessingMode::Fast,
            max_items: 100,
        };

        let mut system = DataProcessingSystem::new(config).await.unwrap();
        let result = system.process_file(&input_file, &output_file).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            SystemError::LoadError(msg) => {
                assert!(msg.contains("Unsupported format"));
            }
            other => panic!("Expected LoadError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        // Test valid configuration
        let valid_config = create_test_config();
        let system = DataProcessingSystem::new(valid_config).await;
        assert!(system.is_ok());

        // Test invalid configuration (max_items = 0)
        let invalid_config = SystemConfig {
            input_format: "json".to_string(),
            output_format: "json".to_string(),
            processing_mode: ProcessingMode::Fast,
            max_items: 0,
        };

        let result = DataProcessingSystem::new(invalid_config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            SystemError::ConfigError(msg) => {
                assert!(msg.contains("max_items cannot be zero"));
            }
            other => panic!("Expected ConfigError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_max_items_limiting() {
        let temp_dir = TempDir::new().unwrap();
        let input_file = temp_dir.path().join("large_input.json");
        let output_file = temp_dir.path().join("limited_output.json");

        // Create input with more items than the limit
        let large_input =
            "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10";
        TestUtilities::write_test_file(&input_file, large_input.as_bytes()).await.unwrap();

        let config = SystemConfig {
            input_format: "json".to_string(),
            output_format: "json".to_string(),
            processing_mode: ProcessingMode::Fast,
            max_items: 5, // Limit to 5 items
        };

        let mut system = DataProcessingSystem::new(config).await.unwrap();
        let result = system.process_file(&input_file, &output_file).await.unwrap();

        // Should only process 5 items despite having 10 in input
        assert_eq!(result.items.len(), 5);
        assert_eq!(result.statistics.items_processed, 5);
    }

    #[tokio::test]
    async fn test_different_input_output_formats() {
        let temp_dir = TempDir::new().unwrap();
        let csv_input = temp_dir.path().join("input.csv");
        let txt_output = temp_dir.path().join("output.txt");

        // Create CSV input
        let csv_data = "name,age,city\nJohn,30,NYC\nJane,25,LA";
        TestUtilities::write_test_file(&csv_input, csv_data.as_bytes()).await.unwrap();

        let config = SystemConfig {
            input_format: "csv".to_string(),
            output_format: "txt".to_string(),
            processing_mode: ProcessingMode::Balanced,
            max_items: 100,
        };

        let mut system = DataProcessingSystem::new(config).await.unwrap();
        let result = system.process_file(&csv_input, &txt_output).await.unwrap();

        // Verify CSV was processed correctly
        assert_eq!(result.items.len(), 3); // Header + 2 data rows

        // Check that CSV-specific metadata was added
        for item in &result.items {
            assert_eq!(item.metadata.get("format").unwrap(), "csv");
            assert!(item.metadata.contains_key("columns"));
        }

        // Verify text output format
        let output_content = fs::read_to_string(&txt_output).await.unwrap();
        assert!(output_content.contains("ID: "));
        assert!(output_content.contains("Content: BALANCED:"));
        assert!(output_content.contains(" | ")); // CSV transformation marker
    }

    #[tokio::test]
    async fn test_performance_requirements() {
        let temp_dir = TempDir::new().unwrap();
        let input_file = temp_dir.path().join("perf_test.json");
        let output_file = temp_dir.path().join("perf_output.json");

        // Create moderately sized test data
        let test_lines: Vec<String> = (0..50).map(|i| format!("test line {}", i)).collect();
        let test_data = test_lines.join("\n");
        TestUtilities::write_test_file(&input_file, test_data.as_bytes()).await.unwrap();

        let config = create_test_config();
        let mut system = DataProcessingSystem::new(config).await.unwrap();

        // Measure processing time
        let start = std::time::Instant::now();
        let result = system.process_file(&input_file, &output_file).await.unwrap();
        let total_time = start.elapsed();

        // Verify performance requirements
        assert!(
            total_time < Duration::from_secs(1),
            "Processing should complete within 1 second, took {:?}",
            total_time
        );

        assert_eq!(result.items.len(), 50);
        assert!(
            result.statistics.processing_time < Duration::from_millis(500),
            "Processing component should be fast, took {:?}",
            result.statistics.processing_time
        );
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        // Test that multiple systems can work concurrently
        let temp_dir = TempDir::new().unwrap();
        let mut handles = vec![];

        for i in 0..5 {
            let input_file = temp_dir.path().join(format!("input_{}.json", i));
            let output_file = temp_dir.path().join(format!("output_{}.json", i));
            let test_data = format!("concurrent test data {}", i);

            TestUtilities::write_test_file(&input_file, test_data.as_bytes()).await.unwrap();

            let handle = tokio::spawn(async move {
                let config = create_test_config();
                let mut system = DataProcessingSystem::new(config).await.unwrap();
                system.process_file(&input_file, &output_file).await
            });

            handles.push(handle);
        }

        // Wait for all concurrent operations to complete
        let mut results = vec![];
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            results.push(result.unwrap());
        }

        // Verify all operations completed successfully
        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.items.len(), 1);
            assert!(result.items[0].content.contains(&format!("concurrent test data {}", i)));
        }
    }

    #[tokio::test]
    async fn test_system_state_consistency() {
        // Test that system maintains consistent state across operations
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let mut system = DataProcessingSystem::new(config).await.unwrap();

        // Verify initial state
        let formats = system.get_supported_formats().await;
        assert!(formats.contains(&"json".to_string()));
        assert!(formats.contains(&"csv".to_string()));

        // Perform multiple operations
        for i in 0..3 {
            let input_file = temp_dir.path().join(format!("test_{}.json", i));
            let output_file = temp_dir.path().join(format!("result_{}.json", i));
            let test_data = format!("test data iteration {}", i);

            TestUtilities::write_test_file(&input_file, test_data.as_bytes()).await.unwrap();

            let result = system.process_file(&input_file, &output_file).await;
            assert!(result.is_ok(), "Operation {} should succeed", i);
        }

        // Verify system state is still consistent
        let formats_after = system.get_supported_formats().await;
        assert_eq!(formats, formats_after, "Supported formats should remain consistent");

        let validation_result = system.validate_config().await;
        assert!(validation_result.is_ok(), "Configuration should remain valid");
    }
}

// Example of how to run these integration tests:
// cargo test --test integration_test_example
//
// To run a specific test:
// cargo test --test integration_test_example test_complete_processing_workflow
//
// To run with detailed output:
// cargo test --test integration_test_example -- --nocapture --test-threads=1
