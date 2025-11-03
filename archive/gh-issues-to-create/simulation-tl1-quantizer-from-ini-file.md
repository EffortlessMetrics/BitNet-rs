# Simulation: `TL1Quantizer::from_ini_file` in `tl1.rs` is a simplified implementation

The `TL1Quantizer::from_ini_file` function in `crates/bitnet-quantization/src/tl1.rs` has a comment "Simplified ini parsing - in practice would use a proper ini parser". It performs a simplified INI parsing. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/tl1.rs`

**Function:** `TL1Quantizer::from_ini_file`

**Code:**
```rust
    pub fn from_ini_file(path: &str) -> Result<Self> {
        // Simplified ini parsing - in practice would use a proper ini parser
        let mut config = TL1Config::default();

        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with("block_size=") {
                    if let Ok(size) = line.split('=').nth(1).unwrap_or("64").parse() {
                        config.block_size = size;
                    }
                } else if line.starts_with("lookup_table_size=") {
                    if let Ok(size) = line.split('=').nth(1).unwrap_or("256").parse() {
                        config.lookup_table_size = size;
                    }
                } else if line.starts_with("use_asymmetric=") {
                    config.use_asymmetric = line.split('=').nth(1).unwrap_or("false") == "true";
                } else if line.starts_with("precision_bits=")
                    && let Ok(bits) = line.split('=').nth(1).unwrap_or("2").parse()
                {
                    config.precision_bits = bits;
                }
            }
        }

        Ok(Self::with_config(config))
    }
```

## Proposed Fix

The `TL1Quantizer::from_ini_file` function should be implemented to use a proper INI parser. This would involve using a library like `ini` to parse the INI file and extract the configuration parameters.

### Example Implementation

```rust
    pub fn from_ini_file(path: &str) -> Result<Self> {
        let config_file = ini::Ini::load_from_file(path)?;
        let section = config_file.section(Some("tl1_config")).unwrap();

        let config = TL1Config {
            block_size: section.get("block_size").unwrap_or("64").parse()?,
            lookup_table_size: section.get("lookup_table_size").unwrap_or("256").parse()?,
            use_asymmetric: section.get("use_asymmetric").unwrap_or("false") == "true",
            precision_bits: section.get("precision_bits").unwrap_or("2").parse()?,
        };

        Ok(Self::with_config(config))
    }
```
