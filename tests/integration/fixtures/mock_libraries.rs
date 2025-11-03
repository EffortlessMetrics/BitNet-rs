/// Mock Library Generator for Integration Test Fixtures
///
/// Generates platform-specific stub library files with correct file headers
/// for build.rs detection without requiring actual C++ compilation.
///
/// # Test Specification
///
/// Tests feature spec: bitnet-integration-tests.md#mock-library-generator
use super::FixtureError;
use std::path::PathBuf;

/// Platform types for library generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Linux (.so shared libraries with ELF headers)
    Linux,
    /// macOS (.dylib dynamic libraries with Mach-O headers)
    MacOS,
    /// Windows (.lib static libraries with PE headers)
    Windows,
}

/// Library types for mock generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryType {
    /// BitNet library (libbitnet.{so,dylib,lib})
    BitNet,
    /// Llama library (libllama.{so,dylib,lib})
    Llama,
    /// GGML library (libggml.{so,dylib,lib})
    Ggml,
}

/// Mock library generator
pub struct MockLibrary {
    lib_type: LibraryType,
    platform: Platform,
    output_path: PathBuf,
}

impl MockLibrary {
    /// Create a new mock library generator
    pub fn new(lib_type: LibraryType, platform: Platform) -> Self {
        Self { lib_type, platform, output_path: PathBuf::new() }
    }

    /// Set the output directory path
    pub fn at_path(mut self, path: PathBuf) -> Self {
        self.output_path = path;
        self
    }

    /// Generate the mock library file
    ///
    /// Creates a stub library file with correct platform-specific header:
    /// - Linux: ELF header (magic bytes `\x7fELF`)
    /// - macOS: Mach-O header (magic bytes `0xFEEDFACE`)
    /// - Windows: PE header (magic bytes `MZ`)
    ///
    /// # Returns
    ///
    /// - `Ok(())`: Library file created successfully
    /// - `Err(FixtureError)`: Failed to create library file
    ///
    /// # Test Coverage
    ///
    /// Tests feature spec: bitnet-integration-tests.md#platform-specific-scenarios
    pub fn generate(&self) -> Result<(), FixtureError> {
        let filename = Self::library_filename(self.lib_type, self.platform);
        let full_path = self.output_path.join(&filename);

        // Ensure output directory exists
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = match self.platform {
            Platform::Linux => Self::generate_elf_stub(),
            Platform::MacOS => Self::generate_macho_stub(),
            Platform::Windows => Self::generate_pe_stub(),
        };

        std::fs::write(&full_path, content).map_err(|e| {
            FixtureError::LibraryGeneration(format!(
                "Failed to write mock library {}: {}",
                full_path.display(),
                e
            ))
        })?;

        Ok(())
    }

    /// Get the library filename for a given type and platform
    pub fn library_filename(lib_type: LibraryType, platform: Platform) -> String {
        let (prefix, base_name, extension) = match platform {
            Platform::Linux => ("lib", Self::base_name(lib_type), "so"),
            Platform::MacOS => ("lib", Self::base_name(lib_type), "dylib"),
            Platform::Windows => ("", Self::base_name_windows(lib_type), "lib"),
        };

        format!("{}{}.{}", prefix, base_name, extension)
    }

    /// Get base library name (Linux/macOS)
    fn base_name(lib_type: LibraryType) -> &'static str {
        match lib_type {
            LibraryType::BitNet => "bitnet",
            LibraryType::Llama => "llama",
            LibraryType::Ggml => "ggml",
        }
    }

    /// Get base library name (Windows - no lib prefix)
    fn base_name_windows(lib_type: LibraryType) -> &'static str {
        match lib_type {
            LibraryType::BitNet => "bitnet",
            LibraryType::Llama => "llama",
            LibraryType::Ggml => "ggml",
        }
    }

    /// Generate minimal ELF header for Linux .so files
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-17-linux-so-libraries
    fn generate_elf_stub() -> Vec<u8> {
        // Minimal ELF64 header (64 bytes)
        // Reference: https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
        let mut header = vec![0u8; 64];

        // ELF magic number
        header[0..4].copy_from_slice(b"\x7fELF");

        // EI_CLASS: 64-bit
        header[4] = 2;

        // EI_DATA: Little-endian
        header[5] = 1;

        // EI_VERSION: Current version
        header[6] = 1;

        // EI_OSABI: System V
        header[7] = 0;

        // e_type: ET_DYN (shared object)
        header[16] = 0x03;
        header[17] = 0x00;

        // e_machine: x86-64
        header[18] = 0x3E;
        header[19] = 0x00;

        header
    }

    /// Generate minimal Mach-O header for macOS .dylib files
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-18-macos-dylib-libraries
    fn generate_macho_stub() -> Vec<u8> {
        // Minimal Mach-O 64-bit header (32 bytes)
        // Reference: https://opensource.apple.com/source/xnu/xnu-2050.18.24/EXTERNAL_HEADERS/mach-o/loader.h
        let mut header = vec![0u8; 32];

        // Magic number: MH_MAGIC_64 (0xFEEDFACF for 64-bit)
        header[0..4].copy_from_slice(&0xFEEDFACF_u32.to_le_bytes());

        // CPU type: x86_64 (0x01000007)
        header[4..8].copy_from_slice(&0x01000007_u32.to_le_bytes());

        // CPU subtype: x86_64_ALL (0x00000003)
        header[8..12].copy_from_slice(&0x00000003_u32.to_le_bytes());

        // File type: MH_DYLIB (0x00000006)
        header[12..16].copy_from_slice(&0x00000006_u32.to_le_bytes());

        header
    }

    /// Generate minimal PE header for Windows .lib files
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-19-windows-lib-libraries
    fn generate_pe_stub() -> Vec<u8> {
        // Minimal PE/COFF header (64 bytes)
        // Reference: https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
        let mut header = vec![0u8; 64];

        // DOS stub magic: "MZ"
        header[0..2].copy_from_slice(b"MZ");

        // PE signature offset (at 0x3C, points to 0x40)
        header[0x3C..0x40].copy_from_slice(&[0x40, 0x00, 0x00, 0x00]);

        // PE signature: "PE\0\0"
        // (Would be at offset 0x40 in a real PE file, simplified here)

        header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_filename_linux() {
        assert_eq!(
            MockLibrary::library_filename(LibraryType::BitNet, Platform::Linux),
            "libbitnet.so"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Llama, Platform::Linux),
            "libllama.so"
        );
        assert_eq!(MockLibrary::library_filename(LibraryType::Ggml, Platform::Linux), "libggml.so");
    }

    #[test]
    fn test_library_filename_macos() {
        assert_eq!(
            MockLibrary::library_filename(LibraryType::BitNet, Platform::MacOS),
            "libbitnet.dylib"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Llama, Platform::MacOS),
            "libllama.dylib"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Ggml, Platform::MacOS),
            "libggml.dylib"
        );
    }

    #[test]
    fn test_library_filename_windows() {
        assert_eq!(
            MockLibrary::library_filename(LibraryType::BitNet, Platform::Windows),
            "bitnet.lib"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Llama, Platform::Windows),
            "llama.lib"
        );
        assert_eq!(MockLibrary::library_filename(LibraryType::Ggml, Platform::Windows), "ggml.lib");
    }

    #[test]
    fn test_elf_header_magic() {
        let elf = MockLibrary::generate_elf_stub();
        assert_eq!(&elf[0..4], b"\x7fELF", "ELF magic bytes incorrect");
        assert_eq!(elf[4], 2, "ELF class should be 64-bit");
        assert_eq!(elf[5], 1, "ELF data should be little-endian");
    }

    #[test]
    fn test_macho_header_magic() {
        let macho = MockLibrary::generate_macho_stub();
        let magic = u32::from_le_bytes([macho[0], macho[1], macho[2], macho[3]]);
        assert_eq!(magic, 0xFEEDFACF, "Mach-O magic bytes incorrect");
    }

    #[test]
    fn test_pe_header_magic() {
        let pe = MockLibrary::generate_pe_stub();
        assert_eq!(&pe[0..2], b"MZ", "PE/DOS magic bytes incorrect");
    }
}
