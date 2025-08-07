# Homebrew Formula for BitNet.rs
# This formula will be submitted to homebrew-core once stable

class BitnetRs < Formula
  desc "High-performance Rust implementation of BitNet 1-bit LLM inference"
  homepage "https://github.com/microsoft/BitNet"
  url "https://github.com/microsoft/BitNet/archive/v1.0.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/microsoft/BitNet.git", branch: "main"

  depends_on "rust" => :build

  def install
    # Build CLI and server binaries
    system "cargo", "install", "--locked", "--root", prefix, "--path", "crates/bitnet-cli"
    system "cargo", "install", "--locked", "--root", prefix, "--path", "crates/bitnet-server"
    
    # Install shell completions
    generate_completions_from_executable(bin/"bitnet-cli", "completion")
    generate_completions_from_executable(bin/"bitnet-server", "completion")
    
    # Install man pages (if available)
    if (buildpath/"docs/man").exist?
      man1.install Dir["docs/man/*.1"]
    end
  end

  test do
    # Test CLI version
    assert_match version.to_s, shell_output("#{bin}/bitnet-cli --version")
    
    # Test server version
    assert_match version.to_s, shell_output("#{bin}/bitnet-server --version")
    
    # Test basic functionality
    system bin/"bitnet-cli", "test"
  end
end