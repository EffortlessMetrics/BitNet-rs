{
  description = "BitNet.rs – reproducible dev env & local CI (Nix flake)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Toolchains: stable (day-to-day) + MSRV
        rustStable = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "clippy" "rustfmt" ];
        };
        rustMsrv = pkgs.rust-bin.stable."1.89.0".default;

        # Native deps inferred from your build logs; adjust as needed
        nativeDeps = with pkgs; [
          pkg-config
          openssl
          cmake
          gnumake
          python3
          git
          jq
          libclang
          zlib
          oniguruma
        ];

        # Common env for rust builds: no sccache, no incremental
        commonEnv = {
          RUSTUP_TOOLCHAIN = "stable";
          RUSTC_WRAPPER = "";
          CARGO_NET_GIT_FETCH_WITH_CLI = "true";
          CARGO_NET_RETRY = "5";
          CARGO_INCREMENTAL = "0";
        };
      in
      {
        # ----------------------------------------------------------
        # Dev shells
        # ----------------------------------------------------------
        devShells = {
          default = pkgs.mkShell {
            name = "bitnet-dev";
            buildInputs = nativeDeps ++ [
              rustStable
              rustMsrv
            ];
            shellHook = ''
              export RUSTUP_TOOLCHAIN=${commonEnv.RUSTUP_TOOLCHAIN}
              export RUSTC_WRAPPER=${commonEnv.RUSTC_WRAPPER}
              export CARGO_NET_GIT_FETCH_WITH_CLI=${commonEnv.CARGO_NET_GIT_FETCH_WITH_CLI}
              export CARGO_INCREMENTAL=${commonEnv.CARGO_INCREMENTAL}
              export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

              echo "╭─────────────────────────────────────────────╮"
              echo "│ BitNet.rs development environment          │"
              echo "├─────────────────────────────────────────────┤"
              echo "│ rustc:  $(rustc --version | cut -d' ' -f2) │"
              echo "│ cargo:  $(cargo --version | cut -d' ' -f2) │"
              echo "│ MSRV:   1.89.0                              │"
              echo "├─────────────────────────────────────────────┤"
              echo "│ Quick start:                                │"
              echo "│  • nix build .#bitnet-server                │"
              echo "│  • nix run .#bitnet-cli -- --help           │"
              echo "│  • nix flake check                          │"
              echo "│  • ./scripts/ci-local.sh workspace          │"
              echo "╰─────────────────────────────────────────────╯"
            '';
          };

          msrv = pkgs.mkShell {
            name = "bitnet-dev-msrv";
            buildInputs = nativeDeps ++ [ rustMsrv ];
            shellHook = ''
              export RUSTUP_TOOLCHAIN=1.89.0
              export RUSTC_WRAPPER=${commonEnv.RUSTC_WRAPPER}
              export CARGO_NET_GIT_FETCH_WITH_CLI=${commonEnv.CARGO_NET_GIT_FETCH_WITH_CLI}
              export CARGO_INCREMENTAL=${commonEnv.CARGO_INCREMENTAL}
              export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

              echo "=== BitNet MSRV shell (1.89.0) ==="
              rustc --version
            '';
          };
        };

        # ----------------------------------------------------------
        # Buildable binaries (for local testing and future CI)
        # ----------------------------------------------------------
        packages = {
          # default package, e.g. bitnet-server
          default = self.packages.${system}.bitnet-server;

          bitnet-server = pkgs.rustPlatform.buildRustPackage {
            pname = "bitnet-server";
            version = "0.1.0";
            src = ./.;

            # Use Cargo workspace; restrict to server crate
            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [ "--package" "bitnet-server" "--no-default-features" "--features" "cpu" ];

            # Use our pinned toolchain
            nativeBuildInputs = [ rustStable ] ++ nativeDeps;

            # Ensure environment parity
            RUSTUP_TOOLCHAIN = commonEnv.RUSTUP_TOOLCHAIN;
            RUSTC = "${rustStable}/bin/rustc";
            CARGO_BUILD_TARGET = null;
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
          };

          bitnet-cli = pkgs.rustPlatform.buildRustPackage {
            pname = "bitnet-cli";
            version = "0.1.0";
            src = ./.;

            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [ "--package" "bitnet-cli" "--no-default-features" "--features" "cpu,full-cli" ];

            nativeBuildInputs = [ rustStable ] ++ nativeDeps;
            RUSTUP_TOOLCHAIN = commonEnv.RUSTUP_TOOLCHAIN;
            RUSTC = "${rustStable}/bin/rustc";
            CARGO_BUILD_TARGET = null;
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
          };

          bitnet-st2gguf = pkgs.rustPlatform.buildRustPackage {
            pname = "bitnet-st2gguf";
            version = "0.1.0";
            src = ./.;

            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [ "--package" "bitnet-st2gguf" ];

            nativeBuildInputs = [ rustStable ] ++ nativeDeps;
            RUSTUP_TOOLCHAIN = commonEnv.RUSTUP_TOOLCHAIN;
            RUSTC = "${rustStable}/bin/rustc";
            CARGO_BUILD_TARGET = null;
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
          };
        };

        # Convenient `nix run .#bitnet-cli` etc.
        apps = {
          bitnet-cli = flake-utils.lib.mkApp {
            drv = self.packages.${system}.bitnet-cli;
          };
          bitnet-server = flake-utils.lib.mkApp {
            drv = self.packages.${system}.bitnet-server;
          };
          bitnet-st2gguf = flake-utils.lib.mkApp {
            drv = self.packages.${system}.bitnet-st2gguf;
          };
        };

        # ----------------------------------------------------------
        # Flake checks -> wrap your scripts/ci-local.sh modes
        # ----------------------------------------------------------
        checks = {
          # Full workspace CPU CI (matches ./scripts/ci-local.sh workspace)
          workspace = pkgs.stdenv.mkDerivation {
            name = "ci-workspace";
            src = ./.;
            nativeBuildInputs = [ rustStable rustMsrv ] ++ nativeDeps;

            buildPhase = ''
              export HOME=$(mktemp -d)
              export RUSTUP_TOOLCHAIN=${commonEnv.RUSTUP_TOOLCHAIN}
              export RUSTC_WRAPPER=${commonEnv.RUSTC_WRAPPER}
              export CARGO_NET_GIT_FETCH_WITH_CLI=${commonEnv.CARGO_NET_GIT_FETCH_WITH_CLI}
              export CARGO_INCREMENTAL=${commonEnv.CARGO_INCREMENTAL}
              export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

              chmod +x scripts/ci-local.sh

              echo ">>> Running workspace CI checks (CPU-only)…"
              ./scripts/ci-local.sh workspace
            '';

            installPhase = ''
              mkdir -p $out
              echo "workspace CI checks passed" > $out/result
            '';
          };

          # Focused bitnet-server receipts validation (matches ./scripts/ci-local.sh bitnet-server-receipts)
          bitnet-server-receipts = pkgs.stdenv.mkDerivation {
            name = "ci-bitnet-server-receipts";
            src = ./.;
            nativeBuildInputs = [ rustStable rustMsrv ] ++ nativeDeps;

            buildPhase = ''
              export HOME=$(mktemp -d)
              export RUSTUP_TOOLCHAIN=${commonEnv.RUSTUP_TOOLCHAIN}
              export RUSTC_WRAPPER=${commonEnv.RUSTC_WRAPPER}
              export CARGO_NET_GIT_FETCH_WITH_CLI=${commonEnv.CARGO_NET_GIT_FETCH_WITH_CLI}
              export CARGO_INCREMENTAL=${commonEnv.CARGO_INCREMENTAL}
              export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

              chmod +x scripts/ci-local.sh

              echo ">>> Running bitnet-server receipts CI checks…"
              ./scripts/ci-local.sh bitnet-server-receipts
            '';

            installPhase = ''
              mkdir -p $out
              echo "bitnet-server receipts checks passed" > $out/result
            '';
          };
        };

        # Optional: makes `nix fmt` run nixpkgs-fmt
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
