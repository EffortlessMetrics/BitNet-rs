# Contributing to BitNet.rs

First off, thank you for considering contributing to `BitNet.rs`! We welcome contributions from everyone. This document provides guidelines to help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Getting Started: Your First Contribution](#getting-started-your-first-contribution)
- [Development Workflow](#development-workflow)
  - [1. Fork and Clone](#1-fork-and-clone)
  - [2. Build the Project](#2-build-the-project)
  - [3. Make Your Changes](#3-make-your-changes)
  - [4. Ensure Code Quality](#4-ensure-code-quality)
  - [5. Run Tests](#5-run-tests)
  - [6. Commit Your Changes](#6-commit-your-changes)
  - [7. Submit a Pull Request](#7-submit-a-pull-request)
- [Using `xtask` for Development](#using-xtask-for-development)
- [Security Vulnerability Reporting](#security-vulnerability-reporting)

## Code of Conduct

This project and everyone participating in it is governed by the [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on GitHub. When filing a bug report, please include:
- A clear and descriptive title.
- A detailed description of the problem, including the steps to reproduce it.
- The version of `BitNet.rs` you are using.
- Your operating system and hardware details (e.g., CPU, GPU).
- Any relevant logs or error messages.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue to discuss it. This allows us to coordinate our efforts and ensure the proposed change aligns with the project's goals.

### Submitting Pull Requests

We welcome pull requests for bug fixes, new features, and documentation improvements. Please follow the [Development Workflow](#development-workflow) outlined below.

## Getting Started: Your First Contribution

If you are looking for a place to start, check out the issues labeled `good first issue` or `help wanted` on GitHub. These are typically smaller, well-defined tasks that are a great way to get familiar with the codebase.

Don't hesitate to ask questions in the issue comments if you need clarification.

## Development Workflow

### 1. Fork and Clone

Fork the repository on GitHub and then clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/BitNet.git
cd BitNet
```

### 2. Build the Project

Follow the instructions in the [`docs/building.md`](./docs/building.md) guide to set up your development environment and build the project. A standard debug build is usually sufficient for development:

```bash
cargo build
```

### 3. Make Your Changes

Create a new branch for your changes:

```bash
git checkout -b your-feature-or-fix-name
```

Now, make your code or documentation changes.

### 4. Ensure Code Quality

Before committing your changes, please ensure your code adheres to the project's quality standards by running the following commands:

- **Format the code:**
  ```bash
  cargo fmt --all
  ```
- **Run the linter (`clippy`):**
  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```

### 5. Run Tests

Ensure that your changes do not break any existing functionality by running the test suite. New features should include new tests.

```bash
cargo test --workspace
```

### 6. Commit Your Changes

Use clear and descriptive commit messages. We recommend following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification, but it is not strictly required.

```bash
git commit -m "feat: Add a new feature" -m "Detailed description of the feature."
git push origin your-feature-or-fix-name
```

### 7. Submit a Pull Request

Open a pull request from your fork to the `main` branch of the original repository. In the pull request description, please:
- Link to any relevant issues.
- Provide a clear summary of the changes you have made.
- Explain the motivation for the change.

## Using `xtask` for Development

This repository uses the `xtask` pattern for common development and CI tasks. These are convenience scripts located in the `xtask/` directory.

You can invoke them using `cargo xtask <command>`. Here are a few useful commands:

- **Download a model for testing:**
  ```bash
  cargo xtask download-model
  ```
- **Fetch the C++ implementation for comparison:**
  ```bash
  cargo xtask fetch-cpp
  ```
- **Run cross-validation tests:**
  ```bash
  cargo xtask crossval
  ```

To see all available commands, run:
```bash
cargo xtask --help
```

## Security Vulnerability Reporting

If you believe you have found a security vulnerability, please refer to our security policy in [`SECURITY.md`](./SECURITY.md) for instructions on how to report it. **Please do not open a public issue for security vulnerabilities.**
