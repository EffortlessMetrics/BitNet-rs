# Contributing to BitNet.rs

First off, thank you for considering contributing to BitNet.rs! It's people like you that make BitNet.rs such a great tool. We welcome contributions of all kinds, from documentation improvements to new features and bug fixes.

## Ways to Contribute

*   **Reporting Bugs**: If you find a bug, please open an issue on our GitHub repository. Be sure to include a clear title, a description of the issue, and steps to reproduce it.
*   **Suggesting Enhancements**: If you have an idea for a new feature or an improvement to an existing one, please open an issue to discuss it.
*   **Pull Requests**: If you're ready to contribute code or documentation, we'd love to see your pull requests.

## Development Setup

To get started with development, you'll need to have Rust 1.70.0 or later installed. You can then clone the repository and build the project:

```bash
git clone https://github.com/microsoft/BitNet.rs.git
cd BitNet.rs
cargo build --release
```

### Running Tests

To ensure the quality of the code, please run the full test suite before submitting a pull request:

```bash
# Run the standard Rust tests
cargo test --workspace

# Run the cross-validation tests against the legacy C++ implementation
# This requires C++ build tools to be installed.
cargo test --workspace --features crossval
```

### Code Quality

We use `rustfmt` for code formatting and `clippy` for linting. Please ensure your code adheres to our quality standards:

```bash
# Format the code
cargo fmt --all

# Run clippy for lints
cargo clippy --all-targets --all-features -- -D warnings
```

## Spec-Driven Design

BitNet.rs follows a **spec-driven design** process for significant new features or architectural changes. This ensures that all changes are well-planned, discussed, and aligned with the project's goals before implementation begins.

If you plan to make a substantial change, please follow these steps:

1.  **Open an Issue**: Start by opening an issue on GitHub to discuss your proposal at a high level. This helps to get early feedback and ensure your idea aligns with the project's direction.

2.  **Write a Specification**: Once there is general agreement on the idea, the next step is to write a specification document. Specs are located in the `.kiro/specs/` directory. Create a new subdirectory for your feature (e.g., `.kiro/specs/my-new-feature/`).

3.  **Create `requirements.md` and `design.md`**: Inside your new spec directory, create two files:
    *   `requirements.md`: This document should outline the "what" and "why" of your feature. It should include user stories and acceptance criteria.
    *   `design.md`: This document should detail the "how". It should include architectural diagrams, API designs, data models, and a testing strategy.

4.  **Submit a Pull Request for the Spec**: Open a pull request with your new spec files. This will be the main forum for discussion and refinement of your proposal. The project maintainers will work with you to finalize the spec.

5.  **Implement the Feature**: Once the spec is approved and merged, you can begin implementing the feature in a separate pull request. This PR should reference the spec and the original issue.

This process helps us build a high-quality, maintainable, and well-documented codebase. For smaller changes and bug fixes, a full spec is not required.

## Pull Request Process

1.  Ensure all tests and quality checks are passing.
2.  Update the documentation if you have changed any user-facing APIs or added new features.
3.  Update the `CHANGELOG.md` with a summary of your changes under the `[Unreleased]` section.
4.  Open a pull request with a clear title and description of your changes.

Thank you for your contributions!
