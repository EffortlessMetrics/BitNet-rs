# Proposed Documentation Structure (Diataxis Framework)

This document outlines a proposed reorganization of the `docs/` directory to align with the [Diataxis framework](https://diataxis.fr/). The goal is to create a more intuitive, scalable, and user-focused documentation structure.

## Introduction to Diataxis

The Diataxis framework categorizes technical documentation into four distinct types, based on the user's immediate need:

1.  **Tutorials:** Practical, learning-oriented lessons for beginners.
2.  **How-to Guides:** Goal-oriented steps to solve a specific, real-world problem.
3.  **Explanation:** Conceptual, understanding-oriented discussions that clarify and illuminate a topic.
4.  **Reference:** Information-oriented, technical descriptions of the machinery.

Adopting this structure will make it easier for users to find the exact type of documentation they need, whether they are trying to learn, accomplish a task, understand a concept, or look up technical details.

## Proposed Directory Structure

We propose creating four new subdirectories within `docs/` to house the different types of documentation:

```
docs/
├── tutorials/
├── how-to/
├── explanation/
└── reference/
```

The existing `docs/testing/` subdirectory would be moved into `docs/how-to/testing/`.

## File Mapping

The following table maps the existing documentation files to their proposed new locations within the Diataxis structure.

| Current Location                       | Proposed New Location                    | Diataxis Category |
| -------------------------------------- | ---------------------------------------- | ----------------- |
| `getting-started.md`                   | `tutorials/getting-started.md`           | Tutorial          |
| `building.md`                          | `how-to/building-from-source.md`         | How-to Guide      |
| `deployment.md`                        | `how-to/deployment-options.md`           | How-to Guide      |
| `performance-guide.md`                 | `how-to/optimizing-performance.md`       | How-to Guide      |
| `performance-tuning.md`                | `how-to/performance-tuning.md`           | How-to Guide      |
| `cross-validation-setup.md`            | `how-to/setting-up-cross-validation.md`  | How-to Guide      |
| `testing/` (directory)                 | `how-to/testing/`                        | How-to Guide      |
| `architecture.md`                      | `explanation/architecture.md`            | Explanation       |
| `cpp-to-rust-migration.md`             | `explanation/cpp-to-rust-migration.md`   | Explanation       |
| `migration-guide.md`                   | `explanation/migration-overview.md`      | Explanation       |
| `migration-faq.md`                     | `explanation/migration-faq.md`           | Explanation       |
| `api-reference.md`                     | `reference/api-reference.md`             | Reference         |
| `api-compatibility.md`                 | `reference/api-compatibility.md`         | Reference         |
| `release-validation.md`                | `reference/release-validation.md`        | Reference         |
| `xtask.md`                             | `reference/xtask-commands.md`            | Reference         |
| `codegen.md`                           | `reference/codegen.md`                   | Reference         |
| `troubleshooting.md`                   | `how-to/troubleshooting.md`              | How-to Guide      |


## Benefits of This Structure

- **Improved Discoverability:** Users can more easily find the content they are looking for based on their current goal.
- **Clearer Purpose:** The structure makes the purpose of each document immediately clear.
- **Better Maintainability:** It will be easier for contributors to decide where to add new documentation.
- **Scalability:** The structure can easily accommodate new documentation as the project grows.

## Next Steps

This document serves as a proposal. If approved, the next steps would be:
1.  Create the new directories.
2.  Move and rename the files according to the mapping above.
3.  Update all cross-links within the documentation to reflect the new structure.
4.  Update the main `README.md` to link to the new documentation structure.
