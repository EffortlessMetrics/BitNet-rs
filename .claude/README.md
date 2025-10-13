# BitNet.rs Claude Agent System

## Overview

This directory contains specialized Claude agents designed for BitNet.rs PR review workflow. The system provides comprehensive PR review automation with GitHub integration, local verification (since CI billing is disabled), and documentation maintenance.

## Agent Architecture

### Core PR Review Flow

```
pr-initial → pr-test → pr-context → pr-cleanup → pr-finalize → pr-merge → pr-doc-finalize
    ↓           ↓         ↓            ↓           ↓            ↓           ↓
  Setup      Testing   Review     Issue        Merge       Execution   Documentation
Analysis   Validation Analysis   Resolution  Preparation              Finalization
```

### Agent Responsibilities

| Agent | Primary Role | Key Tools | Output |
|-------|-------------|-----------|---------|
| **pr-initial** | PR analysis & setup | git, gh, analysis tools | Review plan & environment setup |
| **pr-test** | Comprehensive validation | just, cargo, xtask, nextest | Test results & quality gates |
| **pr-context** | Review & semantic analysis | GitHub API, code analysis | Context understanding & feedback processing |
| **pr-cleanup** | Issue resolution | cargo, git, editing tools | Fixed code & resolved issues |
| **pr-finalize** | Merge preparation | documentation, validation | Merge-ready state & documentation |
| **pr-merge** | Merge execution | git, GitHub CLI | Successful merge & cleanup |
| **pr-doc-finalize** | Documentation updates | editing, validation tools | Updated documentation following Diátaxis |

## BitNet.rs Integration

### Tooling Alignment
- **Just**: Primary workflow automation (`just cpu`, `just ci`, `just crossval`)
- **xtask**: Complex operations (`cargo run -p xtask -- full-crossval`)
- **MSRV 1.89.0**: Rust 2024 edition compliance validation
- **Feature Gates**: Comprehensive feature flag testing (`cpu`, `cuda`, `ffi`, `iq2s-ffi`)
- **Cross-validation**: C++ parity testing for correctness verification

### GitHub Integration
- **Status Updates**: Real-time progress reporting in PR comments
- **Issue Creation**: Automatic issue creation for identified improvements
- **Review Coordination**: Intelligent response to reviewer feedback
- **Branch Management**: Automated branch cleanup and merge coordination

## Usage Patterns

### Standard PR Review
```markdown
## Typical Flow

1. **PR Created** → Invoke `pr-initial`
   - Analyzes changes and scope
   - Sets up validation environment
   - Posts initial status with review plan
   - **Guides orchestrator**: "Next: invoke pr-test with [context]"

2. **Testing Phase** → `pr-test` (may loop)
   - Runs comprehensive validation suite
   - **If passing**: "Next: invoke pr-context"
   - **If failing**: "Next: invoke pr-cleanup with [issues]"
   - **If partial**: "Continue with pr-test [subset]"

3. **Context Analysis** → `pr-context`
   - Processes review comments and feedback
   - **If ready**: "Next: invoke pr-finalize"
   - **If issues**: "Next: invoke pr-cleanup with [feedback]"
   - **If waiting**: "Monitor GitHub, then re-invoke pr-context"

4. **Cleanup** → `pr-cleanup` (if needed)
   - Addresses specific issues systematically
   - **When done**: "Next: invoke pr-test for re-validation"

5. **Finalization** → `pr-finalize`
   - Final validation and merge preparation
   - **When ready**: "Next: invoke pr-merge"
   - **If blocked**: "Next: invoke pr-cleanup with [final-issues]"

6. **Merge** → `pr-merge`
   - Executes merge operation
   - **On success**: "Next: invoke pr-doc-finalize"
   - **On failure**: "Manual intervention required"

7. **Documentation** → `pr-doc-finalize`
   - Updates all affected documentation
   - **On completion**: "PR review flow complete"
```

### Agent Communication Pattern

Each agent ends with clear guidance for the orchestrator:

```markdown
## 🎯 Next Steps for Orchestrator

**Invoke**: `[next-agent]` agent
**Context**: [Summary of current state and what next agent needs to know]
**Environment**: [Any special configuration or state information]
**Priority**: [High/Medium/Low based on urgency]

**Expected Flow**: [What should happen next and possible outcomes]
```

## Configuration

### Environment Variables
```bash
# BitNet.rs specific
export BITNET_DETERMINISTIC=1        # Deterministic testing
export BITNET_SEED=42                # Reproducible results
export RAYON_NUM_THREADS=1           # Single-threaded for determinism
export BITNET_GGUF="path/to/model"   # Model path for cross-validation

# GitHub integration
export GH_TOKEN="your-token"         # GitHub CLI authentication
export GITHUB_REPOSITORY="owner/repo" # For automated operations
```

### Agent State Management
Agents coordinate through shared state files in `.claude/`:

```bash
.claude/
├── agents/                    # Agent definitions (this directory)
├── pr-state.json             # Current PR review state
├── pr-review.log            # Comprehensive review log
├── test-results/            # Test artifacts and reports
├── test-failures/           # Failure analysis and logs
├── context-analysis.md      # Review context and analysis
├── cleanup-log.md          # Issue resolution tracking
├── merge-log.md            # Merge history and documentation
├── doc-updates.log         # Documentation update tracking
└── merge-operations.log    # Merge operation audit trail
```

## Validation Matrix

### PR Type Detection
Agents automatically adjust validation based on changed files:

| Change Type | Features Tested | Validation Level | Cross-Validation |
|-------------|----------------|------------------|------------------|
| **Core** (kernels, inference) | `cpu`, `ffi` | Comprehensive | Required |
| **GPU** (CUDA kernels) | `cuda`, `cpu` | GPU-specific | If FFI changes |
| **API** (lib.rs, public APIs) | `cpu`, breaking-change | Full + API | If FFI changes |
| **FFI** (C bindings) | `cpu`, `ffi`, `crossval` | Full + C++ parity | Required |
| **Quantization** (I2_S, IQ2_S) | `cpu`, `iq2s-ffi` | Parity testing | Backend comparison |
| **Documentation** | None | Light | None |
| **Config/Build** | `cpu` | Build matrix | None |

### Quality Gates
All PRs must pass:
- ✅ MSRV 1.89.0 compilation
- ✅ Feature flag consistency
- ✅ Workspace test suite
- ✅ Clippy (pedantic level)
- ✅ Cargo fmt verification
- ✅ Security audit clean
- ✅ API compatibility (if applicable)
- ✅ Cross-validation parity (if FFI changes)
- ✅ Performance within bounds

## Customization

### Agent Behavior Modification
Each agent can be customized by editing its markdown file:

1. **Tools Available**: Modify tool permissions and capabilities
2. **Responsibilities**: Adjust scope and focus areas
3. **Commands**: Update BitNet.rs specific tooling commands
4. **Decision Matrix**: Modify flow logic and criteria
5. **Success Criteria**: Adjust quality gates and requirements

### Flow Customization
Modify the standard flow by:
- Adding new agents for specialized tasks
- Changing decision criteria in existing agents
- Adjusting coordination patterns
- Adding new validation steps

## Best Practices

### Agent Invocation
1. **Always provide context**: Include relevant information from previous agent
2. **Set clear expectations**: Specify what success/failure looks like
3. **Monitor progress**: Check GitHub status updates and logs
4. **Handle failures gracefully**: Each agent has error recovery procedures

### GitHub Integration
1. **Use status comments**: Keep PR updated with progress
2. **Address reviewer feedback**: Don't ignore human input
3. **Maintain audit trail**: Log all decisions and actions
4. **Coordinate timing**: Respect human reviewer availability

### Quality Assurance
1. **Trust but verify**: Agents provide validation, but spot-check results
2. **Maintain flexibility**: Override agent decisions when appropriate
3. **Learn from patterns**: Use logs to improve agent behavior
4. **Keep humans in the loop**: Complex decisions still need human judgment

## Troubleshooting

### Common Issues
- **Agent stuck in loop**: Check `.claude/pr-state.json` for state corruption
- **GitHub API issues**: Verify `gh auth status` and token permissions
- **Build failures**: Ensure environment matches CLAUDE.md requirements
- **Performance issues**: Check if deterministic testing is enabled properly

### Recovery Procedures
- **Reset agent state**: `rm .claude/pr-state.json`
- **Manual intervention**: Any agent can be bypassed for manual work
- **Rollback capability**: All agents support rollback to previous state
- **Emergency stops**: Agents detect critical failures and pause for human input

## Future Enhancements

### Planned Improvements
- **Performance trend analysis**: Track performance regression patterns
- **Automated benchmark comparison**: Enhanced performance validation
- **Security scan integration**: Deeper security analysis capabilities
- **Documentation quality metrics**: Quantitative documentation assessment

### Extension Points
- Add new validation types for emerging features
- Integrate with additional tooling (cargo-nextest, etc.)
- Enhanced GitHub integration (project boards, etc.)
- Custom notification systems for different types of changes

---

This agent system is designed to provide comprehensive, reliable PR review automation while maintaining the flexibility to handle complex cases and coordinate effectively with human reviewers.
