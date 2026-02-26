# bitnet-rs Tech Debt Cleanup - Quick Start Guide

**Target**: Issues #343-#420 (78 issues)
**Outcome**: 17 tracking items (4 epics + 13 discrete issues)
**Reduction**: 78% fewer open issues

---

## 30-Second Summary

The TDD scaffolding phase left 78 issues (#343-#420). Analysis shows:
- **44 should be closed** (resolved by PRs or duplicates)
- **23 should be consolidated** into 4 tracking epics
- **13 remain as discrete issues** (actionable work)

**Impact**: Clean backlog, clear roadmap, better prioritization.

---

## Quick Execution (3 Steps)

### Step 1: Bulk Close (5 minutes)
```bash
cd /home/steven/code/Rust/BitNet-rs
./bulk_close_commands.sh  # Interactive, confirms each batch
```

**Closes**: 44 issues with PR references and verification commands.

### Step 2: Create Epics (15 minutes)
```bash
# Manually create 4 GitHub issues using epic_templates.md:
# 1. Epic 1: TL1/TL2 Production Quantization (6 issues)
# 2. Epic 2: Tokenizer Production Hardening (8 issues)
# 3. Epic 3: GPU Device Discovery & Memory Management (9 issues)
# 4. Epic 4: Server Production Observability (3 issues)

# Copy description from epic_templates.md for each
# Apply labels: epic, area/*, priority/*, milestone/*
```

### Step 3: Apply Labels (10 minutes)
```bash
# High priority (MVP-adjacent)
gh issue edit 417 --add-label "priority/high,area/performance,area/quantization,mvp:blocker,milestone/v0.1.x"
gh issue edit 413 --add-label "priority/high,area/testing,area/performance"
gh issue edit 414 --add-label "priority/high,area/testing,area/gpu"

# Medium priority (enhancements)
gh issue edit 344 418 --add-label "priority/medium,area/quantization,enhancement"
gh issue edit 376 384 --add-label "priority/medium,area/tokenization,enhancement"
gh issue edit 393 --add-label "priority/medium,area/quantization,bug"
gh issue edit 407 --add-label "priority/medium,area/models,bug"
gh issue edit 388 --add-label "priority/medium,area/inference,bug"

# Low priority (future work)
gh issue edit 350 353 370 371 385 --add-label "priority/low,area/server,milestone/v0.3.0"
gh issue edit 373 379 380 --add-label "priority/low,area/inference,enhancement,milestone/v0.2.0"
gh issue edit 387 405 --add-label "priority/low,area/validation,enhancement"
```

**Done!** Backlog cleaned, epics created, labels applied.

---

## What Gets Closed (44 issues)

### Resolved by PRs (25 issues)
- **PR #431** (Real Inference): #343, #345, #351, #352, #360, #378, #415
- **PR #448** (OTLP Migration): #359, #391
- **PR #430** (Tokenizer): #357, #377, #382, #383
- **PR #475** (Fixtures/Validation): #347, #358, #410
- **Feature Gates**: #408

### Duplicates/False Positives (9 issues)
#354, #356, #364, #374, #386, #390, #392, #394, #403

### Deferred/Stale (10 issues)
#368, #369, #372, #375, #389, #396, #411, #412, #420

---

## What Gets Consolidated (23 issues → 4 epics)

### Epic 1: TL1/TL2 Quantization (6 issues)
#346, #399, #401, #403, #416, #419

### Epic 2: Tokenizer Hardening (8 issues)
#381, #395, #397, #398, #400, #402, #404, #409

### Epic 3: GPU Discovery (9 issues)
#355, #361, #362, #363, #364, #365, #366, #367, #406

### Epic 4: Server Observability (3 issues)
#353, #370, #371 (currently kept as discrete, will consolidate after epic creation)

---

## What Stays Open (13 discrete issues)

### High Priority
- **#417**: QK256 dequantization (mvp:blocker)
- **#413**: Model loading timeouts
- **#414**: GPU cross-validation coverage

### Medium Priority
- **#344, #418**: Quantization enhancements
- **#376, #384**: Tokenizer enhancements
- **#393, #407, #388**: Bugs (GGUF, KV-cache)

### Low Priority
- **#350, #353, #370, #371, #385**: Server features
- **#373, #379, #380**: Inference optimizations
- **#387, #405**: Validation/monitoring

---

## Verification (After Execution)

```bash
# Check total open issues
gh issue list --state open | wc -l
# Expected: ~34 (78 - 44 closed, before epic consolidation)

# Verify epics created
gh issue list --label "epic" --state open
# Expected: 4 epics

# Check closed issues
gh issue list --search "is:issue is:closed number:343..420" --limit 100 | wc -l
# Expected: 44+ closed

# Check priority distribution
gh issue list --label "priority/high" --state open | wc -l   # Expected: 3
gh issue list --label "priority/medium" --state open | wc -l # Expected: 6
gh issue list --label "priority/low" --state open | wc -l    # Expected: 9
```

---

## Common Questions

### Q: Why close issues resolved by PRs?
**A**: Issues #254, #260, #439 already closed. Analysis confirms #343-#420 range has 25 more resolved by recent PRs (#431, #448, #430, #475). Closing with PR references maintains traceability.

### Q: Are we losing tracking history?
**A**: No. All closed issues have:
- PR reference in closing comment
- Verification command (e.g., `cat ci/inference.json | jq '.receipt.compute_path'`)
- Link to related issues/epics

### Q: Why epics instead of keeping individual issues?
**A**: Epics consolidate related work under milestone-aligned tracking. Example: 6 TL1/TL2 issues become 1 epic with 6 subtasks. Easier to plan sprints.

### Q: Can we reopen closed issues?
**A**: Yes. If analysis was incorrect, reopen with comment explaining rationale. Bulk close script includes PR verification for confidence.

### Q: What if I disagree with an epic consolidation?
**A**: Keep the issue discrete instead. Example: If #417 (mvp:blocker) was in an epic, we'd keep it discrete due to urgency. Epics are for related, non-urgent work.

---

## File Locations

- **Analysis**: `tech_debt_analysis_343_420.md` (detailed breakdown)
- **Bulk Script**: `bulk_close_commands.sh` (executable, interactive)
- **Epic Templates**: `epic_templates.md` (copy descriptions for GitHub)
- **Summary**: `TECH_DEBT_CLEANUP_SUMMARY.md` (executive summary)
- **Quick Start**: `QUICK_START_CLEANUP.md` (this file)

---

## Troubleshooting

### Bulk close script fails
```bash
# Check gh CLI authentication
gh auth status

# Check issue exists before closing
gh issue view 343

# Run individual batch manually if needed
gh issue close 343 -c "Closed as resolved by PR #431..."
```

### Epic creation issues
- Use GitHub web UI if `gh issue create --body-file` fails
- Copy template description manually from `epic_templates.md`
- Apply labels after creation if `--label` flag errors

### Label application errors
```bash
# Check available labels
gh label list

# Create missing labels if needed
gh label create "priority/high" --color "d73a4a"
gh label create "area/quantization" --color "0e8a16"
```

---

**Ready to execute?** Start with Step 1 (bulk close script).

**Questions?** See full analysis in `tech_debt_analysis_343_420.md`.

**Status**: ✅ Ready for cleanup
