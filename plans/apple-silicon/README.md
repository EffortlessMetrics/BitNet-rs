# Apple Silicon Plans

This directory sequences Apple Silicon docs and rails work. The first rollout is
source-of-truth cleanup: it adds a map and then contracts that prevent dense
SLM, BitNet, Metal, MPSGraph, Neural Engine, MacBook, and M4 Mac mini evidence
from being conflated.

Start with [`implementation-plan.md`](implementation-plan.md). The plan is
intentionally documentation-first and does not authorize runtime backend changes,
model binary commits, full Metal inference claims, Neural Engine claims, or live
hardware/model timing in ordinary generic PR CI.
