# BITNET-SPEC-LLAMA3-8B-158-TL1-TL2

Status: proposed
Owner: cpu-proof
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no TL support until TL scalar oracles exist
Policy impact: no policy exception

## Purpose

Define `TL1` and `TL2` separately from `I2_S/QK256`.

## Required TL contract

The TL specs and fixtures must define TL1 tensor layout, TL2 tensor layout,
lookup-table semantics, bit packing, weight scale/group scale semantics,
activation type, row stride, block size, tail behavior, endianness, GGUF
metadata, and differences from `I2_S/QK256`.

## Route separation

- ARM `TL1` is a candidate because upstream lists it for this model.
- x86 `TL2` is a candidate because upstream lists it for this model.
- x86 `TL1` and ARM `TL2` are unsupported upstream and diagnostic-only.

## Hard rule

`TL1` and `TL2` are not QK256. No TL backend work may begin before scalar TL
oracles and fixtures exist.
