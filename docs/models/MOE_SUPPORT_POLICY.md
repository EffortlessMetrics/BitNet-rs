# MoE support policy

MoE support requires more than loading dense decoder weights. A receipt-backed MoE claim must prove router logits, top-k expert selection, expert weight loading, shared expert handling when present, active/total parameter accounting, and per-token expert coverage metrics.

A dense variant proof does not prove MoE variants in the same family. A source-backed active-parameter number is not a runtime routing claim.
