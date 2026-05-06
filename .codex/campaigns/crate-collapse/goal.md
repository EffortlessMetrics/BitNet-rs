# Crate Collapse Goal

Reduce excess public crate seams by moving low-risk microcrates into SRP modules while preserving behavior, feature gates, and public API intent.

One PR should collapse one bounded crate or module cluster. Do not combine crate movement with runtime proof, backend identity, or hardware validation work.
