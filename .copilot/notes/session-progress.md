# Apple Silicon Fleet Session Progress

## Merge Velocity
- Total PRs merged this session: ~430+ (and counting)
- Open PRs: ~188 (down from ~300+)
- CI runs cancelled (noise reduction): ~5,000+
- Agents dispatched: 49

## Apple Silicon PRs Created & Merged
### MERGED
- PR #1722: ARM NEON build error fixes (vgetq_lane_f32 + is_aarch64_feature_detected)
- PR #1768: 25 ARM clippy errors fixed
- PR #1823: Gate Avx2Kernel behind x86_64
- PR #1923: NEON softmax
- PR #2015: NEON activations (SiLU, GELU, ReLU, Swish)
- PR #2087: NEON pooling (max, avg, global)
- PR #2101: NEON micro-benchmarks (Criterion)
- PR #2115: NEON scatter-gather
- PR #2129: NEON convolution
- PR #2168: Apple Silicon E2E inference tests
- PR #2180: NEON padding & clipping

### IN CI PIPELINE
- PR #2177: NEON fused attention v2
- PR #2191: NEON fused MLP
- PR #2192: Metal memory management tests
- PR #2207: wgpu compute pipeline tests  
- PR #2208: NEON token sampling
- PR #2213: NEON layer fusion
- PR #2214: Apple Silicon property tests (proptest)

### Metal Tests Merged
- PR #2072: Metal device integration tests
- PR #2152: Metal compute validation tests
- Earlier: Metal shader validation, Metal performance tests

## Key Learnings
1. Guards workflow requires SHA-pinned action refs — floating refs like @v4 fail
2. CI Core uses `cancel-in-progress: true` — cascade after merge kills ~200+ runs
3. Cherry-pick rebase loses NEW files when mod.rs conflicts use --theirs (takes main's version)
4. Check-run names are case-sensitive: "Clippy" not "clippy", "Documentation" not "documentation"
5. Rate limit of 5000/hr gets consumed fast when scanning 188+ PRs with check-runs
6. GPU-hal PRs have ~100 pub mod declarations — biggest conflict surface in repo
7. Cap agents at 3-4 simultaneous to avoid token rate limits

## Process Improvements Made
- Fixed check name matching (was lowercase, needed Title Case)
- Added SHA cross-referencing for Guards status
- Batch cancel non-essential CI after each merge wave
- Dedup pub mod lines automatically during rebase
- Verify non-empty diff after cherry-pick (close empty PRs)
