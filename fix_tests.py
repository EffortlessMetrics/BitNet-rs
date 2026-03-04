import re

with open('crates/bitnet-common/tests/proptest_wave30_common.rs', 'r') as f:
    content = f.read()

content = content.replace(
    'fn arb_shape(max_rank: usize, max_dim: usize) -> impl Strategy<Value = Vec<usize>> {\n    prop::collection::vec(1usize..=max_dim, 1..=max_rank)\n}\n',
    ''
)

with open('crates/bitnet-common/tests/proptest_wave30_common.rs', 'w') as f:
    f.write(content)
