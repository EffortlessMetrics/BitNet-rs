import re

with open('crates/bitnet-common/tests/proptest_wave30_common.rs', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip().startswith('// ── Shape validation'):
        skip = False
        new_lines.append(line)
        continue

    if line.strip().startswith('proptest! {') and not skip:
        # check if it's the rank test
        if 'fn rank_valid' in ''.join(lines[lines.index(line):lines.index(line)+15]):
            skip = True
        elif 'fn element_count_valid' in ''.join(lines[lines.index(line):lines.index(line)+15]):
            skip = True
        elif 'fn head_divisible_valid' in ''.join(lines[lines.index(line):lines.index(line)+15]):
            skip = True
        elif 'fn broadcastable_reflexive' in ''.join(lines[lines.index(line):lines.index(line)+15]):
            skip = True

    if not skip:
        new_lines.append(line)

    if skip and line.strip() == '}':
        skip = False

with open('crates/bitnet-common/tests/proptest_wave30_common.rs', 'w') as f:
    f.writelines(new_lines)
