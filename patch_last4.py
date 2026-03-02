with open("crates/bitnet-gpu-hal/src/embedding_operations.rs", "r") as f:
    text = f.read()

text = text.replace("x[2 * i] * cos_val - x[2 * i + 1] * sin_val", "x[2 * i].mul_add(cos_val, -(x[2 * i + 1] * sin_val))")
text = text.replace("x[2 * i] * sin_val + x[2 * i + 1] * cos_val", "x[2 * i].mul_add(sin_val, x[2 * i + 1] * cos_val)")
text = text.replace("x0 * cos_val - x1 * sin_val", "x0.mul_add(cos_val, -(x1 * sin_val))")
text = text.replace("x0 * sin_val + x1 * cos_val", "x0.mul_add(sin_val, x1 * cos_val)")
text = text.replace("num_heads.leading_zeros() as u32", "num_heads.leading_zeros()")
text = text.replace("2.0_f32.powf(-(8.0 / closest_pow2 as f32))", "(-(8.0 / closest_pow2 as f32)).exp2()")
text = text.replace("2.0_f32.powf(-(8.0 / (2 * closest_pow2) as f32))", "(-(8.0 / (2 * closest_pow2) as f32)).exp2()")
text = text.replace("acc.iter_mut().for_each(|v| *v /= n);", "for v in acc.iter_mut() { *v /= n; }")
text = text.replace("acc.iter_mut().for_each(|v| *v /= total);", "for v in acc.iter_mut() { *v /= total; }")
text = text.replace("vec.iter_mut().for_each(|v| *v /= norm);", "for v in vec.iter_mut() { *v /= norm; }")
text = text.replace("vec.iter_mut().for_each(|v| *v = (*v - mean) / std);", "for v in vec.iter_mut() { *v = (*v - mean) / std; }")
text = text.replace("if let Some(ref pe) = self.positional {\n            if !pe.add_to(&mut output, seq_len, position_offset) {", "if let Some(ref pe) = self.positional\n            && !pe.add_to(&mut output, seq_len, position_offset)\n        {")

with open("crates/bitnet-gpu-hal/src/embedding_operations.rs", "w") as f:
    f.write(text)

with open("crates/bitnet-gpu-hal/src/mqa_gqa.rs", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "for h in 0..num_heads {" in line:
        lines[i] = "            for <item> in per_head_outputs.iter().take(num_heads) {\n" # I don't think it is right, let's just add #[allow(clippy::needless_range_loop)]

with open("crates/bitnet-gpu-hal/src/mqa_gqa.rs", "r") as f:
    text = f.read()
text = text.replace("for h in 0..num_heads {", "#[allow(clippy::needless_range_loop)]\n            for h in 0..num_heads {")
with open("crates/bitnet-gpu-hal/src/mqa_gqa.rs", "w") as f:
    f.write(text)

with open("crates/bitnet-gpu-hal/src/prompt_cache.rs", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "if let Some(id) = victim_id {" in line:
        if "if let Some(entry) = self.entries.remove(&id) {" in lines[i+1]:
            lines[i] = "        if let Some(id) = victim_id\n"
            lines[i+1] = "            && let Some(entry) = self.entries.remove(&id)\n"
            lines.insert(i+2, "        {\n")
            brace_count = 0
            for j in range(i+3, len(lines)):
                if "{" in lines[j]:
                    brace_count += lines[j].count("{")
                if "}" in lines[j]:
                    brace_count -= lines[j].count("}")
                if brace_count == -1:
                    lines.pop(j)
                    break
            break

with open("crates/bitnet-gpu-hal/src/prompt_cache.rs", "w") as f:
    f.writelines(lines)
