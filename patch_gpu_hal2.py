with open("crates/bitnet-gpu-hal/src/embedding_operations.rs", "r") as f:
    content = f.read()

content = content.replace("out[2 * i] = x[2 * i] * cos_val - x[2 * i + 1] * sin_val;", "out[2 * i] = x[2 * i].mul_add(cos_val, -(x[2 * i + 1] * sin_val));")
content = content.replace("out[2 * i + 1] = x[2 * i] * sin_val + x[2 * i + 1] * cos_val;", "out[2 * i + 1] = x[2 * i].mul_add(sin_val, x[2 * i + 1] * cos_val);")
content = content.replace("data[row + 2 * i] = x0 * cos_val - x1 * sin_val;", "data[row + 2 * i] = x0.mul_add(cos_val, -(x1 * sin_val));")
content = content.replace("data[row + 2 * i + 1] = x0 * sin_val + x1 * cos_val;", "data[row + 2 * i + 1] = x0.mul_add(sin_val, x1 * cos_val);")
content = content.replace("1_usize << (usize::BITS - 1 - num_heads.leading_zeros() as u32);", "1_usize << (usize::BITS - 1 - num_heads.leading_zeros());")
content = content.replace("2.0_f32.powf(-(8.0 / closest_pow2 as f32));", "(-(8.0 / closest_pow2 as f32)).exp2();")
content = content.replace("2.0_f32.powf(-(8.0 / (2 * closest_pow2) as f32));", "(-(8.0 / (2 * closest_pow2) as f32)).exp2();")

content = content.replace("acc.iter_mut().for_each(|v| *v /= n);", "for v in acc.iter_mut() { *v /= n; }")
content = content.replace("acc.iter_mut().for_each(|v| *v /= total);", "for v in acc.iter_mut() { *v /= total; }")
content = content.replace("vec.iter_mut().for_each(|v| *v /= norm);", "for v in vec.iter_mut() { *v /= norm; }")
content = content.replace("vec.iter_mut().for_each(|v| *v = (*v - mean) / std);", "for v in vec.iter_mut() { *v = (*v - mean) / std; }")

content = content.replace("""if let Some(ref pe) = self.positional {
            if !pe.add_to(&mut output, seq_len, position_offset) {
                return None;
            }
        }""", """if let Some(ref pe) = self.positional
            && !pe.add_to(&mut output, seq_len, position_offset) {
                return None;
            }""")


with open("crates/bitnet-gpu-hal/src/embedding_operations.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/token_streaming.rs", "r") as f:
    content = f.read()

content = content.replace("""if let Some(ts) = self.first_insert {
            if ts.elapsed() >= self.max_delay {
                return true;
            }
        }""", """if let Some(ts) = self.first_insert
            && ts.elapsed() >= self.max_delay {
                return true;
            }""")

with open("crates/bitnet-gpu-hal/src/token_streaming.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/model_architecture.rs", "r") as f:
    content = f.read()

content = content.replace("""if let Some(v) = metadata.get(*key) {
                if let Ok(n) = v.parse::<usize>() {
                    return Some(n);
                }
            }""", """if let Some(v) = metadata.get(*key)
                && let Ok(n) = v.parse::<usize>() {
                    return Some(n);
                }""")

with open("crates/bitnet-gpu-hal/src/model_architecture.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/prompt_cache.rs", "r") as f:
    content = f.read()

content = content.replace("""if let Some(id) = victim_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.trie.remove(&entry.kv_state.prefix_tokens);
                let freed = entry.kv_state.memory_bytes();
                self.stats.bytes_used -= freed;
                self.stats.entry_count = self.entries.len();
            }
        }""", """if let Some(id) = victim_id
            && let Some(entry) = self.entries.remove(&id) {
                self.trie.remove(&entry.kv_state.prefix_tokens);
                let freed = entry.kv_state.memory_bytes();
                self.stats.bytes_used -= freed;
                self.stats.entry_count = self.entries.len();
            }""")

with open("crates/bitnet-gpu-hal/src/prompt_cache.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/semantic_search.rs", "r") as f:
    content = f.read()

content = content.replace("6364136223846793005", "6_364_136_223_846_793_005")
content = content.replace("0xcbf29ce484222325", "0xcbf2_9ce4_8422_2325")
content = content.replace("0x100000001b3", "0x0100_0000_01b3")

with open("crates/bitnet-gpu-hal/src/semantic_search.rs", "w") as f:
    f.write(content)
