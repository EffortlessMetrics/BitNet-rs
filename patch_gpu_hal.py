with open("crates/bitnet-gpu-hal/src/token_streaming.rs", "r") as f:
    content = f.read()

content = content.replace(
    "if (prev + 1) as usize >= self.threshold {\n            if !self.paused.swap(true, Ordering::SeqCst) {\n                self.pause_count.fetch_add(1, Ordering::SeqCst);\n            }\n        }",
    "if (prev + 1) as usize >= self.threshold && !self.paused.swap(true, Ordering::SeqCst) {\n            self.pause_count.fetch_add(1, Ordering::SeqCst);\n        }"
)
content = content.replace(
    "if let Some(ts) = self.first_insert {\n            if ts.elapsed() >= self.max_delay {\n                return true;\n            }\n        }",
    "if let Some(ts) = self.first_insert {\n            if ts.elapsed() >= self.max_delay {\n                return true;\n            }\n        }" # keep as is
)

with open("crates/bitnet-gpu-hal/src/token_streaming.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/model_architecture.rs", "r") as f:
    content = f.read()

content = content.replace(
    "let layer = match self.architecture.layers.first() {\n            Some(l) => l,\n            None => return 0,\n        };",
    "let Some(layer) = self.architecture.layers.first() else { return 0 };"
)
content = content.replace(
    "if let Some(v) = metadata.get(*key) {\n                if let Ok(n) = v.parse::<usize>() {\n                    return Some(n);\n                }\n            }",
    "if let Some(v) = metadata.get(*key) {\n                if let Ok(n) = v.parse::<usize>() {\n                    return Some(n);\n                }\n            }"
)
content = content.replace(
    "fn make_spec(",
    "#[allow(clippy::too_many_arguments)]\nfn make_spec("
)

with open("crates/bitnet-gpu-hal/src/model_architecture.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/semantic_search.rs", "r") as f:
    content = f.read()

content = content.replace(
    "let entry = match self.entry_point {\n            Some(e) => e,\n            None => return Ok(Vec::new()),\n        };",
    "let Some(entry) = self.entry_point else { return Ok(Vec::new()) };"
)

with open("crates/bitnet-gpu-hal/src/semantic_search.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-gpu-hal/src/prompt_cache.rs", "r") as f:
    content = f.read()

content = content.replace(
    "if let Some(id) = victim_id {\n            if let Some(entry) = self.entries.remove(&id) {\n                self.trie.remove(&entry.kv_state.prefix_tokens);\n                let freed = entry.kv_state.memory_bytes();\n                self.stats.bytes_used -= freed;\n                self.stats.entry_count = self.entries.len();\n            }\n        }",
    "if let Some(id) = victim_id {\n            if let Some(entry) = self.entries.remove(&id) {\n                self.trie.remove(&entry.kv_state.prefix_tokens);\n                let freed = entry.kv_state.memory_bytes();\n                self.stats.bytes_used -= freed;\n                self.stats.entry_count = self.entries.len();\n            }\n        }"
)

with open("crates/bitnet-gpu-hal/src/prompt_cache.rs", "w") as f:
    f.write(content)
