with open("crates/bitnet-http-retry/src/lib.rs", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "let jitter = (attempt as u64 * 37) % 200;" in line:
        new_lines.append(line.replace("attempt as u64", "u64::from(attempt)"))
    elif "let raw = match value {" in line:
        new_lines.append("    let Some(raw) = value else { return 5 };\n")
        skip = True
    elif skip and "    };" in line:
        skip = False
    elif not skip:
        if "        .map(|d| d.as_secs().clamp(1, 3600))" in line:
            new_lines.append("        .map_or(5, |d| d.as_secs().clamp(1, 3600))\n")
        elif "        .unwrap_or(5)" in line:
            pass
        else:
            new_lines.append(line)

with open("crates/bitnet-http-retry/src/lib.rs", "w") as f:
    f.writelines(new_lines)
