pub(super) fn parse_json_string(value: &str) -> Result<String, String> {
    if !value.starts_with('"') || !value.ends_with('"') || value.len() < 2 {
        return Err("expected quoted JSON string".to_string());
    }

    let mut chars = value[1..value.len() - 1].chars();
    let mut decoded = String::new();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => decode_escape(&mut chars, &mut decoded)?,
            '"' => return Err("unescaped quote in JSON string".to_string()),
            control if control.is_control() => {
                return Err("unescaped control character in JSON string".to_string());
            }
            other => decoded.push(other),
        }
    }

    Ok(decoded)
}

fn decode_escape(chars: &mut std::str::Chars<'_>, decoded: &mut String) -> Result<(), String> {
    let escaped =
        chars.next().ok_or_else(|| "unterminated escape sequence in JSON string".to_string())?;
    match escaped {
        '"' => decoded.push('"'),
        '\\' => decoded.push('\\'),
        '/' => decoded.push('/'),
        'b' => decoded.push('\u{0008}'),
        'f' => decoded.push('\u{000c}'),
        'n' => decoded.push('\n'),
        'r' => decoded.push('\r'),
        't' => decoded.push('\t'),
        'u' => decoded.push(decode_unicode_escape(chars)?),
        other => return Err(format!("unsupported JSON string escape: \\{other}")),
    }
    Ok(())
}

fn decode_unicode_escape(chars: &mut std::str::Chars<'_>) -> Result<char, String> {
    let code = read_hex_escape(chars)?;
    match code {
        0xD800..=0xDBFF => {
            if chars.next() != Some('\\') || chars.next() != Some('u') {
                return Err("high surrogate must be followed by a low surrogate".to_string());
            }
            let low = read_hex_escape(chars)?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err("high surrogate must be followed by a low surrogate".to_string());
            }
            let scalar = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            char::from_u32(scalar).ok_or_else(|| "invalid Unicode escape".to_string())
        }
        0xDC00..=0xDFFF => Err("low surrogate without preceding high surrogate".to_string()),
        scalar => char::from_u32(scalar).ok_or_else(|| "invalid Unicode escape".to_string()),
    }
}

fn read_hex_escape(chars: &mut std::str::Chars<'_>) -> Result<u32, String> {
    let mut code = 0_u32;
    for _ in 0..4 {
        let digit =
            chars.next().ok_or_else(|| "short Unicode escape in JSON string".to_string())?;
        code = (code << 4)
            | digit.to_digit(16).ok_or_else(|| "invalid Unicode escape digit".to_string())?;
    }
    Ok(code)
}
