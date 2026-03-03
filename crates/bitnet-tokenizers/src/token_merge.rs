//! Token merge and split utilities.
//!
//! Subword operations: merge adjacent tokens, split tokens at boundaries,
//! detokenize sequences, and manipulate token spans.

/// A token span with position information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub token_id: u32,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl TokenSpan {
    pub fn new(token_id: u32, text: impl Into<String>, start: usize, end: usize) -> Self {
        Self { token_id, text: text.into(), start, end }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Merge adjacent token spans into a single span with concatenated text.
pub fn merge_spans(spans: &[TokenSpan]) -> Option<TokenSpan> {
    if spans.is_empty() {
        return None;
    }
    let merged_text: String = spans.iter().map(|s| s.text.as_str()).collect();
    let start = spans.first().unwrap().start;
    let end = spans.last().unwrap().end;
    Some(TokenSpan { token_id: spans[0].token_id, text: merged_text, start, end })
}

/// Split a text into character-level spans starting from `offset`.
pub fn char_split(text: &str, offset: usize) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut pos = offset;
    for (i, ch) in text.chars().enumerate() {
        let ch_len = ch.len_utf8();
        spans.push(TokenSpan {
            token_id: i as u32,
            text: ch.to_string(),
            start: pos,
            end: pos + ch_len,
        });
        pos += ch_len;
    }
    spans
}

/// Split text at whitespace boundaries into word-level spans.
pub fn word_split(text: &str, offset: usize) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut token_id = 0u32;
    let mut chars = text.char_indices().peekable();

    while let Some(&(i, _)) = chars.peek() {
        // Skip whitespace
        let start_byte = i;
        let mut end_byte = i;
        let is_ws = text[i..].starts_with(char::is_whitespace);

        if is_ws {
            while let Some(&(j, c)) = chars.peek() {
                if c.is_whitespace() {
                    end_byte = j + c.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
        } else {
            while let Some(&(j, c)) = chars.peek() {
                if !c.is_whitespace() {
                    end_byte = j + c.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
        }

        let segment = &text[start_byte..end_byte];
        if !is_ws && !segment.is_empty() {
            spans.push(TokenSpan {
                token_id,
                text: segment.to_string(),
                start: offset + start_byte,
                end: offset + end_byte,
            });
            token_id += 1;
        }
    }
    spans
}

/// Reconstruct text from a sequence of token spans.
pub fn detokenize(spans: &[TokenSpan]) -> String {
    spans.iter().map(|s| s.text.as_str()).collect()
}

/// Find spans that overlap a given byte range.
pub fn overlapping_spans(
    spans: &[TokenSpan],
    range_start: usize,
    range_end: usize,
) -> Vec<&TokenSpan> {
    spans.iter().filter(|s| s.start < range_end && s.end > range_start).collect()
}

/// Group consecutive spans by a predicate.
pub fn group_spans<F>(spans: &[TokenSpan], same_group: F) -> Vec<Vec<&TokenSpan>>
where
    F: Fn(&TokenSpan, &TokenSpan) -> bool,
{
    if spans.is_empty() {
        return vec![];
    }
    let mut groups: Vec<Vec<&TokenSpan>> = vec![vec![&spans[0]]];
    for span in &spans[1..] {
        let last_group = groups.last().unwrap();
        let last_span = last_group.last().unwrap();
        if same_group(last_span, span) {
            groups.last_mut().unwrap().push(span);
        } else {
            groups.push(vec![span]);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_span_creation() {
        let span = TokenSpan::new(42, "hello", 0, 5);
        assert_eq!(span.token_id, 42);
        assert_eq!(span.text, "hello");
        assert_eq!(span.len(), 5);
        assert!(!span.is_empty());
    }

    #[test]
    fn test_empty_span() {
        let span = TokenSpan::new(0, "", 5, 5);
        assert!(span.is_empty());
    }

    #[test]
    fn test_merge_spans() {
        let spans = vec![TokenSpan::new(1, "hel", 0, 3), TokenSpan::new(2, "lo", 3, 5)];
        let merged = merge_spans(&spans).unwrap();
        assert_eq!(merged.text, "hello");
        assert_eq!(merged.start, 0);
        assert_eq!(merged.end, 5);
    }

    #[test]
    fn test_merge_empty() {
        assert!(merge_spans(&[]).is_none());
    }

    #[test]
    fn test_char_split() {
        let spans = char_split("abc", 10);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "a");
        assert_eq!(spans[0].start, 10);
        assert_eq!(spans[2].text, "c");
        assert_eq!(spans[2].end, 13);
    }

    #[test]
    fn test_word_split() {
        let spans = word_split("hello world foo", 0);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "hello");
        assert_eq!(spans[1].text, "world");
        assert_eq!(spans[2].text, "foo");
    }

    #[test]
    fn test_detokenize() {
        let spans = vec![
            TokenSpan::new(0, "Hello", 0, 5),
            TokenSpan::new(1, " ", 5, 6),
            TokenSpan::new(2, "world", 6, 11),
        ];
        assert_eq!(detokenize(&spans), "Hello world");
    }

    #[test]
    fn test_overlapping_spans() {
        let spans = vec![
            TokenSpan::new(0, "aa", 0, 2),
            TokenSpan::new(1, "bb", 2, 4),
            TokenSpan::new(2, "cc", 4, 6),
        ];
        let overlap = overlapping_spans(&spans, 1, 5);
        assert_eq!(overlap.len(), 3);
    }

    #[test]
    fn test_no_overlap() {
        let spans = vec![TokenSpan::new(0, "aa", 0, 2)];
        let overlap = overlapping_spans(&spans, 5, 10);
        assert!(overlap.is_empty());
    }

    #[test]
    fn test_group_spans() {
        let spans = vec![
            TokenSpan::new(0, "a", 0, 1),
            TokenSpan::new(1, "b", 1, 2),
            TokenSpan::new(2, " ", 2, 3),
            TokenSpan::new(3, "c", 3, 4),
        ];
        let groups =
            group_spans(&spans, |a, b| !a.text.trim().is_empty() && !b.text.trim().is_empty());
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn test_unicode_char_split() {
        let spans = char_split("héllo", 0);
        assert_eq!(spans.len(), 5);
        assert_eq!(spans[1].text, "é");
    }

    #[test]
    fn test_word_split_with_offset() {
        let spans = word_split("a b", 100);
        assert_eq!(spans[0].start, 100);
        assert_eq!(spans[1].start, 102);
    }
}
