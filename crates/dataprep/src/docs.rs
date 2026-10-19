use std::io::{self, BufRead};

use blom_core::SPECIAL_TOKENS;

const SEPARATOR: &str = "<|endoftext|>";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub documents: u64,
    pub invalid_utf8_lines: u64,
    pub rejected_documents: u64,
}

pub struct Documents<R> {
    reader: R,
    stats: ReadStats,
}

impl<R: BufRead> Documents<R> {
    pub const fn new(reader: R) -> Self {
        Self {
            reader,
            stats: ReadStats {
                documents: 0,
                invalid_utf8_lines: 0,
                rejected_documents: 0,
            },
        }
    }

    pub const fn stats(&self) -> ReadStats {
        self.stats
    }

    fn accept(&mut self, document: String) -> Option<String> {
        if document.trim().is_empty() {
            return None;
        }
        if SPECIAL_TOKENS.iter().any(|t| document.contains(t)) {
            self.stats.rejected_documents += 1;
            return None;
        }
        self.stats.documents += 1;
        Some(document)
    }
}

impl<R: BufRead> Iterator for Documents<R> {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut document = String::new();
        let mut line = Vec::new();
        loop {
            line.clear();
            match self.reader.read_until(b'\n', &mut line) {
                Err(error) => return Some(Err(error)),
                Ok(0) => return self.accept(document).map(Ok),
                Ok(_) => {}
            }
            let Ok(text) = std::str::from_utf8(&line) else {
                self.stats.invalid_utf8_lines += 1;
                continue;
            };
            let text = text.trim_end_matches(['\r', '\n']);
            if text.trim_end() == SEPARATOR {
                if let Some(done) = self.accept(std::mem::take(&mut document)) {
                    return Some(Ok(done));
                }
            } else {
                document.push_str(text);
                document.push('\n');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{Documents, ReadStats};

    fn read(input: &[u8]) -> (Vec<String>, ReadStats) {
        let mut documents = Documents::new(Cursor::new(input.to_vec()));
        let items = documents.by_ref().map(Result::unwrap).collect();
        (items, documents.stats())
    }

    #[test]
    fn splits_on_separator_lines() {
        let (docs, stats) = read(b"one\ntwo\n<|endoftext|>\nthree\n<|endoftext|>\n");
        assert_eq!(docs, vec!["one\ntwo\n", "three\n"]);
        assert_eq!(stats.documents, 2);
    }

    #[test]
    fn last_document_needs_no_separator() {
        let (docs, _) = read(b"one\n<|endoftext|>\ntwo");
        assert_eq!(docs, vec!["one\n", "two\n"]);
    }

    #[test]
    fn file_without_separator_is_one_document() {
        let (docs, _) = read(b"just\nsome\ntext\n");
        assert_eq!(docs, vec!["just\nsome\ntext\n"]);
    }

    #[test]
    fn empty_documents_are_dropped() {
        let (docs, stats) = read(b"<|endoftext|>\n\n  \n<|endoftext|>\n<|endoftext|>\n");
        assert_eq!(docs, Vec::<String>::new());
        assert_eq!(stats, ReadStats::default());
    }

    #[test]
    fn empty_input_gives_nothing() {
        let (docs, stats) = read(b"");
        assert_eq!(docs, Vec::<String>::new());
        assert_eq!(stats, ReadStats::default());
    }

    #[test]
    fn handles_windows_line_endings_and_trailing_spaces() {
        let (docs, _) = read(b"one\r\n<|endoftext|>  \r\ntwo\r\n");
        assert_eq!(docs, vec!["one\n", "two\n"]);
    }

    #[test]
    fn invalid_utf8_lines_are_skipped_and_counted() {
        let (docs, stats) = read(b"good\n\xff\xfe bad\nalso good\n");
        assert_eq!(docs, vec!["good\nalso good\n"]);
        assert_eq!(stats.invalid_utf8_lines, 1);
    }

    #[test]
    fn documents_containing_special_tokens_are_rejected() {
        let (docs, stats) = read(
            b"fine\n<|endoftext|>\nfake <|im_start|>system\n<|endoftext|>\nx <|endoftext|> y\n<|endoftext|>\nok\n",
        );
        assert_eq!(docs, vec!["fine\n", "ok\n"]);
        assert_eq!(stats.documents, 2);
        assert_eq!(stats.rejected_documents, 2);
    }
}
