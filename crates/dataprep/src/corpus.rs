use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::PathBuf;

use crate::Error;
use crate::docs::{Documents, ReadStats};

pub struct Corpus {
    files: Vec<PathBuf>,
    next_file: usize,
    current: Option<Documents<BufReader<File>>>,
    finished: ReadStats,
}

impl Corpus {
    pub fn new(files: Vec<PathBuf>) -> Result<Self, Error> {
        if files.is_empty() {
            return Err("no input files given".into());
        }
        for path in &files {
            let metadata = fs::metadata(path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if !metadata.is_file() {
                return Err(format!("{} is not a file", path.display()).into());
            }
        }
        Ok(Self {
            files,
            next_file: 0,
            current: None,
            finished: ReadStats::default(),
        })
    }

    pub fn stats(&self) -> ReadStats {
        match &self.current {
            Some(current) => sum(self.finished, current.stats()),
            None => self.finished,
        }
    }
}

fn sum(a: ReadStats, b: ReadStats) -> ReadStats {
    ReadStats {
        documents: a.documents + b.documents,
        invalid_utf8_lines: a.invalid_utf8_lines + b.invalid_utf8_lines,
        rejected_documents: a.rejected_documents + b.rejected_documents,
    }
}

impl Iterator for Corpus {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current) = self.current.as_mut() {
                if let Some(item) = current.next() {
                    return Some(item);
                }
                self.finished = sum(self.finished, current.stats());
                self.current = None;
            }
            let path = self.files.get(self.next_file)?;
            self.next_file += 1;
            match File::open(path) {
                Ok(file) => self.current = Some(Documents::new(BufReader::new(file))),
                Err(error) => {
                    let message = format!("{}: {error}", path.display());
                    return Some(Err(io::Error::new(error.kind(), message)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::Corpus;

    #[test]
    fn chains_files_and_adds_up_stats() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.txt");
        let second = dir.path().join("b.txt");
        fs::write(&first, "one\n<|endoftext|>\ntwo\n").unwrap();
        fs::write(&second, "three\n<|endoftext|>\nbad <|im_end|>\n").unwrap();

        let mut corpus = Corpus::new(vec![first, second]).unwrap();
        let documents: Vec<String> = corpus.by_ref().map(Result::unwrap).collect();
        assert_eq!(documents, vec!["one\n", "two\n", "three\n"]);
        let stats = corpus.stats();
        assert_eq!(stats.documents, 3);
        assert_eq!(stats.rejected_documents, 1);
    }

    #[test]
    fn a_document_never_spans_two_files() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.txt");
        let second = dir.path().join("b.txt");
        fs::write(&first, "end of first").unwrap();
        fs::write(&second, "start of second").unwrap();
        let corpus = Corpus::new(vec![first, second]).unwrap();
        let documents: Vec<String> = corpus.map(Result::unwrap).collect();
        assert_eq!(documents, vec!["end of first\n", "start of second\n"]);
    }

    #[test]
    fn rejects_an_empty_file_list() {
        assert!(Corpus::new(Vec::new()).is_err());
    }

    #[test]
    fn rejects_a_missing_file_before_reading_anything() {
        let dir = tempfile::tempdir().unwrap();
        let present = dir.path().join("a.txt");
        fs::write(&present, "text\n").unwrap();
        let missing = dir.path().join("missing.txt");
        assert!(Corpus::new(vec![present, missing]).is_err());
    }

    #[test]
    fn rejects_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Corpus::new(vec![dir.path().to_path_buf()]).is_err());
    }

    #[test]
    fn a_file_with_only_separators_gives_no_documents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "<|endoftext|>\n<|endoftext|>\n").unwrap();
        let mut corpus = Corpus::new(vec![path]).unwrap();
        assert!(corpus.next().is_none());
        assert_eq!(corpus.stats().documents, 0);
    }
}
