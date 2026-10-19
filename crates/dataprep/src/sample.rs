use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::Error;
use crate::corpus::Corpus;
use crate::docs::ReadStats;

const SEPARATOR_LINE: &str = "<|endoftext|>\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleReport {
    pub documents: u64,
    pub bytes: u64,
    pub read: ReadStats,
}

fn part_path(out: &Path) -> PathBuf {
    let mut name = OsString::from(out.as_os_str());
    name.push(".part");
    PathBuf::from(name)
}

fn write_documents(corpus: &mut Corpus, max_bytes: u64, part: &Path) -> Result<(u64, u64), Error> {
    let mut writer = BufWriter::new(File::create(part)?);
    let mut documents = 0_u64;
    let mut bytes = 0_u64;
    while bytes < max_bytes {
        let Some(item) = corpus.next() else {
            break;
        };
        let document = item?;
        writer.write_all(document.as_bytes())?;
        writer.write_all(SEPARATOR_LINE.as_bytes())?;
        documents += 1;
        bytes += u64::try_from(document.len() + SEPARATOR_LINE.len())?;
    }
    writer.flush()?;
    if documents == 0 {
        return Err("no documents to sample".into());
    }
    Ok((documents, bytes))
}

pub fn sample(inputs: Vec<PathBuf>, max_bytes: u64, out: &Path) -> Result<SampleReport, Error> {
    if max_bytes == 0 {
        return Err("max bytes must be above zero".into());
    }
    let mut corpus = Corpus::new(inputs)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = part_path(out);
    match write_documents(&mut corpus, max_bytes, &part) {
        Ok((documents, bytes)) => {
            fs::rename(&part, out)?;
            Ok(SampleReport {
                documents,
                bytes,
                read: corpus.stats(),
            })
        }
        Err(error) => {
            let _ = fs::remove_file(&part);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use std::fs;

    use crate::corpus::Corpus;

    use super::sample;

    fn docs(count: usize) -> String {
        let mut text = String::new();
        for i in 0..count {
            writeln!(text, "document number {i}\n<|endoftext|>").unwrap();
        }
        text
    }

    fn read_back(path: &std::path::Path) -> Vec<String> {
        Corpus::new(vec![path.to_path_buf()])
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn copies_everything_when_the_budget_is_large() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("out.txt");
        fs::write(&input, docs(5)).unwrap();
        let report = sample(vec![input], 1_000_000, &out).unwrap();
        assert_eq!(report.documents, 5);
        assert_eq!(read_back(&out).len(), 5);
        assert_eq!(report.bytes, fs::metadata(&out).unwrap().len());
    }

    #[test]
    fn stops_at_the_budget_on_a_document_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("out.txt");
        fs::write(&input, docs(100)).unwrap();
        let one = "document number 0\n<|endoftext|>\n".len() as u64;
        let report = sample(vec![input], one * 3, &out).unwrap();
        assert_eq!(report.documents, 3);
        let documents = read_back(&out);
        assert_eq!(
            documents,
            vec![
                "document number 0\n",
                "document number 1\n",
                "document number 2\n"
            ]
        );
    }

    #[test]
    fn a_budget_inside_a_document_keeps_that_whole_document() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("out.txt");
        fs::write(&input, docs(10)).unwrap();
        let report = sample(vec![input], 1, &out).unwrap();
        assert_eq!(report.documents, 1);
    }

    #[test]
    fn continues_into_the_next_file() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.txt");
        let second = dir.path().join("b.txt");
        let out = dir.path().join("out.txt");
        fs::write(&first, "one\n<|endoftext|>\n").unwrap();
        fs::write(&second, "two\n<|endoftext|>\n").unwrap();
        let report = sample(vec![first, second], 1_000, &out).unwrap();
        assert_eq!(report.documents, 2);
        assert_eq!(read_back(&out), vec!["one\n", "two\n"]);
    }

    #[test]
    fn skips_rejected_documents_and_counts_them() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("out.txt");
        fs::write(
            &input,
            "fine\n<|endoftext|>\nbad <|im_end|>\n<|endoftext|>\nalso fine\n",
        )
        .unwrap();
        let report = sample(vec![input], 1_000, &out).unwrap();
        assert_eq!(report.documents, 2);
        assert_eq!(report.read.rejected_documents, 1);
        assert_eq!(read_back(&out), vec!["fine\n", "also fine\n"]);
    }

    #[test]
    fn creates_the_output_directory() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("deep").join("er").join("out.txt");
        fs::write(&input, docs(2)).unwrap();
        sample(vec![input], 1_000, &out).unwrap();
        assert!(out.is_file());
    }

    #[test]
    fn replaces_an_existing_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("out.txt");
        fs::write(&input, docs(2)).unwrap();
        fs::write(&out, "old content that is not a sample").unwrap();
        sample(vec![input], 1_000, &out).unwrap();
        assert_eq!(read_back(&out).len(), 2);
    }

    #[test]
    fn rejects_a_zero_budget() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        fs::write(&input, docs(2)).unwrap();
        let error = sample(vec![input], 0, &dir.path().join("out.txt")).unwrap_err();
        assert!(error.to_string().contains("above zero"));
    }

    #[test]
    fn an_input_without_documents_is_an_error_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.txt");
        let out = dir.path().join("out.txt");
        fs::write(&input, "<|endoftext|>\n\n").unwrap();
        let error = sample(vec![input], 1_000, &out).unwrap_err();
        assert!(error.to_string().contains("no documents"));
        assert!(!out.exists());
        assert!(!dir.path().join("out.txt.part").exists());
    }

    #[test]
    fn a_missing_input_is_an_error_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");
        assert!(sample(vec![dir.path().join("nope.txt")], 1_000, &out).is_err());
        assert!(!out.exists());
    }
}
