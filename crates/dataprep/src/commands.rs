use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::json;
use tokenizers::Tokenizer;

use crate::Error;
use crate::corpus::Corpus;
use crate::docs::ReadStats;
use crate::shards::{ShardStats, write_shard};
use crate::tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardReport {
    pub shard: ShardStats,
    pub read: ReadStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizeReport {
    pub vocab_size: usize,
    pub train: ShardReport,
    pub val: ShardReport,
}

pub fn train_tokenizer(
    inputs: Vec<PathBuf>,
    vocab_size: usize,
    out: &Path,
) -> Result<(usize, ReadStats), Error> {
    let mut corpus = Corpus::new(inputs)?;
    let failure: Arc<Mutex<Option<io::Error>>> = Arc::default();
    let sink = Arc::clone(&failure);
    let documents = corpus.by_ref().map_while(move |item| match item {
        Ok(document) => Some(document),
        Err(error) => {
            if let Ok(mut slot) = sink.lock() {
                *slot = Some(error);
            }
            None
        }
    });
    let result = tokenizer::train(documents, vocab_size, out);
    let failed = failure.lock().ok().and_then(|mut slot| slot.take());
    if let Some(error) = failed {
        let _ = fs::remove_file(out);
        return Err(error.into());
    }
    Ok((result?, corpus.stats()))
}

pub(crate) fn load_tokenizer(path: &Path) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(path)
        .map_err(|error| format!("cannot load tokenizer {}: {error}", path.display()).into())
}

pub(crate) fn ensure_disjoint(train: &[PathBuf], val: &[PathBuf]) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for path in train {
        seen.insert(fs::canonicalize(path)?);
    }
    for path in val {
        if seen.contains(&fs::canonicalize(path)?) {
            return Err(format!("{} is in both the train and val inputs", path.display()).into());
        }
    }
    Ok(())
}

pub fn tokenize(
    tokenizer_path: &Path,
    train_inputs: Vec<PathBuf>,
    val_inputs: Vec<PathBuf>,
    out_dir: &Path,
) -> Result<TokenizeReport, Error> {
    let tokenizer = load_tokenizer(tokenizer_path)?;
    ensure_disjoint(&train_inputs, &val_inputs)?;
    let mut train_corpus = Corpus::new(train_inputs)?;
    let mut val_corpus = Corpus::new(val_inputs)?;
    fs::create_dir_all(out_dir)?;

    let train_path = out_dir.join("train.bin");
    let val_path = out_dir.join("val.bin");
    let train_stats = write_shard(&tokenizer, &mut train_corpus, &train_path)?;
    let val_stats = match write_shard(&tokenizer, &mut val_corpus, &val_path) {
        Ok(stats) => stats,
        Err(error) => {
            let _ = fs::remove_file(&train_path);
            return Err(error);
        }
    };

    let report = TokenizeReport {
        vocab_size: tokenizer.get_vocab_size(true),
        train: ShardReport {
            shard: train_stats,
            read: train_corpus.stats(),
        },
        val: ShardReport {
            shard: val_stats,
            read: val_corpus.stats(),
        },
    };
    let describe = |part: &ShardReport| {
        json!({
            "documents": part.shard.documents,
            "tokens": part.shard.tokens,
            "rejected_documents": part.read.rejected_documents,
            "invalid_utf8_lines": part.read.invalid_utf8_lines,
        })
    };
    let meta = json!({
        "format": "u16-le",
        "vocab_size": report.vocab_size,
        "end_of_text_id": 0,
        "train": describe(&report.train),
        "val": describe(&report.val),
    });
    let text = serde_json::to_string_pretty(&meta)?;
    fs::write(out_dir.join("meta.json"), text + "\n")?;
    Ok(report)
}

pub fn peek(tokenizer_path: &Path, shard: &Path, count: usize) -> Result<String, Error> {
    let tokenizer = load_tokenizer(tokenizer_path)?;
    let mut bytes = Vec::new();
    File::open(shard)?
        .take(u64::try_from(count)?.saturating_mul(2))
        .read_to_end(&mut bytes)?;
    let (pairs, _) = bytes.as_chunks::<2>();
    let ids: Vec<u32> = pairs
        .iter()
        .map(|pair| u32::from(u16::from_le_bytes(*pair)))
        .collect();
    tokenizer.decode(&ids, false)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{peek, tokenize, train_tokenizer};
    use crate::shards::read_all;

    fn write_docs(path: &Path, prefix: &str, count: usize) {
        let mut text = String::new();
        for i in 0..count {
            writeln!(
                text,
                "{prefix}: hello number {i}, how are you today?\nAssistant: I am fine, thank you for asking about number {i}.\n<|endoftext|>"
            )
            .unwrap();
        }
        fs::write(path, text).unwrap();
    }

    struct Setup {
        dir: tempfile::TempDir,
        train: PathBuf,
        val: PathBuf,
        tokenizer: PathBuf,
    }

    fn setup() -> Setup {
        let dir = tempfile::tempdir().unwrap();
        let train = dir.path().join("train.txt");
        let val = dir.path().join("val.txt");
        write_docs(&train, "User", 200);
        write_docs(&val, "Visitor", 20);
        let tokenizer = dir.path().join("tokenizer.json");
        train_tokenizer(vec![train.clone()], 400, &tokenizer).unwrap();
        Setup {
            dir,
            train,
            val,
            tokenizer,
        }
    }

    #[test]
    fn tokenizes_train_and_val_into_separate_shards_with_meta() {
        let s = setup();
        let out = s.dir.path().join("out");
        let report = tokenize(&s.tokenizer, vec![s.train], vec![s.val], &out).unwrap();
        assert_eq!(report.train.shard.documents, 200);
        assert_eq!(report.val.shard.documents, 20);
        assert_eq!(
            read_all(&out.join("train.bin")).unwrap().len() as u64,
            report.train.shard.tokens
        );
        assert_eq!(
            read_all(&out.join("val.bin")).unwrap().len() as u64,
            report.val.shard.tokens
        );
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["format"], "u16-le");
        assert_eq!(meta["train"]["documents"], 200);
        assert_eq!(meta["val"]["documents"], 20);
        assert_eq!(meta["end_of_text_id"], 0);
    }

    #[test]
    fn rejects_a_file_used_for_both_train_and_val() {
        let s = setup();
        let out = s.dir.path().join("out");
        let result = tokenize(&s.tokenizer, vec![s.train.clone()], vec![s.train], &out);
        assert!(result.is_err());
        assert!(!out.join("train.bin").exists());
    }

    #[test]
    fn empty_validation_data_fails_and_removes_the_train_shard() {
        let s = setup();
        let out = s.dir.path().join("out");
        let bad = s.dir.path().join("bad.txt");
        fs::write(&bad, "only <|im_start|> bad\n").unwrap();
        let result = tokenize(&s.tokenizer, vec![s.train], vec![bad], &out);
        assert!(result.is_err());
        assert!(!out.join("train.bin").exists());
        assert!(!out.join("val.bin").exists());
        assert!(!out.join("meta.json").exists());
    }

    #[test]
    fn reports_rejected_documents() {
        let s = setup();
        let out = s.dir.path().join("out");
        let mixed = s.dir.path().join("mixed.txt");
        fs::write(&mixed, "fine\n<|endoftext|>\nbad <|im_end|>\n").unwrap();
        let report = tokenize(&s.tokenizer, vec![s.train], vec![mixed], &out).unwrap();
        assert_eq!(report.val.shard.documents, 1);
        assert_eq!(report.val.read.rejected_documents, 1);
    }

    #[test]
    fn missing_tokenizer_is_an_error() {
        let s = setup();
        let out = s.dir.path().join("out");
        let missing = s.dir.path().join("nope.json");
        assert!(tokenize(&missing, vec![s.train], vec![s.val], &out).is_err());
    }

    #[test]
    fn training_the_tokenizer_on_a_missing_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("tokenizer.json");
        let missing = dir.path().join("nope.txt");
        assert!(train_tokenizer(vec![missing], 400, &out).is_err());
        assert!(!out.exists());
    }

    #[test]
    fn peek_shows_the_first_tokens_as_text() {
        let s = setup();
        let out = s.dir.path().join("out");
        tokenize(&s.tokenizer, vec![s.train], vec![s.val], &out).unwrap();
        let text = peek(&s.tokenizer, &out.join("train.bin"), 12).unwrap();
        assert!(text.starts_with("User: hello number 0"));
        let longer = peek(&s.tokenizer, &out.join("train.bin"), 1000).unwrap();
        assert!(longer.len() > text.len());
        assert!(longer.contains("<|endoftext|>"));
    }
}
