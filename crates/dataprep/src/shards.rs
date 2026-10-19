use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use blom_core::SPECIAL_TOKENS;
use tokenizers::Tokenizer;

use crate::Error;

const BATCH: usize = 2048;
const MAX_VOCAB: usize = 65_536;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShardStats {
    pub documents: u64,
    pub tokens: u64,
}

pub fn write_shard(
    tokenizer: &Tokenizer,
    documents: impl Iterator<Item = io::Result<String>>,
    out: &Path,
) -> Result<ShardStats, Error> {
    let vocab = tokenizer.get_vocab_size(true);
    if vocab > MAX_VOCAB {
        return Err(format!("vocab size {vocab} does not fit in 16-bit shards").into());
    }
    let end_id = tokenizer
        .token_to_id(SPECIAL_TOKENS[0])
        .ok_or("tokenizer has no end-of-text token")?;

    let temp = out.with_extension("tmp");
    match write_documents(tokenizer, documents, end_id, &temp) {
        Ok(stats) if stats.documents > 0 => {
            fs::rename(&temp, out)?;
            Ok(stats)
        }
        Ok(_) => {
            fs::remove_file(&temp)?;
            Err("no documents to write".into())
        }
        Err(error) => {
            let _ = fs::remove_file(&temp);
            Err(error)
        }
    }
}

fn write_documents(
    tokenizer: &Tokenizer,
    mut documents: impl Iterator<Item = io::Result<String>>,
    end_id: u32,
    path: &Path,
) -> Result<ShardStats, Error> {
    let mut writer = BufWriter::new(File::create(path)?);
    let mut stats = ShardStats::default();
    loop {
        let mut batch = Vec::with_capacity(BATCH);
        for item in documents.by_ref().take(BATCH) {
            let document = item?;
            if SPECIAL_TOKENS.iter().any(|token| document.contains(token)) {
                return Err("a document contains a special token string".into());
            }
            batch.push(document);
        }
        if batch.is_empty() {
            break;
        }
        for encoding in tokenizer.encode_batch(batch, false)? {
            for &id in encoding.get_ids().iter().chain(std::iter::once(&end_id)) {
                writer.write_all(&u16::try_from(id)?.to_le_bytes())?;
                stats.tokens += 1;
            }
            stats.documents += 1;
        }
    }
    writer.flush()?;
    Ok(stats)
}

pub fn read_all(path: &Path) -> io::Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    let (pairs, rest) = bytes.as_chunks::<2>();
    if !rest.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shard length is not a multiple of 2 bytes",
        ));
    }
    Ok(pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect())
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::path::Path;

    use tokenizers::Tokenizer;

    use super::{read_all, write_shard};
    use crate::tokenizer::train;

    fn corpus() -> Vec<String> {
        (0..200)
            .map(|i| format!("User: hello number {i}, how are you today?\nAssistant: I am fine, thank you for asking about number {i}.\n"))
            .collect()
    }

    fn trained(dir: &Path) -> Tokenizer {
        let path = dir.join("tokenizer.json");
        train(corpus().into_iter(), 400, &path).unwrap();
        Tokenizer::from_file(&path).unwrap()
    }

    fn ok(documents: &[&str]) -> Vec<io::Result<String>> {
        documents.iter().map(|d| Ok((*d).to_string())).collect()
    }

    #[test]
    fn round_trips_documents_through_a_shard() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = trained(dir.path());
        let out = dir.path().join("train.bin");
        let documents = [
            "Hello there, friend.\n",
            "naïve café 日本語 🙂\n",
            "third\n",
        ];

        let stats = write_shard(&tokenizer, ok(&documents).into_iter(), &out).unwrap();
        assert_eq!(stats.documents, 3);

        let tokens = read_all(&out).unwrap();
        assert_eq!(tokens.len() as u64, stats.tokens);
        let decoded: Vec<String> = tokens
            .split(|&id| id == 0)
            .filter(|part| !part.is_empty())
            .map(|part| {
                let ids: Vec<u32> = part.iter().map(|&id| u32::from(id)).collect();
                tokenizer.decode(&ids, false).unwrap()
            })
            .collect();
        assert_eq!(decoded, documents);
        assert!(!out.with_extension("tmp").exists());
    }

    #[test]
    fn every_document_ends_with_end_of_text() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = trained(dir.path());
        let out = dir.path().join("train.bin");
        write_shard(&tokenizer, ok(&["one\n", "two\n"]).into_iter(), &out).unwrap();
        let tokens = read_all(&out).unwrap();
        assert_eq!(tokens.iter().filter(|&&id| id == 0).count(), 2);
        assert_eq!(tokens.last(), Some(&0));
    }

    #[test]
    fn no_documents_is_an_error_and_leaves_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = trained(dir.path());
        let out = dir.path().join("val.bin");
        assert!(write_shard(&tokenizer, std::iter::empty(), &out).is_err());
        assert!(!out.exists());
        assert!(!out.with_extension("tmp").exists());
    }

    #[test]
    fn special_token_text_is_an_error_and_leaves_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = trained(dir.path());
        let out = dir.path().join("train.bin");
        let documents = ok(&["fine\n", "bad <|im_start|> text\n"]);
        assert!(write_shard(&tokenizer, documents.into_iter(), &out).is_err());
        assert!(!out.exists());
        assert!(!out.with_extension("tmp").exists());
    }

    #[test]
    fn read_errors_are_propagated_and_leave_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = trained(dir.path());
        let out = dir.path().join("train.bin");
        let documents = vec![
            Ok("fine\n".to_string()),
            Err(io::Error::other("disk failure")),
        ];
        assert!(write_shard(&tokenizer, documents.into_iter(), &out).is_err());
        assert!(!out.exists());
        assert!(!out.with_extension("tmp").exists());
    }

    #[test]
    fn odd_length_shard_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.bin");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        assert!(read_all(&path).is_err());
    }
}
