use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use blom_core::SPECIAL_TOKENS;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{
    AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    TokenizerBuilder,
};

use crate::Error;

pub const MIN_VOCAB_SIZE: usize = 256 + SPECIAL_TOKENS.len();

pub fn train(
    documents: impl Iterator<Item = String> + Send,
    vocab_size: usize,
    out: &Path,
) -> Result<usize, Error> {
    if vocab_size < MIN_VOCAB_SIZE {
        return Err(
            format!("vocab size {vocab_size} is below the minimum {MIN_VOCAB_SIZE}").into(),
        );
    }
    let seen = Arc::new(AtomicU64::new(0));
    let tally = Arc::clone(&seen);
    let stream = documents.inspect(move |_| {
        tally.fetch_add(1, Ordering::Relaxed);
    });

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(
            SPECIAL_TOKENS
                .iter()
                .map(|token| AddedToken::from(*token, true))
                .collect(),
        )
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect::<HashSet<char>>())
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(None::<NormalizerWrapper>)
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(
            ByteLevel::default().add_prefix_space(false),
        )))
        .with_post_processor(Some(PostProcessorWrapper::ByteLevel(
            ByteLevel::default().add_prefix_space(false),
        )))
        .with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevelDecoder::default())))
        .build()?;

    tokenizer.train(&mut trainer, stream)?;
    if seen.load(Ordering::Relaxed) == 0 {
        return Err("no documents to train the tokenizer on".into());
    }
    tokenizer.save(out, true)?;
    Ok(tokenizer.get_vocab_size(true))
}

#[cfg(test)]
mod tests {
    use blom_core::SPECIAL_TOKENS;
    use tokenizers::Tokenizer;

    use super::{MIN_VOCAB_SIZE, train};

    fn corpus() -> Vec<String> {
        (0..200)
            .map(|i| format!("User: hello number {i}, how are you today?\nAssistant: I am fine, thank you for asking about number {i}.\n"))
            .collect()
    }

    #[test]
    fn special_tokens_get_their_fixed_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        train(corpus().into_iter(), 400, &path).unwrap();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        for (id, token) in SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(tokenizer.token_to_id(token), u32::try_from(id).ok());
        }
    }

    #[test]
    fn round_trips_text_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        let size = train(corpus().into_iter(), 400, &path).unwrap();
        assert!(size <= 400);
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        for text in [
            "Assistant: I am fine, thank you!",
            "  leading spaces and\ttabs\n",
            "naïve café 日本語 🙂",
        ] {
            let encoding = tokenizer.encode(text, false).unwrap();
            assert_eq!(tokenizer.decode(encoding.get_ids(), false).unwrap(), text);
        }
    }

    #[test]
    fn rejects_empty_corpus_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        assert!(train(std::iter::empty(), 400, &path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn rejects_too_small_vocabulary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        assert!(train(corpus().into_iter(), MIN_VOCAB_SIZE - 1, &path).is_err());
    }
}
