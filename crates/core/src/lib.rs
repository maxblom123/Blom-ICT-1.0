pub mod chat;
pub mod hash;

pub const SPECIAL_TOKENS: [&str; 3] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];

#[cfg(test)]
mod tests {
    use super::SPECIAL_TOKENS;

    #[test]
    fn special_tokens_are_unique() {
        for (i, a) in SPECIAL_TOKENS.iter().enumerate() {
            for b in &SPECIAL_TOKENS[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn end_of_text_is_id_zero() {
        assert_eq!(SPECIAL_TOKENS[0], "<|endoftext|>");
    }
}
