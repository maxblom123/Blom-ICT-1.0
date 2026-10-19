pub mod docs;
pub mod tokenizer;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
