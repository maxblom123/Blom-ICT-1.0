pub mod chat;
pub mod commands;
pub mod corpus;
pub mod docs;
pub mod sample;
pub mod shards;
pub mod sketch;
pub mod tokenizer;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
