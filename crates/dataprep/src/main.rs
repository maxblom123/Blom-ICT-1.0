use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dataprep::{Error, commands};

#[derive(Parser)]
#[command(name = "dataprep", version, about = "Data pipeline for Blom-ICT-1.0")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Train a byte-level BPE tokenizer on text files")]
    TrainTokenizer {
        #[arg(long, num_args = 1.., required = true, help = "Text files to train on")]
        input: Vec<PathBuf>,
        #[arg(long, default_value_t = 32_000, help = "Vocabulary size")]
        vocab_size: usize,
        #[arg(long, help = "Where to write tokenizer.json")]
        out: PathBuf,
    },
    #[command(about = "Turn text files into train and validation token shards")]
    Tokenize {
        #[arg(long, help = "Path to tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, num_args = 1.., required = true, help = "Training text files")]
        train: Vec<PathBuf>,
        #[arg(long, num_args = 1.., required = true, help = "Validation text files")]
        val: Vec<PathBuf>,
        #[arg(long, help = "Directory for train.bin, val.bin and meta.json")]
        out_dir: PathBuf,
    },
    #[command(about = "Decode the first tokens of a shard back to text")]
    Peek {
        #[arg(long, help = "Path to tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, help = "Path to a .bin shard")]
        shard: PathBuf,
        #[arg(long, default_value_t = 200, help = "How many tokens to show")]
        tokens: usize,
    },
}

fn run(cli: Cli) -> Result<(), Error> {
    match cli.command {
        Command::TrainTokenizer {
            input,
            vocab_size,
            out,
        } => {
            let (size, read) = commands::train_tokenizer(input, vocab_size, &out)?;
            println!("saved {}: vocab size {size}", out.display());
            println!(
                "documents {} | rejected {} | invalid utf-8 lines {}",
                read.documents, read.rejected_documents, read.invalid_utf8_lines
            );
        }
        Command::Tokenize {
            tokenizer,
            train,
            val,
            out_dir,
        } => {
            let report = commands::tokenize(&tokenizer, train, val, &out_dir)?;
            for (name, part) in [("train", report.train), ("val", report.val)] {
                println!(
                    "{name}: {} documents, {} tokens | rejected {} | invalid utf-8 lines {}",
                    part.shard.documents,
                    part.shard.tokens,
                    part.read.rejected_documents,
                    part.read.invalid_utf8_lines
                );
            }
        }
        Command::Peek {
            tokenizer,
            shard,
            tokens,
        } => {
            println!("{}", commands::peek(&tokenizer, &shard, tokens)?);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
