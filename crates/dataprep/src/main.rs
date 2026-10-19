use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dataprep::sample::sample;
use dataprep::sketch::{SketchParams, build_sketch};
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
    #[command(about = "Copy whole documents from text files until a byte budget is reached")]
    Sample {
        #[arg(long, num_args = 1.., required = true, help = "Text files to sample from, in order")]
        input: Vec<PathBuf>,
        #[arg(long, help = "Stop after at least this many bytes")]
        max_bytes: u64,
        #[arg(long, help = "Where to write the sample")]
        out: PathBuf,
    },
    #[command(about = "Build the frozen n-gram sketch table from the training shard")]
    BuildSketch {
        #[arg(long, help = "Directory with train.bin and meta.json")]
        data_dir: PathBuf,
        #[arg(long, help = "Directory for the sketch files")]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 4_194_301, help = "Rows in the table")]
        rows: u64,
        #[arg(long, num_args = 1.., default_values_t = [2_usize, 3], help = "N-gram orders")]
        orders: Vec<usize>,
        #[arg(long, default_value_t = 2, help = "Hash functions per order")]
        hashes: usize,
        #[arg(long, default_value_t = 32, help = "Width of one stored vector")]
        part_dim: usize,
        #[arg(
            long,
            default_value_t = 20_261_016,
            help = "Seed for hashes and projection"
        )]
        seed: u64,
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
        Command::Sample {
            input,
            max_bytes,
            out,
        } => {
            let report = sample(input, max_bytes, &out)?;
            println!(
                "sampled {} documents, {} bytes to {} | rejected {} | invalid utf-8 lines {}",
                report.documents,
                report.bytes,
                out.display(),
                report.read.rejected_documents,
                report.read.invalid_utf8_lines
            );
        }
        Command::BuildSketch {
            data_dir,
            out_dir,
            rows,
            orders,
            hashes,
            part_dim,
            seed,
        } => {
            let params = SketchParams {
                seed,
                rows,
                orders,
                n_hash: hashes,
                part_dim,
            };
            let report = build_sketch(&data_dir, &out_dir, &params)?;
            let permille = u128::from(report.rows_used) * 1000 / u128::from(report.rows.max(1));
            println!("sketch written to {}", out_dir.display());
            println!(
                "pairs {} | lookups {} | rows used {} of {} ({}.{}%) | max count {}",
                report.pairs,
                report.lookups,
                report.rows_used,
                report.rows,
                permille / 10,
                permille % 10,
                report.max_count
            );
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
