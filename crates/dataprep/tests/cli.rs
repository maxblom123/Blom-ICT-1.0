#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&dyn AsRef<OsStr>]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dataprep"));
    for arg in args {
        command.arg(arg.as_ref());
    }
    command.output().expect("failed to start dataprep")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn write_docs(path: &Path, prefix: &str, count: usize) {
    let docs: Vec<String> = (0..count)
        .map(|i| format!("{prefix}: hello number {i}, how are you today?\nAssistant: I am fine, thank you for asking about number {i}."))
        .collect();
    fs::write(path, docs.join("\n<|endoftext|>\n")).expect("failed to write test data");
}

#[test]
fn full_pipeline_through_the_real_binary() {
    let dir = tempfile::tempdir().unwrap();
    let train = dir.path().join("train.txt");
    let val = dir.path().join("val.txt");
    let tok_file = dir.path().join("tokenizer.json");
    let out = dir.path().join("out");
    let shard = out.join("train.bin");
    write_docs(&train, "User", 200);
    write_docs(&val, "Visitor", 20);

    let first = run(&[
        &"train-tokenizer",
        &"--vocab-size",
        &"400",
        &"--input",
        &train,
        &"--out",
        &tok_file,
    ]);
    assert!(first.status.success(), "{}", text(&first.stderr));
    assert!(text(&first.stdout).contains("documents 200"));
    assert!(tok_file.is_file());

    let second = run(&[
        &"tokenize",
        &"--tokenizer",
        &tok_file,
        &"--train",
        &train,
        &"--val",
        &val,
        &"--out-dir",
        &out,
    ]);
    assert!(second.status.success(), "{}", text(&second.stderr));
    let report = text(&second.stdout);
    assert!(report.contains("train: 200 documents"));
    assert!(report.contains("val: 20 documents"));
    assert!(shard.is_file());
    assert!(out.join("val.bin").is_file());
    assert!(out.join("meta.json").is_file());

    let third = run(&[
        &"peek",
        &"--tokenizer",
        &tok_file,
        &"--shard",
        &shard,
        &"--tokens",
        &"12",
    ]);
    assert!(third.status.success(), "{}", text(&third.stderr));
    assert!(text(&third.stdout).starts_with("User: hello number 0"));
}

#[test]
fn a_missing_input_file_fails_with_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nope.txt");
    let out = dir.path().join("tokenizer.json");
    let output = run(&[&"train-tokenizer", &"--input", &missing, &"--out", &out]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).starts_with("error: "));
    assert!(text(&output.stderr).contains("nope.txt"));
    assert!(!out.exists());
}

#[test]
fn the_same_file_for_train_and_val_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data.txt");
    let tok_file = dir.path().join("tokenizer.json");
    let out = dir.path().join("out");
    write_docs(&data, "User", 50);

    let first = run(&[
        &"train-tokenizer",
        &"--vocab-size",
        &"400",
        &"--input",
        &data,
        &"--out",
        &tok_file,
    ]);
    assert!(first.status.success(), "{}", text(&first.stderr));

    let second = run(&[
        &"tokenize",
        &"--tokenizer",
        &tok_file,
        &"--train",
        &data,
        &"--val",
        &data,
        &"--out-dir",
        &out,
    ]);
    assert!(!second.status.success());
    assert!(text(&second.stderr).contains("both the train and val"));
    assert!(!out.join("train.bin").exists());
}

#[test]
fn missing_arguments_are_a_usage_error() {
    let output = run(&[&"tokenize"]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("--tokenizer"));
}
