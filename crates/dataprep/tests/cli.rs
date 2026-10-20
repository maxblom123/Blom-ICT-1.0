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

#[test]
fn the_sketch_is_built_through_the_real_binary() {
    let dir = tempfile::tempdir().unwrap();
    let train = dir.path().join("train.txt");
    let val = dir.path().join("val.txt");
    let tok_file = dir.path().join("tokenizer.json");
    let data = dir.path().join("data");
    let sketch = dir.path().join("sketch");
    write_docs(&train, "User", 200);
    write_docs(&val, "Visitor", 20);

    let made = run(&[
        &"train-tokenizer",
        &"--vocab-size",
        &"400",
        &"--input",
        &train,
        &"--out",
        &tok_file,
    ]);
    assert!(made.status.success(), "{}", text(&made.stderr));
    let tokenized = run(&[
        &"tokenize",
        &"--tokenizer",
        &tok_file,
        &"--train",
        &train,
        &"--val",
        &val,
        &"--out-dir",
        &data,
    ]);
    assert!(tokenized.status.success(), "{}", text(&tokenized.stderr));

    let built = run(&[
        &"build-sketch",
        &"--data-dir",
        &data,
        &"--out-dir",
        &sketch,
        &"--rows",
        &"1009",
        &"--orders",
        &"2",
        &"3",
        &"--hashes",
        &"2",
        &"--part-dim",
        &"4",
    ]);
    assert!(built.status.success(), "{}", text(&built.stderr));
    let report = text(&built.stdout);
    assert!(report.contains("sketch written to"), "{report}");
    assert!(report.contains("lookups 4"), "{report}");
    assert!(report.contains("of 1009"), "{report}");

    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(sketch.join("sketch.json")).unwrap()).unwrap();
    assert_eq!(manifest["format"], "sketch-v1");
    assert_eq!(manifest["rows"], 1009);
    assert_eq!(manifest["part_dim"], 4);
    let vocab = manifest["vocab_size"].as_u64().unwrap();
    assert_eq!(
        fs::metadata(sketch.join("sums.f32")).unwrap().len(),
        1009 * 4 * 4
    );
    assert_eq!(
        fs::metadata(sketch.join("counts.u32")).unwrap().len(),
        1009 * 4
    );
    assert_eq!(
        fs::metadata(sketch.join("projection.f32")).unwrap().len(),
        vocab * 4 * 4
    );
}

#[test]
fn building_a_sketch_without_data_fails_with_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nothing");
    let out = dir.path().join("sketch");
    let output = run(&[&"build-sketch", &"--data-dir", &missing, &"--out-dir", &out]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).starts_with("error: "));
    assert!(text(&output.stderr).contains("meta.json"));
    assert!(!out.exists());
}

#[test]
fn the_sketch_defaults_are_visible_in_the_help() {
    let output = run(&[&"build-sketch", &"--help"]);
    let help = text(&output.stdout);
    assert!(output.status.success());
    assert!(help.contains("4194301"), "{help}");
    assert!(help.contains("--part-dim"), "{help}");
}

#[test]
fn sampling_goes_through_the_real_binary() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data.txt");
    let out = dir.path().join("sub").join("sample.txt");
    write_docs(&data, "User", 100);

    let output = run(&[
        &"sample",
        &"--input",
        &data,
        &"--max-bytes",
        &"500",
        &"--out",
        &out,
    ]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let report = text(&output.stdout);
    assert!(report.starts_with("sampled "), "{report}");
    assert!(report.contains("rejected 0"), "{report}");
    let sampled = fs::read_to_string(&out).unwrap();
    assert!(sampled.starts_with("User: hello number 0"));
    assert!(sampled.len() >= 500);
    assert!(sampled.len() < 1_000);
    assert!(sampled.ends_with("<|endoftext|>\n"));
}

#[test]
fn sampling_with_a_zero_budget_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data.txt");
    let out = dir.path().join("sample.txt");
    write_docs(&data, "User", 10);
    let output = run(&[
        &"sample",
        &"--input",
        &data,
        &"--max-bytes",
        &"0",
        &"--out",
        &out,
    ]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("above zero"));
    assert!(!out.exists());
}

fn write_chats(path: &Path, tag: &str, count: usize) {
    let lines: Vec<String> = (0..count)
        .map(|i| {
            serde_json::json!({"messages": [
                {"role": "user", "content": format!("{tag}: hello number {i}, how are you today?")},
                {"role": "assistant", "content": format!("I am fine, thank you for asking about number {i}.")},
            ]})
            .to_string()
        })
        .collect();
    fs::write(path, lines.join("\n") + "\n").expect("failed to write chat data");
}

#[test]
fn chat_pipeline_through_the_real_binary() {
    let dir = tempfile::tempdir().unwrap();
    let text_file = dir.path().join("text.txt");
    let train = dir.path().join("sft_train.jsonl");
    let val = dir.path().join("sft_valid.jsonl");
    let tok_file = dir.path().join("tokenizer.json");
    let out = dir.path().join("out");
    write_docs(&text_file, "User", 200);
    write_chats(&train, "t", 100);
    write_chats(&val, "v", 10);

    let made = run(&[
        &"train-tokenizer",
        &"--vocab-size",
        &"400",
        &"--input",
        &text_file,
        &"--out",
        &tok_file,
    ]);
    assert!(made.status.success(), "{}", text(&made.stderr));

    let tokenized = run(&[
        &"tokenize-chat",
        &"--tokenizer",
        &tok_file,
        &"--train",
        &train,
        &"--val",
        &val,
        &"--keep",
        &"sft_train=100",
        &"--out-dir",
        &out,
    ]);
    assert!(tokenized.status.success(), "{}", text(&tokenized.stderr));
    let report = text(&tokenized.stdout);
    assert!(report.contains("train: 100 conversations"), "{report}");
    assert!(report.contains("val: 10 conversations"), "{report}");
    for name in [
        "train.bin",
        "train.mask",
        "val.bin",
        "val.mask",
        "meta.json",
    ] {
        assert!(out.join(name).is_file(), "{name}");
    }

    let peeked = run(&[
        &"peek",
        &"--tokenizer",
        &tok_file,
        &"--shard",
        &out.join("train.bin"),
        &"--mask",
        &out.join("train.mask"),
        &"--tokens",
        &"40",
    ]);
    assert!(peeked.status.success(), "{}", text(&peeked.stderr));
    let shown = text(&peeked.stdout);
    assert!(
        shown.starts_with("<|im_start|>user\nt: hello number 0"),
        "{shown}"
    );
    assert!(
        shown.contains("<|im_start|>assistant\n[[[I am fine"),
        "{shown}"
    );
}

#[test]
fn a_keep_rule_for_an_unknown_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let text_file = dir.path().join("text.txt");
    let train = dir.path().join("a_train.jsonl");
    let val = dir.path().join("a_valid.jsonl");
    let tok_file = dir.path().join("tokenizer.json");
    write_docs(&text_file, "User", 50);
    write_chats(&train, "t", 10);
    write_chats(&val, "v", 5);
    let made = run(&[
        &"train-tokenizer",
        &"--vocab-size",
        &"400",
        &"--input",
        &text_file,
        &"--out",
        &tok_file,
    ]);
    assert!(made.status.success(), "{}", text(&made.stderr));
    let output = run(&[
        &"tokenize-chat",
        &"--tokenizer",
        &tok_file,
        &"--train",
        &train,
        &"--val",
        &val,
        &"--keep",
        &"b_train=50",
        &"--out-dir",
        &dir.path().join("out"),
    ]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("b_train"));
}
