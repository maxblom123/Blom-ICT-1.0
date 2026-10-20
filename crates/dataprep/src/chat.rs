use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Lines, Read, Write};
use std::path::{Path, PathBuf};

use blom_core::SPECIAL_TOKENS;
use blom_core::chat::Role;
use blom_core::hash::SplitMix64;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::Error;
use crate::commands::{ensure_disjoint, load_tokenizer};

const BATCH: usize = 1024;
const MAX_VOCAB: usize = 65_536;
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

pub const TRAINED: u8 = 1;
pub const IGNORED: u8 = 0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Keep {
    pub name: String,
    pub percent: u8,
}

impl Keep {
    pub fn parse(spec: &str) -> Result<Self, Error> {
        let (name, percent) = spec
            .split_once('=')
            .ok_or_else(|| format!("--keep needs NAME=PERCENT, got {spec:?}"))?;
        let percent: u8 = percent
            .parse()
            .map_err(|_| format!("--keep percent must be a whole number, got {percent:?}"))?;
        if name.is_empty() || !(1..=100).contains(&percent) {
            return Err(
                format!("--keep needs a name and a percent from 1 to 100, got {spec:?}").into(),
            );
        }
        Ok(Self {
            name: name.to_string(),
            percent,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatReadStats {
    pub conversations: u64,
    pub rejected: u64,
    pub left_out: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatShardStats {
    pub conversations: u64,
    pub tokens: u64,
    pub trained_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatShardReport {
    pub shard: ChatShardStats,
    pub read: ChatReadStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatTokenizeReport {
    pub vocab_size: usize,
    pub train: ChatShardReport,
    pub val: ChatShardReport,
}

fn shape_is_valid(turns: &[Turn]) -> bool {
    let body = match turns.split_first() {
        Some((first, rest)) if first.role == Role::System => rest,
        Some(_) => turns,
        None => return false,
    };
    !body.is_empty()
        && body.len() % 2 == 0
        && body.iter().enumerate().all(|(index, turn)| {
            turn.role
                == if index % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                }
        })
}

fn parse_role(name: &str) -> Option<Role> {
    match name {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        _ => None,
    }
}

pub fn parse_conversation(line: &str) -> Option<Vec<Turn>> {
    let value: Value = serde_json::from_str(line).ok()?;
    let items = value.get("messages")?.as_array()?;
    let mut turns = Vec::with_capacity(items.len());
    for item in items {
        let role = parse_role(item.get("role")?.as_str()?)?;
        let content = item.get("content")?.as_str()?;
        if content.is_empty() || SPECIAL_TOKENS.iter().any(|token| content.contains(token)) {
            return None;
        }
        turns.push(Turn {
            role,
            content: content.to_string(),
        });
    }
    shape_is_valid(&turns).then_some(turns)
}

fn mix(text: &str) -> u64 {
    let hash = text.bytes().fold(FNV_OFFSET, |acc, byte| {
        (acc ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    });
    SplitMix64::new(hash).next_u64()
}

pub fn is_kept(turns: &[Turn], percent: u8) -> bool {
    if percent >= 100 {
        return true;
    }
    let first_user = turns
        .iter()
        .find(|turn| turn.role == Role::User)
        .map_or("", |turn| turn.content.as_str());
    mix(first_user) % 100 < u64::from(percent)
}

struct Input {
    path: PathBuf,
    percent: u8,
}

pub struct ChatCorpus {
    inputs: Vec<Input>,
    next_input: usize,
    current: Option<(Lines<BufReader<File>>, u8, PathBuf)>,
    stats: ChatReadStats,
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl ChatCorpus {
    pub fn new(files: Vec<PathBuf>, keep: &[Keep]) -> Result<Self, Error> {
        if files.is_empty() {
            return Err("no input files given".into());
        }
        let mut inputs = Vec::with_capacity(files.len());
        for path in files {
            let metadata = fs::metadata(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if !metadata.is_file() {
                return Err(format!("{} is not a file", path.display()).into());
            }
            let name = stem(&path);
            let percent = keep
                .iter()
                .find(|rule| rule.name == name)
                .map_or(100, |rule| rule.percent);
            inputs.push(Input { path, percent });
        }
        Ok(Self {
            inputs,
            next_input: 0,
            current: None,
            stats: ChatReadStats::default(),
        })
    }

    pub fn stats(&self) -> ChatReadStats {
        self.stats
    }
}

impl Iterator for ChatCorpus {
    type Item = io::Result<Vec<Turn>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((lines, percent, path)) = self.current.as_mut() {
                match lines.next() {
                    None => self.current = None,
                    Some(Err(error)) => {
                        let message = format!("{}: {error}", path.display());
                        return Some(Err(io::Error::new(error.kind(), message)));
                    }
                    Some(Ok(line)) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let Some(turns) = parse_conversation(&line) else {
                            self.stats.rejected += 1;
                            continue;
                        };
                        if !is_kept(&turns, *percent) {
                            self.stats.left_out += 1;
                            continue;
                        }
                        self.stats.conversations += 1;
                        return Some(Ok(turns));
                    }
                }
                continue;
            }
            let input = self.inputs.get(self.next_input)?;
            self.next_input += 1;
            match File::open(&input.path) {
                Ok(file) => {
                    let lines = BufReader::new(file).lines();
                    self.current = Some((lines, input.percent, input.path.clone()));
                }
                Err(error) => {
                    let message = format!("{}: {error}", input.path.display());
                    return Some(Err(io::Error::new(error.kind(), message)));
                }
            }
        }
    }
}

pub struct Frame {
    system: Vec<u32>,
    user: Vec<u32>,
    assistant: Vec<u32>,
    end: u32,
    newline: Vec<u32>,
    end_of_text: u32,
}

fn special_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, Error> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| format!("tokenizer has no {token} token").into())
}

fn plain_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    Ok(tokenizer.encode(text, false)?.get_ids().to_vec())
}

impl Frame {
    pub fn new(tokenizer: &Tokenizer) -> Result<Self, Error> {
        let vocab = tokenizer.get_vocab_size(true);
        if vocab > MAX_VOCAB {
            return Err(format!("vocab size {vocab} does not fit in 16-bit shards").into());
        }
        let start = special_id(tokenizer, SPECIAL_TOKENS[1])?;
        let header = |role: Role| -> Result<Vec<u32>, Error> {
            let mut ids = vec![start];
            ids.extend(plain_ids(tokenizer, &format!("{}\n", role.as_str()))?);
            Ok(ids)
        };
        Ok(Self {
            system: header(Role::System)?,
            user: header(Role::User)?,
            assistant: header(Role::Assistant)?,
            end: special_id(tokenizer, SPECIAL_TOKENS[2])?,
            newline: plain_ids(tokenizer, "\n")?,
            end_of_text: special_id(tokenizer, SPECIAL_TOKENS[0])?,
        })
    }

    pub fn end_of_text(&self) -> u32 {
        self.end_of_text
    }

    fn header(&self, role: Role) -> &[u32] {
        match role {
            Role::System => &self.system,
            Role::User => &self.user,
            Role::Assistant => &self.assistant,
        }
    }

    pub fn append(
        &self,
        turns: &[Turn],
        bodies: &[Vec<u32>],
        tokens: &mut Vec<u32>,
        mask: &mut Vec<u8>,
    ) {
        for (turn, body) in turns.iter().zip(bodies) {
            let learn = if turn.role == Role::Assistant {
                TRAINED
            } else {
                IGNORED
            };
            let header = self.header(turn.role);
            tokens.extend_from_slice(header);
            mask.resize(mask.len() + header.len(), IGNORED);
            tokens.extend_from_slice(body);
            tokens.push(self.end);
            mask.resize(mask.len() + body.len() + 1, learn);
            tokens.extend_from_slice(&self.newline);
            mask.resize(mask.len() + self.newline.len(), IGNORED);
        }
        tokens.push(self.end_of_text);
        mask.push(IGNORED);
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(OsString::from).unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

struct Output {
    final_path: PathBuf,
    temp: PathBuf,
    writer: BufWriter<File>,
}

impl Output {
    fn create(path: &Path) -> Result<Self, Error> {
        let temp = temp_path(path);
        let writer = BufWriter::new(File::create(&temp)?);
        Ok(Self {
            final_path: path.to_path_buf(),
            temp,
            writer,
        })
    }
}

pub fn write_chat_shard(
    tokenizer: &Tokenizer,
    conversations: impl Iterator<Item = io::Result<Vec<Turn>>>,
    tokens_path: &Path,
    mask_path: &Path,
) -> Result<ChatShardStats, Error> {
    let frame = Frame::new(tokenizer)?;
    let mut outputs = [Output::create(tokens_path)?, Output::create(mask_path)?];
    let result = write_conversations(tokenizer, &frame, conversations, &mut outputs);
    let flushed = outputs
        .iter_mut()
        .try_for_each(|output| output.writer.flush());
    let temps: Vec<PathBuf> = outputs.iter().map(|output| output.temp.clone()).collect();
    let finals: Vec<PathBuf> = outputs
        .iter()
        .map(|output| output.final_path.clone())
        .collect();
    drop(outputs);
    let cleanup = || {
        for temp in &temps {
            let _ = fs::remove_file(temp);
        }
    };
    match (result, flushed) {
        (Ok(stats), Ok(())) if stats.conversations > 0 => {
            for (temp, target) in temps.iter().zip(&finals) {
                if let Err(error) = fs::rename(temp, target) {
                    cleanup();
                    for target in &finals {
                        let _ = fs::remove_file(target);
                    }
                    return Err(error.into());
                }
            }
            Ok(stats)
        }
        (Ok(_), Ok(())) => {
            cleanup();
            Err("no conversations to write".into())
        }
        (Err(error), _) => {
            cleanup();
            Err(error)
        }
        (Ok(_), Err(error)) => {
            cleanup();
            Err(error.into())
        }
    }
}

fn write_conversations(
    tokenizer: &Tokenizer,
    frame: &Frame,
    mut conversations: impl Iterator<Item = io::Result<Vec<Turn>>>,
    outputs: &mut [Output; 2],
) -> Result<ChatShardStats, Error> {
    let mut stats = ChatShardStats::default();
    let mut tokens: Vec<u32> = Vec::new();
    let mut mask: Vec<u8> = Vec::new();
    loop {
        let mut batch: Vec<Vec<Turn>> = Vec::with_capacity(BATCH);
        for item in conversations.by_ref().take(BATCH) {
            batch.push(item?);
        }
        if batch.is_empty() {
            break;
        }
        let contents: Vec<&str> = batch
            .iter()
            .flatten()
            .map(|turn| turn.content.as_str())
            .collect();
        let mut encodings = tokenizer.encode_batch(contents, false)?.into_iter();
        for turns in &batch {
            let bodies: Vec<Vec<u32>> = turns
                .iter()
                .map(|_| {
                    encodings
                        .next()
                        .map(|encoding| encoding.get_ids().to_vec())
                        .ok_or("tokenizer returned too few encodings")
                })
                .collect::<Result<_, _>>()?;
            tokens.clear();
            mask.clear();
            frame.append(turns, &bodies, &mut tokens, &mut mask);
            let [token_out, mask_out] = &mut *outputs;
            for &id in &tokens {
                token_out
                    .writer
                    .write_all(&u16::try_from(id)?.to_le_bytes())?;
            }
            mask_out.writer.write_all(&mask)?;
            stats.conversations += 1;
            stats.tokens += tokens.len() as u64;
            stats.trained_tokens += mask.iter().map(|&flag| u64::from(flag)).sum::<u64>();
        }
    }
    Ok(stats)
}

pub fn read_mask(path: &Path) -> io::Result<Vec<u8>> {
    fs::read(path)
}

fn check_keep_names(inputs: &[PathBuf], keep: &[Keep]) -> Result<(), Error> {
    let stems: HashSet<String> = inputs.iter().map(|path| stem(path)).collect();
    for rule in keep {
        if !stems.contains(&rule.name) {
            return Err(format!("--keep {} does not match any input file", rule.name).into());
        }
    }
    Ok(())
}

pub fn tokenize_chat(
    tokenizer_path: &Path,
    train_inputs: Vec<PathBuf>,
    val_inputs: Vec<PathBuf>,
    keep: &[Keep],
    out_dir: &Path,
) -> Result<ChatTokenizeReport, Error> {
    let tokenizer = load_tokenizer(tokenizer_path)?;
    ensure_disjoint(&train_inputs, &val_inputs)?;
    let all: Vec<PathBuf> = train_inputs.iter().chain(&val_inputs).cloned().collect();
    check_keep_names(&all, keep)?;
    let mut train_corpus = ChatCorpus::new(train_inputs, keep)?;
    let mut val_corpus = ChatCorpus::new(val_inputs, keep)?;
    fs::create_dir_all(out_dir)?;

    let names = |name: &str| {
        (
            out_dir.join(format!("{name}.bin")),
            out_dir.join(format!("{name}.mask")),
        )
    };
    let (train_bin, train_mask) = names("train");
    let (val_bin, val_mask) = names("val");
    let remove_all = || {
        for path in [&train_bin, &train_mask, &val_bin, &val_mask] {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_file(out_dir.join("meta.json"));
    };

    let train_stats = match write_chat_shard(&tokenizer, &mut train_corpus, &train_bin, &train_mask)
    {
        Ok(stats) => stats,
        Err(error) => {
            remove_all();
            return Err(error);
        }
    };
    let val_stats = match write_chat_shard(&tokenizer, &mut val_corpus, &val_bin, &val_mask) {
        Ok(stats) => stats,
        Err(error) => {
            remove_all();
            return Err(error);
        }
    };

    let report = ChatTokenizeReport {
        vocab_size: tokenizer.get_vocab_size(true),
        train: ChatShardReport {
            shard: train_stats,
            read: train_corpus.stats(),
        },
        val: ChatShardReport {
            shard: val_stats,
            read: val_corpus.stats(),
        },
    };
    let describe = |part: &ChatShardReport| {
        json!({
            "documents": part.shard.conversations,
            "tokens": part.shard.tokens,
            "trained_tokens": part.shard.trained_tokens,
            "rejected_documents": part.read.rejected,
            "left_out_by_keep": part.read.left_out,
        })
    };
    let meta = json!({
        "format": "u16-le",
        "kind": "chat",
        "mask_format": "u8",
        "vocab_size": report.vocab_size,
        "end_of_text_id": 0,
        "train": describe(&report.train),
        "val": describe(&report.val),
    });
    let text = serde_json::to_string_pretty(&meta)?;
    if let Err(error) = fs::write(out_dir.join("meta.json"), text + "\n") {
        remove_all();
        return Err(error.into());
    }
    Ok(report)
}

pub fn peek_masked(
    tokenizer_path: &Path,
    shard: &Path,
    mask: &Path,
    count: usize,
) -> Result<String, Error> {
    let tokenizer = load_tokenizer(tokenizer_path)?;
    let limit = u64::try_from(count)?;
    let mut token_bytes = Vec::new();
    File::open(shard)?
        .take(limit.saturating_mul(2))
        .read_to_end(&mut token_bytes)?;
    let mut flags = Vec::new();
    File::open(mask)?.take(limit).read_to_end(&mut flags)?;
    let (pairs, _) = token_bytes.as_chunks::<2>();
    let ids: Vec<u32> = pairs
        .iter()
        .map(|pair| u32::from(u16::from_le_bytes(*pair)))
        .collect();
    if ids.len() != flags.len() {
        return Err("the mask and the shard have different lengths".into());
    }
    let mut out = String::new();
    let mut start = 0;
    while start < ids.len() {
        let flag = flags.get(start).copied().unwrap_or(IGNORED);
        let end = (start..ids.len())
            .find(|&index| flags.get(index).copied() != Some(flag))
            .unwrap_or(ids.len());
        let run = ids.get(start..end).unwrap_or_default();
        let text = tokenizer.decode(run, false)?;
        if flag == TRAINED {
            out.push_str("[[[");
            out.push_str(&text);
            out.push_str("]]]");
        } else {
            out.push_str(&text);
        }
        start = end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use blom_core::chat::{Message, Role, render};
    use tokenizers::Tokenizer;

    use super::{
        ChatCorpus, Frame, IGNORED, Keep, TRAINED, Turn, is_kept, parse_conversation, peek_masked,
        tokenize_chat, write_chat_shard,
    };
    use crate::shards::read_all;
    use crate::tokenizer::train;

    fn corpus() -> Vec<String> {
        (0..300)
            .map(|i| format!("<|im_start|>user\nhello number {i}, how are you today?<|im_end|>\n<|im_start|>assistant\nI am fine, thank you for asking about number {i}. def f(x):\n    return x\n\n"))
            .map(|text| text.replace("<|im_start|>", " ").replace("<|im_end|>", " "))
            .collect()
    }

    fn tokenizer(dir: &Path) -> Tokenizer {
        let path = dir.join("tokenizer.json");
        train(corpus().into_iter(), 500, &path).unwrap();
        Tokenizer::from_file(&path).unwrap()
    }

    fn turn(role: Role, content: &str) -> Turn {
        Turn {
            role,
            content: content.to_string(),
        }
    }

    fn sample_turns() -> Vec<Turn> {
        vec![
            turn(Role::System, "Be brief."),
            turn(Role::User, "Write f:\n\n    def f(x):\n\treturn x"),
            turn(
                Role::Assistant,
                "Sure! naïve café 日本語 🙂\n\n```python\nx = 1\n```",
            ),
            turn(Role::User, "  thanks  "),
            turn(Role::Assistant, "You're welcome."),
        ]
    }

    fn line(turns: &[(&str, &str)]) -> String {
        let messages: Vec<serde_json::Value> = turns
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect();
        serde_json::json!({ "messages": messages }).to_string()
    }

    fn encode_all(tokenizer: &Tokenizer, turns: &[Turn]) -> (Vec<u32>, Vec<u8>) {
        let frame = Frame::new(tokenizer).unwrap();
        let bodies: Vec<Vec<u32>> = turns
            .iter()
            .map(|t| {
                tokenizer
                    .encode(t.content.as_str(), false)
                    .unwrap()
                    .get_ids()
                    .to_vec()
            })
            .collect();
        let (mut tokens, mut mask) = (Vec::new(), Vec::new());
        frame.append(turns, &bodies, &mut tokens, &mut mask);
        (tokens, mask)
    }

    #[test]
    fn parses_a_valid_conversation_with_a_system_message() {
        let text = line(&[("system", "s"), ("user", "u"), ("assistant", "a")]);
        let turns = parse_conversation(&text).unwrap();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].role, Role::System);
    }

    #[test]
    fn rejects_malformed_or_badly_ordered_conversations() {
        let cases = [
            "not json".to_string(),
            "[]".to_string(),
            "{\"messages\": []}".to_string(),
            "{\"messages\": [{\"role\": \"user\"}]}".to_string(),
            line(&[("user", "u")]),
            line(&[("assistant", "a"), ("user", "u")]),
            line(&[("user", "u"), ("user", "u2"), ("assistant", "a")]),
            line(&[("user", "u"), ("assistant", "a"), ("user", "u2")]),
            line(&[("system", "s")]),
            line(&[("user", "u"), ("system", "s"), ("assistant", "a")]),
            line(&[("user", "u"), ("tool", "t")]),
            line(&[("user", ""), ("assistant", "a")]),
        ];
        for case in cases {
            assert!(parse_conversation(&case).is_none(), "{case}");
        }
    }

    #[test]
    fn rejects_special_token_text() {
        for token in blom_core::SPECIAL_TOKENS {
            let text = line(&[("user", &format!("hi {token}")), ("assistant", "a")]);
            assert!(parse_conversation(&text).is_none());
        }
    }

    #[test]
    fn keep_parses_and_validates() {
        assert_eq!(
            Keep::parse("smoltalk_train=30").unwrap(),
            Keep {
                name: "smoltalk_train".into(),
                percent: 30
            }
        );
        for bad in ["x", "x=", "x=0", "x=101", "x=-1", "=5", "x=1.5"] {
            assert!(Keep::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn keep_is_deterministic_and_roughly_proportional() {
        let make = |i: usize| {
            vec![
                turn(Role::User, &format!("question {i}")),
                turn(Role::Assistant, "a"),
            ]
        };
        let kept = (0..10_000).filter(|&i| is_kept(&make(i), 30)).count();
        assert!((2_700..3_300).contains(&kept), "{kept}");
        assert!((0..100).all(|i| is_kept(&make(i), 100)));
        let again = (0..10_000).filter(|&i| is_kept(&make(i), 30)).count();
        assert_eq!(kept, again);
    }

    #[test]
    fn keep_depends_only_on_the_first_user_message() {
        let a = vec![
            turn(Role::System, "x"),
            turn(Role::User, "same"),
            turn(Role::Assistant, "1"),
        ];
        let b = vec![turn(Role::User, "same"), turn(Role::Assistant, "2")];
        for percent in 1..100 {
            assert_eq!(is_kept(&a, percent), is_kept(&b, percent));
        }
    }

    #[test]
    fn pieces_tokenize_exactly_like_the_whole_rendered_text() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = tokenizer(dir.path());
        let turns = sample_turns();
        let (tokens, _) = encode_all(&tokenizer, &turns);

        let messages: Vec<Message<'_>> = turns
            .iter()
            .map(|t| Message {
                role: t.role,
                content: &t.content,
            })
            .collect();
        let whole = render(&messages, false).unwrap();
        let expected = tokenizer.encode(whole.as_str(), false).unwrap();
        assert_eq!(&tokens[..tokens.len() - 1], expected.get_ids());
        assert_eq!(tokens.last(), Some(&0));
    }

    #[test]
    fn the_prompt_part_matches_what_generation_will_see() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = tokenizer(dir.path());
        let turns = sample_turns();
        let (tokens, mask) = encode_all(&tokenizer, &turns);

        let last_reply = turns.len() - 1;
        let prompt: Vec<Message<'_>> = turns[..last_reply]
            .iter()
            .map(|t| Message {
                role: t.role,
                content: &t.content,
            })
            .collect();
        let rendered = render(&prompt, true).unwrap();
        let prompt_ids = tokenizer.encode(rendered.as_str(), false).unwrap();
        let first_trained_of_last = {
            let reply_ids = tokenizer
                .encode(turns[last_reply].content.as_str(), false)
                .unwrap();
            tokens.len() - 1 - 1 - 1 - reply_ids.get_ids().len()
        };
        assert_eq!(&tokens[..first_trained_of_last], prompt_ids.get_ids());
        assert_eq!(mask[first_trained_of_last], TRAINED);
    }

    #[test]
    fn only_assistant_text_and_its_end_marker_are_trained() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = tokenizer(dir.path());
        let turns = sample_turns();
        let (tokens, mask) = encode_all(&tokenizer, &turns);
        assert_eq!(tokens.len(), mask.len());

        let mut spans: Vec<Vec<u32>> = Vec::new();
        let mut open = false;
        for (&id, &flag) in tokens.iter().zip(&mask) {
            if flag == TRAINED {
                if !open {
                    spans.push(Vec::new());
                    open = true;
                }
                spans.last_mut().unwrap().push(id);
            } else {
                open = false;
            }
        }
        let decoded: Vec<String> = spans
            .iter()
            .map(|ids| tokenizer.decode(ids, false).unwrap())
            .collect();
        assert_eq!(
            decoded,
            vec![
                format!("{}<|im_end|>", turns[2].content),
                format!("{}<|im_end|>", turns[4].content),
            ]
        );
        assert_eq!(mask.last(), Some(&IGNORED));
    }

    fn write_lines(path: &Path, lines: &[String]) {
        fs::write(path, lines.join("\n") + "\n").unwrap();
    }

    fn conversations(count: usize, tag: &str) -> Vec<String> {
        (0..count)
            .map(|i| {
                line(&[
                    (
                        "user",
                        &format!("{tag} question number {i}, how are you today?"),
                    ),
                    (
                        "assistant",
                        &format!("I am fine, thank you for asking about number {i}."),
                    ),
                ])
            })
            .collect()
    }

    #[test]
    fn corpus_counts_rejected_and_left_out_conversations() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a_train.jsonl");
        let second = dir.path().join("b_train.jsonl");
        let mut lines = conversations(300, "a");
        lines.push("garbage".to_string());
        lines.push(String::new());
        write_lines(&first, &lines);
        write_lines(&second, &conversations(50, "b"));

        let keep = [Keep::parse("a_train=50").unwrap()];
        let mut corpus = ChatCorpus::new(vec![first, second], &keep).unwrap();
        let got: Vec<_> = corpus.by_ref().map(Result::unwrap).collect();
        let stats = corpus.stats();
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.conversations, got.len() as u64);
        assert_eq!(stats.conversations + stats.left_out, 350);
        assert!(
            (140..=210).contains(&(stats.conversations - 50)),
            "{stats:?}"
        );
    }

    #[test]
    fn corpus_rejects_bad_inputs_up_front() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ChatCorpus::new(Vec::new(), &[]).is_err());
        assert!(ChatCorpus::new(vec![dir.path().join("missing.jsonl")], &[]).is_err());
        assert!(ChatCorpus::new(vec![dir.path().to_path_buf()], &[]).is_err());
    }

    #[test]
    fn corpus_reports_invalid_utf8_as_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        fs::write(&path, [0xFF_u8, 0xFE, b'\n']).unwrap();
        let mut corpus = ChatCorpus::new(vec![path], &[]).unwrap();
        assert!(corpus.next().unwrap().is_err());
    }

    fn setup(dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let tok_path = dir.join("tokenizer.json");
        train(corpus().into_iter(), 500, &tok_path).unwrap();
        let train_file = dir.join("x_train.jsonl");
        let val_file = dir.join("x_valid.jsonl");
        write_lines(&train_file, &conversations(120, "t"));
        write_lines(&val_file, &conversations(10, "v"));
        (dir.join("out"), tok_path, train_file, val_file)
    }

    #[test]
    fn writes_aligned_token_and_mask_shards_with_meta() {
        let dir = tempfile::tempdir().unwrap();
        let (out, tok, train_file, val_file) = setup(dir.path());
        let report = tokenize_chat(&tok, vec![train_file], vec![val_file], &[], &out).unwrap();
        assert_eq!(report.train.shard.conversations, 120);
        assert_eq!(report.val.shard.conversations, 10);
        for (name, part) in [("train", report.train), ("val", report.val)] {
            let tokens = read_all(&out.join(format!("{name}.bin"))).unwrap();
            let mask = fs::read(out.join(format!("{name}.mask"))).unwrap();
            assert_eq!(tokens.len() as u64, part.shard.tokens);
            assert_eq!(mask.len(), tokens.len());
            assert!(mask.iter().all(|&flag| flag == 0 || flag == 1));
            let trained: u64 = mask.iter().map(|&flag| u64::from(flag)).sum();
            assert_eq!(trained, part.shard.trained_tokens);
            assert!(trained > 0 && trained < part.shard.tokens);
        }
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["format"], "u16-le");
        assert_eq!(meta["kind"], "chat");
        assert_eq!(meta["train"]["documents"], 120);
        assert_eq!(meta["train"]["tokens"], report.train.shard.tokens);
        assert_eq!(meta["end_of_text_id"], 0);
        assert!(!out.join("train.bin.tmp").exists());
        assert!(!out.join("train.mask.tmp").exists());
    }

    #[test]
    fn keep_reduces_the_train_shard_only_for_the_named_file() {
        let dir = tempfile::tempdir().unwrap();
        let (out, tok, train_file, val_file) = setup(dir.path());
        let keep = [Keep::parse("x_train=50").unwrap()];
        let report = tokenize_chat(&tok, vec![train_file], vec![val_file], &keep, &out).unwrap();
        assert!((40..=80).contains(&report.train.shard.conversations));
        assert_eq!(report.val.shard.conversations, 10);
        assert_eq!(
            report.train.read.left_out,
            120 - report.train.shard.conversations
        );
    }

    #[test]
    fn a_keep_name_that_matches_nothing_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let (out, tok, train_file, val_file) = setup(dir.path());
        let keep = [Keep::parse("typo=50").unwrap()];
        let error = tokenize_chat(&tok, vec![train_file], vec![val_file], &keep, &out).unwrap_err();
        assert!(error.to_string().contains("typo"));
        assert!(!out.join("train.bin").exists());
    }

    #[test]
    fn the_same_file_for_train_and_val_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (out, tok, train_file, _) = setup(dir.path());
        let error =
            tokenize_chat(&tok, vec![train_file.clone()], vec![train_file], &[], &out).unwrap_err();
        assert!(error.to_string().contains("both the train and val"));
    }

    #[test]
    fn empty_validation_data_removes_every_output() {
        let dir = tempfile::tempdir().unwrap();
        let (out, tok, train_file, _) = setup(dir.path());
        let bad = dir.path().join("bad_valid.jsonl");
        fs::write(&bad, "garbage\n").unwrap();
        assert!(tokenize_chat(&tok, vec![train_file], vec![bad], &[], &out).is_err());
        for name in [
            "train.bin",
            "train.mask",
            "val.bin",
            "val.mask",
            "meta.json",
        ] {
            assert!(!out.join(name).exists(), "{name}");
        }
    }

    #[test]
    fn a_read_error_leaves_no_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = tokenizer(dir.path());
        let bin = dir.path().join("a.bin");
        let mask = dir.path().join("a.mask");
        let items = vec![
            Ok(sample_turns()),
            Err(std::io::Error::other("disk failure")),
        ];
        assert!(write_chat_shard(&tokenizer, items.into_iter(), &bin, &mask).is_err());
        assert!(!bin.exists() && !mask.exists());
        assert!(!dir.path().join("a.bin.tmp").exists());
        assert!(!dir.path().join("a.mask.tmp").exists());
    }

    #[test]
    fn peek_marks_the_trained_spans() {
        let dir = tempfile::tempdir().unwrap();
        let (out, tok, train_file, val_file) = setup(dir.path());
        tokenize_chat(&tok, vec![train_file], vec![val_file], &[], &out).unwrap();
        let text = peek_masked(&tok, &out.join("train.bin"), &out.join("train.mask"), 60).unwrap();
        assert!(
            text.starts_with("<|im_start|>user\nt question number 0"),
            "{text}"
        );
        assert!(
            text.contains("<|im_start|>assistant\n[[[I am fine"),
            "{text}"
        );
        assert!(text.contains("number 0.<|im_end|>]]]\n"), "{text}");
    }
}
