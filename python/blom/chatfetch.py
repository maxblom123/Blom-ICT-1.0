import argparse
import gzip
import http.client
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import pyarrow as pa
import pyarrow.parquet as pq

from blom.fetch import (
    BATCH_ROWS,
    SPECIAL_TOKENS,
    TIMEOUT_SECONDS,
    FetchError,
    Stats,
    download,
    is_valid_key,
)

CHAT_VALID_MODULUS = 200
HUB = "https://huggingface.co"
ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
}
DOWNLOAD_ATTEMPTS = 5
OPENCODE_MIN_SCORE = 0.9
MAGPIE_QUALITIES = frozenset({"good", "excellent"})

Message = dict[str, str]
Row = Mapping[str, object]
Cleaned = tuple[list[Message] | None, str]


@dataclass(frozen=True)
class ChatRule:
    min_chars: int = 20
    max_chars: int = 60_000
    max_non_ascii: float = 0.15


Adapter = Callable[[Row, ChatRule], Cleaned]
Lister = Callable[[str], list[str]]


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    pattern: str
    reader: str
    adapter: Adapter
    columns: tuple[str, ...] | None = None


def non_ascii_fraction(text: str) -> float:
    if text.isascii():
        return 0.0
    return sum(1 for char in text if ord(char) > 127) / len(text)


def clean_messages(
    raw: object, rule: ChatRule, *, role_key: str = "role", content_key: str = "content"
) -> Cleaned:
    if not isinstance(raw, list) or not raw:
        return None, "malformed"
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            return None, "malformed"
        role = ROLE_ALIASES.get(str(item.get(role_key)))
        content = item.get(content_key)
        if role is None or not isinstance(content, str):
            return None, "malformed"
        text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return None, "empty"
        messages.append({"role": role, "content": text})
    turns = messages[1:] if messages[0]["role"] == "system" else messages
    if not turns or len(turns) % 2 != 0:
        return None, "roles"
    for index, message in enumerate(turns):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            return None, "roles"
    joined = "\n".join(message["content"] for message in messages)
    if any(token in joined for token in SPECIAL_TOKENS):
        return None, "special_token"
    if len(joined) < rule.min_chars:
        return None, "short"
    if len(joined) > rule.max_chars:
        return None, "long"
    if non_ascii_fraction(joined) > rule.max_non_ascii:
        return None, "language"
    try:
        joined.encode("utf-8")
    except UnicodeEncodeError:
        return None, "unencodable"
    return messages, "kept"


def adapt_messages(row: Row, rule: ChatRule) -> Cleaned:
    return clean_messages(row.get("messages"), rule)


def pair_adapter(prompt_key: str, answer_key: str) -> Adapter:
    def adapt(row: Row, rule: ChatRule) -> Cleaned:
        prompt = row.get(prompt_key)
        answer = row.get(answer_key)
        if not isinstance(prompt, str) or not isinstance(answer, str):
            return None, "malformed"
        raw = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        return clean_messages(raw, rule)

    return adapt


def adapt_opencode(row: Row, rule: ChatRule) -> Cleaned:
    score = row.get("average_test_score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None, "unscored"
    if score < OPENCODE_MIN_SCORE:
        return None, "low_score"
    return pair_adapter("input", "output")(row, rule)


def adapt_magpie(row: Row, rule: ChatRule) -> Cleaned:
    if row.get("language") != "EN":
        return None, "language"
    if row.get("input_quality") not in MAGPIE_QUALITIES:
        return None, "quality"
    if row.get("llama_guard_2") != "safe":
        return None, "unsafe"
    if row.get("repeat_count") != 0:
        return None, "repeated"
    return clean_messages(row.get("conversations"), rule, role_key="from", content_key="value")


SOURCES = (
    Source(
        name="smoltalk",
        repo="HuggingFaceTB/smoltalk",
        pattern=r"^data/all/train-.*\.parquet$",
        reader="parquet",
        adapter=adapt_messages,
        columns=("messages",),
    ),
    Source(
        name="tulu3",
        repo="allenai/tulu-3-sft-mixture",
        pattern=r"^data/train-.*\.parquet$",
        reader="parquet",
        adapter=adapt_messages,
        columns=("messages",),
    ),
    Source(
        name="ultrachat",
        repo="HuggingFaceH4/ultrachat_200k",
        pattern=r"^data/train_sft-.*\.parquet$",
        reader="parquet",
        adapter=adapt_messages,
        columns=("messages",),
    ),
    Source(
        name="oasst2",
        repo="OpenAssistant/oasst2",
        pattern=r"^2023-11-05_oasst2_ready\.messages\.jsonl\.gz$",
        reader="oasst",
        adapter=adapt_messages,
    ),
    Source(
        name="magpie",
        repo="Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1",
        pattern=r"^data/train-.*\.parquet$",
        reader="parquet",
        adapter=adapt_magpie,
        columns=("conversations", "language", "input_quality", "llama_guard_2", "repeat_count"),
    ),
    Source(
        name="codefeedback",
        repo="m-a-p/CodeFeedback-Filtered-Instruction",
        pattern=r"^CodeFeedback-Filtered-Instruction\.jsonl$",
        reader="jsonl",
        adapter=pair_adapter("query", "answer"),
    ),
    Source(
        name="oss-instruct",
        repo="ise-uiuc/Magicoder-OSS-Instruct-75K",
        pattern=r"^data-oss_instruct-decontaminated\.jsonl$",
        reader="jsonl",
        adapter=pair_adapter("problem", "solution"),
    ),
    Source(
        name="evol-code",
        repo="ise-uiuc/Magicoder-Evol-Instruct-110K",
        pattern=r"^data-evol_instruct-decontaminated\.jsonl$",
        reader="jsonl",
        adapter=pair_adapter("instruction", "response"),
    ),
    Source(
        name="opencode",
        repo="nvidia/OpenCodeInstruct",
        pattern=r"^data/train-.*\.parquet$",
        reader="parquet",
        adapter=adapt_opencode,
        columns=("input", "output", "average_test_score"),
    ),
)
SOURCE_NAMES = tuple(source.name for source in SOURCES)


def select_files(names: Iterable[str], pattern: str) -> list[str]:
    matcher = re.compile(pattern)
    return sorted(name for name in names if matcher.search(name))


def list_repo_files(repo: str) -> list[str]:
    request = urllib.request.Request(
        f"{HUB}/api/datasets/{repo}", headers={"User-Agent": "blom-chatfetch"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        info = json.load(response)
    siblings = info.get("siblings") if isinstance(info, dict) else None
    if not isinstance(siblings, list):
        raise FetchError(f"{repo}: the hub did not list any files")
    return [
        item["rfilename"] for item in siblings if isinstance(item, dict) and "rfilename" in item
    ]


def download_retrying(
    url: str,
    destination: Path,
    *,
    attempts: int = DOWNLOAD_ATTEMPTS,
    pause: Callable[[float], None] = time.sleep,
) -> bool:
    if attempts < 1:
        raise FetchError("attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return download(url, destination)
        except (FetchError, OSError, http.client.HTTPException) as error:
            if attempt == attempts:
                raise
            print(f"retry {attempt}/{attempts - 1} after: {error}", file=sys.stderr, flush=True)
            pause(2.0**attempt)
    raise FetchError(f"{url}: download failed")


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FetchError(f"{path} line {number}: {error}") from error
            if not isinstance(record, dict):
                raise FetchError(f"{path} line {number}: not a JSON object")
            yield record


def iter_parquet(path: Path, columns: Sequence[str] | None) -> Iterator[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    wanted = None if columns is None else list(columns)
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=wanted):
        yield from batch.to_pylist()


def build_oasst_threads(records: Iterable[Row]) -> Iterator[dict[str, object]]:
    messages: dict[str, Row] = {}
    children: defaultdict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for record in records:
        identifier = record.get("message_id")
        if not isinstance(identifier, str) or record.get("deleted") is True:
            continue
        if record.get("lang") != "en" or record.get("review_result") is False:
            continue
        messages[identifier] = record
        parent = record.get("parent_id")
        if isinstance(parent, str):
            children[parent].append(identifier)
        else:
            roots.append(identifier)

    def best_child(identifier: str) -> str | None:
        options = [child for child in children.get(identifier, ()) if child in messages]
        if not options:
            return None

        def rank(child: str) -> tuple[int, str]:
            value = messages[child].get("rank")
            order = value if isinstance(value, int) and not isinstance(value, bool) else 1 << 30
            return order, child

        return min(options, key=rank)

    for root in roots:
        thread: list[dict[str, object]] = []
        current: str | None = root
        while current is not None:
            record = messages[current]
            thread.append({"role": record.get("role"), "content": record.get("text")})
            current = best_child(current)
        if thread and thread[-1]["role"] == "prompter":
            thread.pop()
        if thread:
            yield {"messages": thread}


def iter_rows(source: Source, path: Path) -> Iterator[Row]:
    if source.reader == "parquet":
        return iter_parquet(path, source.columns)
    if source.reader == "jsonl":
        return iter_jsonl(path)
    if source.reader == "oasst":
        return build_oasst_threads(iter_jsonl(path))
    raise FetchError(f"unknown reader {source.reader!r}")


class ChatWriter:
    def __init__(self, train_out: Path, valid_out: Path) -> None:
        self.train_out = train_out
        self.valid_out = valid_out
        self.stats = Stats()
        self.train_part = train_out.with_name(train_out.name + ".part")
        self.valid_part = valid_out.with_name(valid_out.name + ".part")
        self.train: IO[str] | None = None
        self.valid: IO[str] | None = None

    def __enter__(self) -> Self:
        self.train_out.parent.mkdir(parents=True, exist_ok=True)
        self.train = self.train_part.open("w", encoding="utf-8", newline="\n")
        self.valid = self.valid_part.open("w", encoding="utf-8", newline="\n")
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        for handle in (self.train, self.valid):
            if handle is not None:
                handle.close()
        if kind is None:
            self.train_part.replace(self.train_out)
            self.valid_part.replace(self.valid_out)
        else:
            self.train_part.unlink(missing_ok=True)
            self.valid_part.unlink(missing_ok=True)

    def add(self, messages: list[Message]) -> None:
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        split = "valid" if is_valid_key(first_user, CHAT_VALID_MODULUS) else "train"
        handle = self.valid if split == "valid" else self.train
        if handle is None:
            raise FetchError("writer is not open")
        handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        self.stats.kept[split] += 1
        self.stats.chars[split] += sum(len(m["content"]) for m in messages)


def output_paths(source: Source, out_dir: Path) -> tuple[Path, Path]:
    return out_dir / f"{source.name}_train.jsonl", out_dir / f"{source.name}_valid.jsonl"


def fetch_source(
    source: Source,
    *,
    files: int | None,
    download_dir: Path,
    out_dir: Path,
    lister: Lister = list_repo_files,
    rule: ChatRule | None = None,
) -> Stats:
    if files is not None and files < 1:
        raise FetchError("files must be at least 1")
    wanted = select_files(lister(source.repo), source.pattern)
    if not wanted:
        raise FetchError(f"{source.name}: no files in {source.repo} match {source.pattern}")
    if files is not None:
        wanted = wanted[:files]
    local: list[Path] = []
    for name in wanted:
        destination = download_dir / source.name / Path(name).name
        url = f"{HUB}/datasets/{source.repo}/resolve/main/{name}"
        fresh = download_retrying(url, destination)
        print(f"{'downloaded' if fresh else 'have'} {source.name}/{destination.name}", flush=True)
        local.append(destination)
    chosen = rule or ChatRule()
    train_out, valid_out = output_paths(source, out_dir)
    with ChatWriter(train_out, valid_out) as writer:
        for path in local:
            for row in iter_rows(source, path):
                messages, reason = source.adapter(row, chosen)
                if messages is None:
                    writer.stats.rejected[reason] += 1
                else:
                    writer.add(messages)
    return writer.stats


def report(name: str, stats: Stats) -> None:
    reasons = ", ".join(f"{k} {v}" for k, v in stats.rejected.most_common())
    print(
        f"{name}: train {stats.kept['train']} conversations, {stats.chars['train']} characters"
        f" | valid {stats.kept['valid']} conversations, {stats.chars['valid']} characters"
        f" | rejected {reasons or 'none'}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blom.chatfetch", description="Fetch chat training data")
    parser.add_argument(
        "--source",
        nargs="+",
        choices=(*SOURCE_NAMES, "all"),
        default=["all"],
        help="Which datasets to fetch",
    )
    parser.add_argument("--files", type=int, help="Only the first N files of each dataset")
    parser.add_argument(
        "--resume", action="store_true", help="Skip datasets whose output files already exist"
    )
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = SOURCE_NAMES if "all" in args.source else tuple(dict.fromkeys(args.source))
    try:
        for source in SOURCES:
            if source.name not in names:
                continue
            if args.resume and all(path.exists() for path in output_paths(source, args.out_dir)):
                print(f"have {source.name}", flush=True)
                continue
            stats = fetch_source(
                source, files=args.files, download_dir=args.download_dir, out_dir=args.out_dir
            )
            report(source.name, stats)
    except (FetchError, OSError, pa.ArrowException) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())