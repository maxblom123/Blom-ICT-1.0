import gzip
import http.client
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from blom import chatfetch
from blom.chatfetch import ChatRule, Source
from blom.fetch import SPECIAL_TOKENS, FetchError

RULE = ChatRule()
LONG = "This is a reasonably long message with enough characters. "


def turn(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def pair(user: str = LONG, assistant: str = LONG) -> list[dict[str, str]]:
    return [turn("user", user), turn("assistant", assistant)]


def prompt_for(split: str) -> str:
    for i in range(2000):
        text = (f"question number {i} " + LONG).strip()
        if chatfetch.is_valid_key(text, chatfetch.CHAT_VALID_MODULUS) == (split == "valid"):
            return text
    raise AssertionError


def test_keeps_a_plain_exchange() -> None:
    messages, reason = chatfetch.clean_messages(pair(), RULE)
    assert reason == "kept"
    assert messages == pair(LONG.strip(), LONG.strip())


def test_keeps_a_system_message_and_several_turns() -> None:
    raw = [turn("system", "Be brief and kind."), *pair(), *pair()]
    messages, reason = chatfetch.clean_messages(raw, RULE)
    assert reason == "kept"
    assert messages is not None
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant"]


def test_maps_sharegpt_roles() -> None:
    raw = [{"from": "human", "value": LONG}, {"from": "gpt", "value": LONG}]
    messages, reason = chatfetch.clean_messages(raw, RULE, role_key="from", content_key="value")
    assert reason == "kept"
    assert messages is not None
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_normalises_line_endings_and_strips() -> None:
    raw = pair("  hello\r\nthere " + LONG + "  ", LONG)
    messages, _ = chatfetch.clean_messages(raw, RULE)
    assert messages is not None
    assert messages[0]["content"].startswith("hello\nthere")
    assert not messages[0]["content"].endswith(" ")


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        pytest.param(None, "malformed", id="none"),
        pytest.param([], "malformed", id="empty-list"),
        pytest.param("text", "malformed", id="string"),
        pytest.param(["x"], "malformed", id="not-dict"),
        pytest.param([turn("user", LONG), {"role": "assistant"}], "malformed", id="no-content"),
        pytest.param([turn("user", LONG), turn("tool", LONG)], "malformed", id="unknown-role"),
        pytest.param([turn("user", LONG), turn("assistant", "   ")], "empty", id="blank"),
        pytest.param([turn("user", LONG)], "roles", id="no-reply"),
        pytest.param([turn("assistant", LONG), turn("user", LONG)], "roles", id="wrong-order"),
        pytest.param([*pair(), turn("user", LONG)], "roles", id="ends-with-user"),
        pytest.param([turn("user", LONG), turn("user", LONG)], "roles", id="two-users"),
        pytest.param([turn("system", LONG)], "roles", id="only-system"),
        pytest.param([*pair()[:1], turn("system", LONG), *pair()[1:]], "roles", id="late-system"),
        pytest.param(pair("hi", "ok"), "short", id="short"),
        pytest.param(pair("é" * 60, "é" * 60), "language", id="non-ascii"),
    ],
)
def test_rejects_bad_conversations(raw: object, reason: str) -> None:
    messages, got = chatfetch.clean_messages(raw, RULE)
    assert messages is None
    assert got == reason


@pytest.mark.parametrize("token", SPECIAL_TOKENS)
def test_rejects_special_tokens(token: str) -> None:
    messages, reason = chatfetch.clean_messages(pair(LONG + token, LONG), RULE)
    assert messages is None
    assert reason == "special_token"


def test_rejects_too_long() -> None:
    messages, reason = chatfetch.clean_messages(pair("a" * 40, "b" * 40), ChatRule(max_chars=50))
    assert messages is None
    assert reason == "long"


def test_rejects_unencodable_text() -> None:
    messages, reason = chatfetch.clean_messages(
        pair(LONG + "\ud800", LONG), ChatRule(max_non_ascii=1.0)
    )
    assert messages is None
    assert reason == "unencodable"


def test_allows_a_little_non_ascii() -> None:
    text = LONG * 4 + "café"
    messages, reason = chatfetch.clean_messages(pair(text, LONG), RULE)
    assert reason == "kept"
    assert messages is not None


def test_non_ascii_fraction() -> None:
    assert chatfetch.non_ascii_fraction("abc") == 0.0
    assert chatfetch.non_ascii_fraction("ééab") == 0.5


def test_adapt_messages_reads_the_messages_column() -> None:
    messages, reason = chatfetch.adapt_messages({"messages": pair()}, RULE)
    assert reason == "kept"
    assert messages is not None
    assert chatfetch.adapt_messages({}, RULE) == (None, "malformed")


def test_pair_adapter_builds_one_exchange() -> None:
    adapt = chatfetch.pair_adapter("query", "answer")
    messages, reason = adapt({"query": LONG, "answer": LONG + "!"}, RULE)
    assert reason == "kept"
    assert messages is not None
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert adapt({"query": LONG}, RULE) == (None, "malformed")
    assert adapt({"query": 1, "answer": 2}, RULE) == (None, "malformed")


def opencode_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {"input": LONG, "output": LONG, "average_test_score": 1.0}
    row.update(overrides)
    return row


def test_opencode_needs_a_high_test_score() -> None:
    assert chatfetch.adapt_opencode(opencode_row(), RULE)[1] == "kept"
    assert chatfetch.adapt_opencode(opencode_row(average_test_score=0.9), RULE)[1] == "kept"
    assert chatfetch.adapt_opencode(opencode_row(average_test_score=0.5), RULE)[1] == "low_score"
    assert chatfetch.adapt_opencode(opencode_row(average_test_score=None), RULE)[1] == "unscored"
    assert chatfetch.adapt_opencode(opencode_row(average_test_score=True), RULE)[1] == "unscored"


def magpie_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "language": "EN",
        "input_quality": "good",
        "llama_guard_2": "safe",
        "repeat_count": 0,
        "conversations": [{"from": "human", "value": LONG}, {"from": "gpt", "value": LONG}],
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        pytest.param({}, "kept", id="good"),
        pytest.param({"input_quality": "excellent"}, "kept", id="excellent"),
        pytest.param({"language": "DA"}, "language", id="language"),
        pytest.param({"language": None}, "language", id="no-language"),
        pytest.param({"input_quality": "average"}, "quality", id="quality"),
        pytest.param({"llama_guard_2": "unsafe\nS5"}, "unsafe", id="unsafe"),
        pytest.param({"repeat_count": 2}, "repeated", id="repeated"),
    ],
)
def test_magpie_filters(overrides: dict[str, object], reason: str) -> None:
    assert chatfetch.adapt_magpie(magpie_row(**overrides), RULE)[1] == reason


def oasst_message(
    identifier: str,
    parent: str | None,
    role: str,
    text: str,
    *,
    rank: int | None = None,
    lang: str = "en",
    deleted: bool = False,
    reviewed: bool = True,
) -> dict[str, object]:
    return {
        "message_id": identifier,
        "parent_id": parent,
        "role": role,
        "text": text,
        "rank": rank,
        "lang": lang,
        "deleted": deleted,
        "review_result": reviewed,
    }


def contents(thread: dict[str, object]) -> list[object]:
    messages = thread["messages"]
    assert isinstance(messages, list)
    return [message["content"] for message in messages]


def test_oasst_follows_the_best_ranked_replies() -> None:
    records = [
        oasst_message("r", None, "prompter", "question"),
        oasst_message("a1", "r", "assistant", "second best", rank=1),
        oasst_message("a0", "r", "assistant", "best", rank=0),
        oasst_message("p1", "a0", "prompter", "follow up", rank=0),
        oasst_message("a2", "p1", "assistant", "final", rank=0),
    ]
    threads = list(chatfetch.build_oasst_threads(records))
    assert len(threads) == 1
    assert contents(threads[0]) == ["question", "best", "follow up", "final"]


def test_oasst_drops_a_trailing_prompter_message() -> None:
    records = [
        oasst_message("r", None, "prompter", "question"),
        oasst_message("a", "r", "assistant", "answer", rank=0),
        oasst_message("p", "a", "prompter", "unanswered", rank=0),
    ]
    thread = next(chatfetch.build_oasst_threads(records))
    assert contents(thread) == ["question", "answer"]


def test_oasst_skips_other_languages_deleted_and_unreviewed() -> None:
    records = [
        oasst_message("r1", None, "prompter", "hola", lang="es"),
        oasst_message("r2", None, "prompter", "gone", deleted=True),
        oasst_message("r3", None, "prompter", "bad", reviewed=False),
        oasst_message("r4", None, "prompter", "lonely"),
    ]
    assert list(chatfetch.build_oasst_threads(records)) == []


def test_oasst_ranks_missing_ranks_last_and_is_deterministic() -> None:
    records = [
        oasst_message("r", None, "prompter", "q"),
        oasst_message("b", "r", "assistant", "unranked"),
        oasst_message("a", "r", "assistant", "ranked", rank=3),
    ]
    thread = next(chatfetch.build_oasst_threads(records))
    assert contents(thread) == ["q", "ranked"]


def test_select_files_filters_and_sorts() -> None:
    names = ["data/train-2.parquet", "README.md", "data/train-1.parquet", "data/test.parquet"]
    assert chatfetch.select_files(names, r"^data/train-.*\.parquet$") == [
        "data/train-1.parquet",
        "data/train-2.parquet",
    ]


def test_registry_is_consistent() -> None:
    assert len(set(chatfetch.SOURCE_NAMES)) == len(chatfetch.SOURCES)
    for source in chatfetch.SOURCES:
        assert source.reader in {"parquet", "jsonl", "oasst"}
        assert source.repo.count("/") == 1


def test_iter_jsonl_reads_plain_and_gzip(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2}]
    plain = tmp_path / "x.jsonl"
    plain.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    zipped = tmp_path / "x.jsonl.gz"
    with gzip.open(zipped, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(r) for r in rows) + "\n")
    assert list(chatfetch.iter_jsonl(plain)) == rows
    assert list(chatfetch.iter_jsonl(zipped)) == rows


def test_iter_jsonl_reports_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(FetchError, match="line 2"):
        list(chatfetch.iter_jsonl(path))
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(FetchError, match="not a JSON object"):
        list(chatfetch.iter_jsonl(path))


def test_iter_parquet_reads_only_requested_columns(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"], "c": [0.5, 0.25]}), path)
    assert list(chatfetch.iter_parquet(path, ("a", "b"))) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert next(iter(chatfetch.iter_parquet(path, None))) == {"a": 1, "b": "x", "c": 0.5}


def test_writer_splits_by_the_first_user_message(tmp_path: Path) -> None:
    train_out = tmp_path / "t.jsonl"
    valid_out = tmp_path / "v.jsonl"
    with chatfetch.ChatWriter(train_out, valid_out) as writer:
        writer.add([turn("system", "sys"), *pair(prompt_for("train"), "a")])
        writer.add(pair(prompt_for("valid"), "b"))
    assert writer.stats.kept["train"] == 1
    assert writer.stats.kept["valid"] == 1
    train = [json.loads(line) for line in train_out.read_text(encoding="utf-8").splitlines()]
    valid = [json.loads(line) for line in valid_out.read_text(encoding="utf-8").splitlines()]
    assert train[0]["messages"][0]["role"] == "system"
    assert valid[0]["messages"][1]["content"] == "b"
    assert not train_out.with_name("t.jsonl.part").exists()


def fail_inside_writer(train_out: Path, valid_out: Path) -> None:
    with chatfetch.ChatWriter(train_out, valid_out) as writer:
        writer.add(pair(prompt_for("train"), "a"))
        raise RuntimeError("boom")


def test_writer_removes_partial_files_on_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        fail_inside_writer(tmp_path / "t.jsonl", tmp_path / "v.jsonl")
    assert list(tmp_path.iterdir()) == []


def fake_source(path_pattern: str = r"^data/part-.*\.parquet$") -> Source:
    return Source(
        name="fake",
        repo="owner/fake",
        pattern=path_pattern,
        reader="parquet",
        adapter=chatfetch.adapt_messages,
        columns=("messages",),
    )


def write_fake_part(path: Path, conversations: list[list[dict[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"messages": conversations, "extra": ["x"] * len(conversations)}), path)


def test_fetch_source_end_to_end(tmp_path: Path) -> None:
    downloads = tmp_path / "dl"
    out = tmp_path / "out"
    good_train = pair(prompt_for("train"), LONG)
    good_valid = pair(prompt_for("valid"), LONG)
    bad = [turn("user", LONG)]
    write_fake_part(downloads / "fake" / "part-1.parquet", [good_train, bad])
    write_fake_part(downloads / "fake" / "part-2.parquet", [good_valid])
    names = ["data/part-1.parquet", "data/part-2.parquet", "README.md"]
    stats = chatfetch.fetch_source(
        fake_source(), files=None, download_dir=downloads, out_dir=out, lister=lambda _: names
    )
    assert stats.kept["train"] == 1
    assert stats.kept["valid"] == 1
    assert stats.rejected == {"roles": 1}
    assert (out / "fake_train.jsonl").exists()
    assert (out / "fake_valid.jsonl").exists()


def test_fetch_source_limits_files(tmp_path: Path) -> None:
    downloads = tmp_path / "dl"
    write_fake_part(downloads / "fake" / "part-1.parquet", [pair(prompt_for("train"), LONG)])
    write_fake_part(downloads / "fake" / "part-2.parquet", [pair(prompt_for("valid"), LONG)])
    names = ["data/part-1.parquet", "data/part-2.parquet"]
    stats = chatfetch.fetch_source(
        fake_source(),
        files=1,
        download_dir=downloads,
        out_dir=tmp_path / "out",
        lister=lambda _: names,
    )
    assert stats.kept["train"] == 1
    assert stats.kept["valid"] == 0


def test_fetch_source_errors(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="no files"):
        chatfetch.fetch_source(
            fake_source(),
            files=None,
            download_dir=tmp_path,
            out_dir=tmp_path,
            lister=lambda _: ["README.md"],
        )
    with pytest.raises(FetchError, match="at least 1"):
        chatfetch.fetch_source(
            fake_source(),
            files=0,
            download_dir=tmp_path,
            out_dir=tmp_path,
            lister=lambda _: ["data/part-1.parquet"],
        )


def test_fetch_source_reads_jsonl_sources(tmp_path: Path) -> None:
    downloads = tmp_path / "dl" / "code"
    downloads.mkdir(parents=True)
    rows = [{"query": prompt_for("train"), "answer": LONG}, {"query": "x", "answer": "y"}]
    (downloads / "data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    source = Source(
        name="code",
        repo="owner/code",
        pattern=r"^data\.jsonl$",
        reader="jsonl",
        adapter=chatfetch.pair_adapter("query", "answer"),
    )
    stats = chatfetch.fetch_source(
        source,
        files=None,
        download_dir=tmp_path / "dl",
        out_dir=tmp_path / "out",
        lister=lambda _: ["data.jsonl"],
    )
    assert stats.kept["train"] == 1
    assert stats.rejected == {"short": 1}


def test_fetch_source_reads_oasst_trees(tmp_path: Path) -> None:
    folder = tmp_path / "dl" / "oasst"
    folder.mkdir(parents=True)
    records = [
        oasst_message("r", None, "prompter", prompt_for("train")),
        oasst_message("a", "r", "assistant", LONG, rank=0),
    ]
    with gzip.open(folder / "m.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(r) for r in records) + "\n")
    source = Source(
        name="oasst",
        repo="owner/oasst",
        pattern=r"^m\.jsonl\.gz$",
        reader="oasst",
        adapter=chatfetch.adapt_messages,
    )
    stats = chatfetch.fetch_source(
        source,
        files=None,
        download_dir=tmp_path / "dl",
        out_dir=tmp_path / "out",
        lister=lambda _: ["m.jsonl.gz"],
    )
    assert stats.kept["train"] == 1


def test_unknown_reader_is_an_error(tmp_path: Path) -> None:
    source = Source(
        name="x", repo="a/b", pattern="x", reader="csv", adapter=chatfetch.adapt_messages
    )
    with pytest.raises(FetchError, match="unknown reader"):
        chatfetch.iter_rows(source, tmp_path / "x")


def test_parser_defaults_and_choices() -> None:
    args = chatfetch.build_parser().parse_args(["--download-dir", "d", "--out-dir", "o"])
    assert args.source == ["all"]
    assert args.files is None
    with pytest.raises(SystemExit):
        chatfetch.build_parser().parse_args(
            ["--source", "nope", "--download-dir", "d", "--out-dir", "o"]
        )


def test_main_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = chatfetch.main(
        [
            "--source",
            "smoltalk",
            "--files",
            "0",
            "--download-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err


def flaky_download(failures: int, error: Exception) -> tuple[list[int], object]:
    calls: list[int] = []

    def attempt(url: str, destination: Path) -> bool:
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return True

    return calls, attempt


def test_download_retrying_recovers_from_transient_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, attempt = flaky_download(2, FetchError("got 10 bytes, expected 20"))
    monkeypatch.setattr(chatfetch, "download", attempt)
    pauses: list[float] = []
    assert chatfetch.download_retrying("u", tmp_path / "f", pause=pauses.append) is True
    assert len(calls) == 3
    assert pauses == [2.0, 4.0]


def test_download_retrying_recovers_from_http_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, attempt = flaky_download(1, http.client.IncompleteRead(b"x"))
    monkeypatch.setattr(chatfetch, "download", attempt)
    assert chatfetch.download_retrying("u", tmp_path / "f", pause=lambda _: None) is True
    assert len(calls) == 2


def test_download_retrying_gives_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls, attempt = flaky_download(99, OSError("reset"))
    monkeypatch.setattr(chatfetch, "download", attempt)
    with pytest.raises(OSError, match="reset"):
        chatfetch.download_retrying("u", tmp_path / "f", attempts=3, pause=lambda _: None)
    assert len(calls) == 3


def test_download_retrying_checks_attempts(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="at least 1"):
        chatfetch.download_retrying("u", tmp_path / "f", attempts=0)


def test_output_paths_are_named_after_the_source(tmp_path: Path) -> None:
    train, valid = chatfetch.output_paths(fake_source(), tmp_path)
    assert train == tmp_path / "fake_train.jsonl"
    assert valid == tmp_path / "fake_valid.jsonl"


def test_resume_skips_finished_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not fetch a finished source")

    monkeypatch.setattr(chatfetch, "fetch_source", refuse)
    for source in chatfetch.SOURCES:
        for path in chatfetch.output_paths(source, tmp_path):
            path.write_text("", encoding="utf-8")
    code = chatfetch.main(["--resume", "--download-dir", str(tmp_path), "--out-dir", str(tmp_path)])
    assert code == 0
    output = capsys.readouterr().out
    for name in chatfetch.SOURCE_NAMES:
        assert f"have {name}" in output


def test_resume_still_fetches_unfinished_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []

    def record_fetch(source: Source, **kwargs: object) -> chatfetch.Stats:
        fetched.append(source.name)
        return chatfetch.Stats()

    monkeypatch.setattr(chatfetch, "fetch_source", record_fetch)
    train, _ = chatfetch.output_paths(chatfetch.SOURCES[0], tmp_path)
    train.write_text("", encoding="utf-8")
    args = ["--source", "smoltalk", "--download-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    assert chatfetch.main(["--resume", *args]) == 0
    assert fetched == ["smoltalk"]
    assert chatfetch.main(args) == 0
    assert fetched == ["smoltalk", "smoltalk"]