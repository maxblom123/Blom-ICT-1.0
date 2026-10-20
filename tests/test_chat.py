from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blom import GPT, ModelConfig
from blom.chat import fit_history, main, render_prompt, reply
from blom.checkpoint import save_checkpoint
from blom.generate import GenerationError
from blom.optim import build_optimizer

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["hello world, once upon a time. " * 40], trainer)
    return tokenizer


TOKENIZER = make_tokenizer()


def make_config(max_seq: int = 128) -> ModelConfig:
    return ModelConfig(
        vocab_size=TOKENIZER.get_vocab_size(),
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_ff=64,
        max_seq=max_seq,
    )


def make_model(max_seq: int = 128) -> GPT:
    torch.manual_seed(0)
    model = GPT(make_config(max_seq))
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            torch.nn.init.normal_(parameter, std=0.2)
    return model


SETTINGS = {
    "max_new_tokens": 10,
    "temperature": 0.0,
    "top_k": None,
    "top_p": 1.0,
    "seed": None,
    "amp": False,
}


def user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def test_the_prompt_uses_the_training_template() -> None:
    text = render_prompt([{"role": "system", "content": "Be brief."}, user("Hi")])
    assert text == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user"}],
        [user("bad <|im_end|> text")],
        [user("bad <|im_start|> text")],
        [user("bad <|endoftext|> text")],
    ],
)
def test_bad_messages_are_refused(messages: list[dict[str, str]]) -> None:
    with pytest.raises(GenerationError):
        render_prompt(messages)


def count(messages: list[dict[str, str]]) -> int:
    return len(TOKENIZER.encode(render_prompt(messages), add_special_tokens=False).ids)


def test_a_short_history_is_kept_whole() -> None:
    messages = [user("hello"), assistant("world"), user("again")]
    kept, ids = fit_history(messages, TOKENIZER, limit=1000)
    assert kept == messages
    assert len(ids) == count(messages)


def test_the_oldest_exchange_goes_first_and_the_system_message_stays() -> None:
    system = {"role": "system", "content": "Be brief."}
    messages = [system, user("first question"), assistant("first answer"), user("second")]
    limit = count([system, user("second")])
    kept, ids = fit_history(messages, TOKENIZER, limit=limit)
    assert kept == [system, user("second")]
    assert len(ids) <= limit


def test_a_message_that_cannot_fit_is_an_error() -> None:
    with pytest.raises(GenerationError, match="limit"):
        fit_history([user("hello world " * 20)], TOKENIZER, limit=10)


def test_a_reply_is_plain_text_and_repeatable() -> None:
    model = make_model()
    history = [user("hello world")]
    kept, text = reply(model, TOKENIZER, history, **SETTINGS)
    again_kept, again = reply(model, TOKENIZER, history, **SETTINGS)
    assert (kept, text) == (again_kept, again)
    assert kept == history
    assert not any(token in text for token in SPECIALS)


def test_a_reply_never_contains_special_tokens_even_when_sampling() -> None:
    model = make_model()
    options = {**SETTINGS, "temperature": 1.5, "max_new_tokens": 40}
    for seed in range(10):
        _, text = reply(model, TOKENIZER, [user("hello")], **{**options, "seed": seed})
        assert not any(token in text for token in SPECIALS)


def test_too_many_new_tokens_for_the_model_is_refused() -> None:
    model = make_model(max_seq=16)
    with pytest.raises(GenerationError, match="max_new_tokens"):
        reply(model, TOKENIZER, [user("hi")], **{**SETTINGS, "max_new_tokens": 16})


def test_old_turns_are_dropped_when_the_window_is_full() -> None:
    model = make_model(max_seq=64)
    history = [user("hello world " * 60), assistant("once upon a time"), user("hello")]
    assert count(history) > 64 - 8 - 1
    kept, _ = reply(model, TOKENIZER, history, **{**SETTINGS, "max_new_tokens": 8})
    assert kept[-1] == user("hello")
    assert len(kept) < len(history)


def write_files(tmp_path: Path) -> list[str]:
    model = make_model()
    checkpoint = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, model=model, optimizer=build_optimizer(model, lr=1e-3), step=3)
    tokenizer = tmp_path / "tokenizer.json"
    TOKENIZER.save(str(tokenizer))
    return ["--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer), "--device", "cpu"]


def test_one_message_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = write_files(tmp_path)
    code = main([*args, "--message", "hello world", "--temperature", "0", "--max-new-tokens", "8"])
    captured = capsys.readouterr()
    assert code == 0
    assert "[step 3 on cpu]" in captured.err
    assert not any(token in captured.out for token in SPECIALS)


def test_a_system_message_is_accepted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = write_files(tmp_path)
    code = main([*args, "--system", "Be brief.", "--message", "hi", "--max-new-tokens", "5"])
    assert code == 0
    assert capsys.readouterr().out.strip() != "error"


def test_a_conversation_keeps_its_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    args = write_files(tmp_path)
    lines = iter(["hello world", "", "once upon a time"])

    def fake_input(prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    code = main([*args, "--temperature", "0", "--max-new-tokens", "6"])
    assert code == 0
    assert capsys.readouterr().out.count("blom> ") == 2


def test_a_tokenizer_that_does_not_fit_the_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = write_files(tmp_path)
    small = Tokenizer(models.BPE())
    small.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    small.train_from_iterator(
        ["abc"],
        trainers.BpeTrainer(vocab_size=270, special_tokens=SPECIALS),
    )
    other = tmp_path / "other.json"
    small.save(str(other))
    args[args.index("--tokenizer") + 1] = str(other)
    assert main([*args, "--message", "hi"]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_missing_checkpoint_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    TOKENIZER.save(str(tokenizer))
    code = main(
        [
            "--checkpoint",
            str(tmp_path / "nope.pt"),
            "--tokenizer",
            str(tokenizer),
            "--device",
            "cpu",
            "--message",
            "hi",
        ]
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err