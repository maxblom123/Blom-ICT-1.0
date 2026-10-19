from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blom import GPT, ModelConfig
from blom.checkpoint import save_checkpoint
from blom.generate import GenerationError
from blom.optim import build_optimizer
from blom.sample import complete, load_tokenizer, main

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def make_tokenizer(specials: list[str] | None = None) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=SPECIALS if specials is None else specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["hello world, once upon a time. " * 40], trainer)
    return tokenizer


TOKENIZER = make_tokenizer()


def make_config(vocab_size: int | None = None) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size or TOKENIZER.get_vocab_size(),
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_ff=64,
        max_seq=64,
    )


def make_model(config: ModelConfig | None = None) -> GPT:
    torch.manual_seed(0)
    model = GPT(config or make_config())
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            torch.nn.init.normal_(parameter, std=0.2)
    return model


SETTINGS = {
    "max_new_tokens": 12,
    "temperature": 0.0,
    "top_k": None,
    "top_p": 1.0,
    "seed": None,
    "amp": False,
}


def write_files(tmp_path: Path, config: ModelConfig | None = None) -> list[str]:
    model = make_model(config)
    checkpoint = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, model=model, optimizer=build_optimizer(model, lr=1e-3), step=7)
    tokenizer = tmp_path / "tokenizer.json"
    TOKENIZER.save(str(tokenizer))
    return ["--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer), "--device", "cpu"]


def test_a_completion_starts_with_the_prompt() -> None:
    text = complete(make_model(), TOKENIZER, "hello wor", **SETTINGS)
    assert text.startswith("hello wor")
    assert len(text) > len("hello wor")


def test_a_seed_makes_sampled_completions_reproducible() -> None:
    sampled = {**SETTINGS, "temperature": 1.0, "seed": 5}
    model = make_model()
    assert complete(model, TOKENIZER, "once", **sampled) == complete(
        model, TOKENIZER, "once", **sampled
    )
    assert complete(model, TOKENIZER, "once", **sampled) != complete(
        model, TOKENIZER, "once", **{**sampled, "seed": 6}
    )


def test_an_empty_prompt_starts_from_the_end_of_text_token() -> None:
    text = complete(make_model(), TOKENIZER, "", **SETTINGS)
    assert text
    assert not text.startswith("<|endoftext|>")


def test_an_empty_prompt_needs_the_end_of_text_token() -> None:
    bare = make_tokenizer(specials=[])
    model = make_model(make_config(bare.get_vocab_size()))
    with pytest.raises(GenerationError, match="empty prompt"):
        complete(model, bare, "", **SETTINGS)


def test_generation_stops_at_the_end_of_text_token() -> None:
    config = replace(make_config(), tie_embeddings=False)
    model = make_model(config)
    torch.nn.init.zeros_(model.lm_head.weight)
    assert complete(model, TOKENIZER, "hello", **SETTINGS) == "hello"


def test_the_tokenizer_loader_reports_problems(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="not a file"):
        load_tokenizer(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(GenerationError, match="cannot load tokenizer"):
        load_tokenizer(broken)


def test_the_command_line_prints_the_completion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = write_files(tmp_path)
    code = main([*files, "--prompt", "hello", "--temperature", "0", "--max-new-tokens", "8"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("hello")
    assert "[step 7 on cpu]" in captured.err


def test_the_command_line_accepts_all_sampling_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = write_files(tmp_path)
    args = ["--prompt", "once", "--top-k", "0", "--top-p", "0.9", "--seed", "3"]
    assert main([*files, *args, "--max-new-tokens", "5"]) == 0
    first = capsys.readouterr().out
    assert main([*files, *args, "--max-new-tokens", "5"]) == 0
    assert capsys.readouterr().out == first


def test_the_command_line_reports_errors_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = write_files(tmp_path)
    missing = ["--checkpoint", str(tmp_path / "nope.pt"), *files[2:]]
    assert main(missing) == 1
    assert "not a file" in capsys.readouterr().err
    no_tokenizer = [*files[:2], "--tokenizer", str(tmp_path / "nope.json"), *files[4:]]
    assert main(no_tokenizer) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert main([*files, "--prompt", "hi", "--max-new-tokens", "0"]) == 1
    assert "max_new_tokens" in capsys.readouterr().err


def test_a_tokenizer_that_does_not_fit_the_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = write_files(tmp_path, make_config(vocab_size=TOKENIZER.get_vocab_size() + 5))
    assert main(files) == 1
    assert "the model expects" in capsys.readouterr().err


def test_an_unknown_device_is_a_usage_error(tmp_path: Path) -> None:
    files = write_files(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main([*files[:4], "--device", "tpu"])
    assert excinfo.value.code == 2
