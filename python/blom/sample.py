import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

from blom.checkpoint import CheckpointError, load_checkpoint, restore
from blom.generate import GenerationError, generate
from blom.model import GPT
from blom.run import resolve_device
from blom.train import TrainError

END_OF_TEXT = "<|endoftext|>"
IM_END = "<|im_end|>"


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.is_file():
        raise GenerationError(f"{path} is not a file")
    try:
        return Tokenizer.from_file(str(path))
    except Exception as error:
        raise GenerationError(f"cannot load tokenizer {path}: {error}") from error


def complete(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seed: int | None,
    amp: bool,
) -> str:
    stop_ids = {
        token_id
        for name in (END_OF_TEXT, IM_END)
        if (token_id := tokenizer.token_to_id(name)) is not None
    }
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        start = tokenizer.token_to_id(END_OF_TEXT)
        if start is None:
            raise GenerationError(f"an empty prompt needs {END_OF_TEXT} in the tokenizer")
        ids = [start]
    new = generate(
        model,
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_ids=stop_ids,
        seed=seed,
        amp=amp,
    )
    return prompt + tokenizer.decode(new, skip_special_tokens=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blom.sample", description="Generate text.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50, help="0 turns top-k off")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = resolve_device(args.device)
        amp = device.type == "cuda" if args.amp is None else args.amp
        tokenizer = load_tokenizer(args.tokenizer)
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)
        if tokenizer.get_vocab_size() != checkpoint.config.vocab_size:
            raise GenerationError(
                f"the tokenizer has {tokenizer.get_vocab_size()} tokens, "
                f"the model expects {checkpoint.config.vocab_size}"
            )
        if checkpoint.config.memory_dim:
            raise GenerationError(
                "this checkpoint uses sketch memory, and sampling with memory is not supported yet"
            )
        model = GPT(checkpoint.config).to(device)
        step = restore(checkpoint, model=model)
        print(f"[step {step} on {device}]", file=sys.stderr)
        text = complete(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p,
            seed=args.seed,
            amp=amp,
        )
    except (CheckpointError, GenerationError, TrainError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
