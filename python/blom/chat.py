import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

from blom.checkpoint import CheckpointError, load_checkpoint, restore
from blom.generate import GenerationError, generate
from blom.model import GPT
from blom.run import resolve_device
from blom.sample import END_OF_TEXT, IM_END, load_tokenizer
from blom.train import TrainError

IM_START = "<|im_start|>"
SPECIAL_TOKENS = (END_OF_TEXT, IM_START, IM_END)
ROLES = ("system", "user", "assistant")

Message = dict[str, str]


def render_prompt(messages: Sequence[Message]) -> str:
    """The same text as the Rust template with a generation prompt at the end."""
    if not messages:
        raise GenerationError("there are no messages to answer")
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ROLES or not isinstance(content, str):
            raise GenerationError(f"bad message {message!r}")
        if any(token in content for token in SPECIAL_TOKENS):
            raise GenerationError("a message contains a special token string")
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def fit_history(
    messages: list[Message], tokenizer: Tokenizer, *, limit: int
) -> tuple[list[Message], list[int]]:
    """Drop the oldest exchanges until the prompt fits in `limit` tokens."""
    kept = list(messages)
    while True:
        ids = tokenizer.encode(render_prompt(kept), add_special_tokens=False).ids
        if len(ids) <= limit:
            return kept, ids
        first = 1 if kept[0]["role"] == "system" else 0
        if len(kept) - first <= 1:
            raise GenerationError(f"the message needs {len(ids)} tokens, the limit is {limit}")
        del kept[first : first + 2]


def reply(
    model: GPT,
    tokenizer: Tokenizer,
    messages: list[Message],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seed: int | None,
    amp: bool,
) -> tuple[list[Message], str]:
    limit = model.config.max_seq - max_new_tokens - 1
    if limit < 1:
        raise GenerationError("max_new_tokens leaves no room for the prompt")
    kept, ids = fit_history(messages, tokenizer, limit=limit)
    stop = {
        token_id for name in SPECIAL_TOKENS if (token_id := tokenizer.token_to_id(name)) is not None
    }
    new = generate(
        model,
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_ids=stop,
        seed=seed,
        amp=amp,
    )
    return kept, tokenizer.decode(new, skip_special_tokens=False).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blom.chat", description="Chat with a fine-tuned model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--system", help="optional system message")
    parser.add_argument("--message", help="ask one question and exit")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50, help="0 turns top-k off")
    parser.add_argument("--top-p", type=float, default=0.9)
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
            raise GenerationError("chatting with a sketch-memory model is not supported")
        model = GPT(checkpoint.config).to(device)
        step = restore(checkpoint, model=model)
        print(f"[step {step} on {device}]", file=sys.stderr)
        history: list[Message] = []
        if args.system:
            history.append({"role": "system", "content": args.system})
        options = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k or None,
            "top_p": args.top_p,
            "seed": args.seed,
            "amp": amp,
        }
        if args.message is not None:
            history.append({"role": "user", "content": args.message})
            _, text = reply(model, tokenizer, history, **options)
            print(text)
            return 0
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            history.append({"role": "user", "content": line})
            history, text = reply(model, tokenizer, history, **options)
            history.append({"role": "assistant", "content": text})
            print(f"blom> {text}")
    except (CheckpointError, GenerationError, TrainError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())