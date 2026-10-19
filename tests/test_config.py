from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from blom import ConfigError, ModelConfig

TINY = {
    "vocab_size": 100,
    "d_model": 64,
    "n_layer": 2,
    "n_head": 4,
    "n_kv_head": 2,
    "d_ff": 128,
    "max_seq": 32,
}


def make(**overrides: object) -> ModelConfig:
    return ModelConfig.from_dict({**TINY, **overrides})


def test_derived_sizes() -> None:
    config = make()
    assert config.head_dim == 16
    assert config.kv_dim == 32


def test_param_count_matches_hand_calculation() -> None:
    assert make().param_count == 80_512
    assert make(tie_embeddings=False).param_count == 86_912
    assert make(qk_norm=False).param_count == 80_448


def test_config_is_immutable() -> None:
    config = make()
    with pytest.raises(AttributeError):
        config.d_model = 128  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"d_model": 66}, "divisible by n_head"),
        ({"d_model": 20, "n_head": 4, "n_kv_head": 2}, "must be even"),
        ({"n_kv_head": 3}, "divisible by n_kv_head"),
        ({"n_layer": 0}, "must be positive"),
        ({"vocab_size": -5}, "must be positive"),
        ({"rope_theta": 0.0}, "must be positive"),
        ({"d_model": 64.0}, "must be int"),
        ({"qk_norm": 1}, "must be bool"),
        ({"d_model": True}, "must be int"),
    ],
)
def test_invalid_configs_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        make(**overrides)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown model config keys: layers"):
        make(layers=3)


def test_missing_keys_are_rejected() -> None:
    incomplete = {key: value for key, value in TINY.items() if key != "d_ff"}
    with pytest.raises(ConfigError, match="d_ff"):
        ModelConfig.from_dict(incomplete)


def test_load_reads_the_model_table(tmp_path: Path) -> None:
    path = tmp_path / "model.toml"
    path.write_text(
        "[model]\nvocab_size = 100\nd_model = 64\nn_layer = 2\nn_head = 4\n"
        "n_kv_head = 2\nd_ff = 128\nmax_seq = 32\nrope_theta = 500000\n",
        encoding="utf-8",
    )
    config = ModelConfig.load(path)
    assert config.rope_theta == 500_000
    assert config.n_layer == 2


def test_load_requires_a_model_table(tmp_path: Path) -> None:
    path = tmp_path / "model.toml"
    path.write_text("[other]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no \\[model\\] table"):
        ModelConfig.load(path)


@given(
    n_kv_head=st.integers(1, 4),
    group=st.integers(1, 4),
    half_head=st.integers(1, 16),
    n_layer=st.integers(1, 8),
    d_ff=st.integers(1, 512),
    vocab_size=st.integers(1, 1000),
    qk=st.booleans(),
)
def test_untied_head_adds_exactly_one_embedding_matrix(
    n_kv_head: int,
    group: int,
    half_head: int,
    n_layer: int,
    d_ff: int,
    vocab_size: int,
    qk: bool,
) -> None:
    n_head = n_kv_head * group
    d_model = n_head * half_head * 2

    def build(*, tie: bool) -> ModelConfig:
        return make(
            n_head=n_head,
            n_kv_head=n_kv_head,
            d_model=d_model,
            n_layer=n_layer,
            d_ff=d_ff,
            vocab_size=vocab_size,
            tie_embeddings=tie,
            qk_norm=qk,
        )

    tied = build(tie=True)
    assert tied.head_dim * tied.n_head == d_model
    assert tied.param_count > vocab_size * d_model
    assert build(tie=False).param_count - tied.param_count == vocab_size * d_model
