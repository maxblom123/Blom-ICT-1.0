from pathlib import Path

import pytest

from blom.config import ModelConfig
from blom.train_config import TrainConfig

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.toml"))


def test_there_are_configs_to_check() -> None:
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_every_committed_config_is_valid_and_consistent(path: Path) -> None:
    model = ModelConfig.load(path)
    train = TrainConfig.load(path)
    assert train.seq_len <= model.max_seq
    assert train.total_steps >= train.warmup_steps + train.decay_steps
