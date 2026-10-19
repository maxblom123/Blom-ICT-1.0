from dataclasses import replace

import pytest
import torch

from blom import GPT, ModelConfig
from blom.layers import KVCache

CONFIG = ModelConfig(
    vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2, d_ff=64, max_seq=16
)


@pytest.fixture(autouse=True)
def seeded() -> None:
    torch.manual_seed(0)


def tokens(batch: int, length: int) -> torch.Tensor:
    return torch.randint(0, 64, (batch, length))


def trained_like(config: ModelConfig = CONFIG) -> GPT:
    model = GPT(config)
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            torch.nn.init.normal_(parameter, std=0.2)
    return model.eval()


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
@pytest.mark.parametrize("qk_norm", [True, False])
@pytest.mark.parametrize("tie", [True, False])
def test_cached_logits_match_uncached_logits_for_every_split(
    n_kv_head: int, qk_norm: bool, tie: bool
) -> None:
    config = replace(CONFIG, n_kv_head=n_kv_head, qk_norm=qk_norm, tie_embeddings=tie)
    model = trained_like(config)
    ids = tokens(2, 12)
    with torch.no_grad():
        expected = model(ids)
        for sizes in ([12], [1] * 12, [5, 1, 1, 5], [3, 4, 5], [11, 1]):
            caches = model.new_caches()
            pieces = []
            start = 0
            for size in sizes:
                pieces.append(model(ids[:, start : start + size], caches))
                start += size
            assert torch.allclose(torch.cat(pieces, dim=1), expected, atol=1e-5), sizes


def test_each_cache_tracks_the_number_of_positions_it_holds() -> None:
    model = trained_like()
    caches = model.new_caches()
    assert len(caches) == CONFIG.n_layer
    with torch.no_grad():
        model(tokens(1, 5), caches)
        model(tokens(1, 3), caches)
    assert [cache.length for cache in caches] == [8, 8]


def test_resetting_a_cache_starts_a_fresh_sequence() -> None:
    model = trained_like()
    ids = tokens(1, 6)
    caches = model.new_caches()
    with torch.no_grad():
        model(tokens(1, 9), caches)
        for cache in caches:
            cache.reset()
        again = model(ids, caches)
        assert torch.allclose(again, model(ids), atol=1e-5)


def test_a_cache_refuses_to_overflow() -> None:
    model = trained_like()
    caches = model.new_caches()
    with torch.no_grad():
        model(tokens(1, CONFIG.max_seq), caches)
        with pytest.raises(ValueError, match="exceeds"):
            model(tokens(1, 1), caches)


def test_the_cache_capacity_is_checked_by_the_cache_itself() -> None:
    cache = KVCache(4)
    k = torch.zeros(1, 2, 3, 8)
    cache.append(k, k)
    with pytest.raises(ValueError, match="cannot add 3 after 3"):
        cache.append(k, k)


def test_a_cache_rejects_a_changed_batch_size() -> None:
    cache = KVCache(8)
    cache.append(torch.zeros(2, 2, 3, 8), torch.zeros(2, 2, 3, 8))
    with pytest.raises(ValueError, match="batch size"):
        cache.append(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8))


def test_a_cache_needs_a_positive_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        KVCache(0)


def test_the_wrong_number_of_caches_is_rejected() -> None:
    model = trained_like()
    with pytest.raises(ValueError, match="expected 2 caches"):
        model(tokens(1, 3), [KVCache(16)])


def test_a_cache_keeps_the_dtype_it_first_received() -> None:
    cache = KVCache(8)
    k = torch.zeros(1, 2, 2, 8, dtype=torch.bfloat16)
    keys, values = cache.append(k, k)
    assert keys.dtype == values.dtype == torch.bfloat16
    assert keys.shape == (1, 2, 2, 8)


def test_models_without_caches_behave_as_before() -> None:
    model = trained_like()
    ids = tokens(2, 7)
    with torch.no_grad():
        assert torch.equal(model(ids), model(ids, None))
