import json
import pytest

from data.temporal_sampling import (
    SAMPLER_REGISTRY,
    EdgeDenseSampler,
    FirstLastSampler,
    LegacyUniformSampler,
    TemporalSampler,
    UniformFixedSampler,
    build_temporal_sampler,
    register_sampler,
)

CONFIGS = [
    {"name": "legacy_uniform", "num_frames": 16},
    {"name": "uniform", "num_frames": 16},
    {"name": "uniform", "num_frames": 32},
    {"name": "preserve_all"},
    {"name": "fixed_fps", "target_fps": 15.0},
    {"name": "first_last", "frames_per_side": 8},
    {"name": "edge_dense", "num_frames": 16, "edge_strength": 1.0},
    {"name": "edge_dense", "num_frames": 16, "edge_strength": 0.5},
]


def test_registry_contains_all_samplers():
    assert set(SAMPLER_REGISTRY) == {
        "legacy_uniform", "uniform", "preserve_all",
        "fixed_fps", "first_last", "edge_dense",
    }


@pytest.mark.parametrize("cfg", CONFIGS)
def test_config_round_trip_and_json(cfg):
    sampler = build_temporal_sampler(cfg)
    assert isinstance(sampler, TemporalSampler)
    assert sampler.config() == cfg
    json.dumps(sampler.config())


@pytest.mark.parametrize("total", [5, 16, 17, 40, 101])
def test_legacy16_matches_old_formula(total):
    old = ([int(i * (total / 16.0)) for i in range(16)] if total >= 16
           else list(range(total)) + [total - 1] * (16 - total))
    assert LegacyUniformSampler(16).sample_indices(total) == old


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c["name"] != "legacy_uniform"])
def test_indices_sorted_unique_in_range(cfg):
    sampler = build_temporal_sampler(cfg)
    for total in range(1, 201):
        idx = sampler.sample_indices(total, fps=30.0)
        assert idx == sorted(set(idx)), (cfg, total)
        assert idx[0] == 0 and idx[-1] <= total - 1
        if cfg["name"] in ("uniform", "first_last", "edge_dense"):
            assert idx[-1] == total - 1


def test_expected_lengths():
    assert len(FirstLastSampler(8).sample_indices(100)) == 16
    assert len(UniformFixedSampler(32).sample_indices(100)) == 32
    assert len(EdgeDenseSampler(16).sample_indices(100)) == 16
    assert len(EdgeDenseSampler(16).sample_indices(10)) == 10  # nessun padding


def test_edge_dense_is_denser_at_edges():
    idx = EdgeDenseSampler(16, 1.0).sample_indices(150)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert gaps[0] < gaps[len(gaps) // 2] and gaps[-1] < gaps[len(gaps) // 2]


def test_edge_strength_zero_is_roughly_uniform():
    a = EdgeDenseSampler(16, 0.0).sample_indices(100)
    b = UniformFixedSampler(16).sample_indices(100)
    assert a == b


@pytest.mark.parametrize("bad", [
    {"name": "inesistente"},
    {"name": "uniform"},                          # manca num_frames
    {"name": "uniform", "num_frames": 8, "x": 1},  # parametro non ammesso
    {},
])
def test_invalid_configs_raise(bad):
    with pytest.raises(ValueError):
        build_temporal_sampler(bad)


def test_abc_cannot_be_instantiated_incomplete():
    with pytest.raises(TypeError):
        TemporalSampler()

    class SenzaConfig(TemporalSampler):
        def sample_indices(self, total_frames, fps=None):
            return [0]

    with pytest.raises(TypeError):
        SenzaConfig()


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        register_sampler("uniform")(UniformFixedSampler)