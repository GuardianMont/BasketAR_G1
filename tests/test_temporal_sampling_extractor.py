import pytest
from data.experiment_config import ExperimentConfig
from data.temporal_sampling import FirstLastSampler, EdgeDenseSampler, UniformFixedSampler
from data.transforms import RandomTemporalShift
from scripts.extract_dataset_features import ManifestVideoDataset


def test_feature_id_edge_dense_label():
    cfg = ExperimentConfig(sampler={"name": "edge_dense", "num_frames": 16, "edge_strength": 0.5})
    assert "edgedense16s50" in cfg.feature_id()


def test_feature_ids_are_distinct():
    samplers = [
        {"name": "legacy_uniform", "num_frames": 16},
        {"name": "uniform", "num_frames": 16},
        {"name": "uniform", "num_frames": 32},
        {"name": "first_last", "frames_per_side": 8},
        {"name": "edge_dense", "num_frames": 16, "edge_strength": 1.0},
    ]
    ids = {ExperimentConfig(sampler=s).feature_id() for s in samplers}
    assert len(ids) == len(samplers)


@pytest.mark.parametrize("sampler", [FirstLastSampler(8), EdgeDenseSampler(16)])
def test_shift_rejected_with_endpoint_samplers(sampler):
    with pytest.raises(ValueError, match="RandomTemporalShift"):
        ManifestVideoDataset("nonexistent_root", "train",
                             temporal_sampler=sampler, transform=RandomTemporalShift())


def test_shift_allowed_with_uniform():
    # deve fallire solo sul manifest mancante, non sul guard
    with pytest.raises(FileNotFoundError):
        ManifestVideoDataset("nonexistent_root", "train",
                             temporal_sampler=UniformFixedSampler(16),
                             transform=RandomTemporalShift())