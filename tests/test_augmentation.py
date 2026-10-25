import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import pandas as pd
import torch

# Import dei moduli sotto test
from scripts.extract_dataset_features import (
    ClassAwareVideoTransform,
    ManifestVideoDataset,
    LABEL_TO_ID,
)


class TestDatasetAndAugmentation(unittest.TestCase):

    def setUp(self):
        """Prepara un ambiente temporaneo con manifest.csv e file video fittizi."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Manifest simulato: 7 campioni "idle", 3 campioni "tiroDaDue0"
        self.manifest_data = {
            "clip_id": [f"clip_{i:03d}" for i in range(10)],
            "path": [f"videos/clip_{i:03d}.mp4" for i in range(10)],
            "split": ["train"] * 10,
            "label": [
                "idle", "idle", "tiroDaDue0", "idle", "idle",
                "tiroDaDue0", "idle", "idle", "tiroDaDue0", "idle"
            ],
        }
        self.df_manifest = pd.DataFrame(self.manifest_data)
        self.df_manifest.to_csv(self.root_path / "manifest.csv", index=False)

        # Creazione dei file video fittizi per superare path.exists()
        for rel_path in self.manifest_data["path"]:
            video_file = self.root_path / rel_path
            video_file.parent.mkdir(parents=True, exist_ok=True)
            video_file.touch()

        self.minority_class_id = LABEL_TO_ID["tiroDaDue0"]
        self.majority_class_id = LABEL_TO_ID["idle"]

    def tearDown(self):
        """Pulisce la directory temporanea."""
        self.temp_dir.cleanup()

    def test_class_aware_transform_filtering(self):
        """Verifica che le trasformazioni si attivino solo sulle classi target."""
        transform = ClassAwareVideoTransform(
            transform_names=["color_jitter", "horizontal_flip"],
            minority_classes={self.minority_class_id},
        )

        self.assertTrue(transform.should_apply(self.minority_class_id))
        self.assertFalse(transform.should_apply(self.majority_class_id))

    def test_spatial_transforms_tensor_shape(self):
        """Verifica che la trasformazione gestisca tensori 4D (C, T, H, W) senza errori di layout."""
        transform = ClassAwareVideoTransform(
            transform_names=["color_jitter", "gaussian_blur"],
            minority_classes={self.minority_class_id},
        )
        
        # Tensore mock: Channels=3, Time=16, Height=224, Width=224
        dummy_clip_norm = torch.randn(3, 16, 224, 224)
        dummy_clip_raw = torch.randn(3, 16, 224, 224)

        out_norm, out_raw, is_flipped = transform.apply_spatial_transforms(
            dummy_clip_norm, dummy_clip_raw, label_id=self.minority_class_id
        )

        self.assertEqual(out_norm.shape, torch.Size([3, 16, 224, 224]))
        self.assertEqual(out_raw.shape, torch.Size([3, 16, 224, 224]))
        self.assertIsInstance(is_flipped, bool)

    def test_oversample_factor_dataset_size(self):
        """Verifica che la dimensione del dataset aumenti con l'oversampling."""
        transform = ClassAwareVideoTransform(
            transform_names=["color_jitter"],
            minority_classes={self.minority_class_id},
        )

        ds_base = ManifestVideoDataset(
            root=self.root_path,
            split="train",
            transform=transform,
            oversample_factor=1,
        )
        self.assertEqual(len(ds_base), 10)

        oversample_factor = 3
        ds_oversampled = ManifestVideoDataset(
            root=self.root_path,
            split="train",
            transform=transform,
            oversample_factor=oversample_factor,
        )

        expected_size = 7 + (3 * oversample_factor)
        self.assertEqual(len(ds_oversampled), expected_size)

    @patch.object(ManifestVideoDataset, "_decode_clip")
    def test_unique_clip_ids_and_indexing(self, mock_decode_clip):
        """Verifica l'indicizzazione e l'assegnazione di clip_id univoci durante l'oversampling."""
        dummy_clip = torch.zeros(3, 16, 224, 224)
        mock_decode_clip.return_value = (dummy_clip, dummy_clip, False)

        transform = ClassAwareVideoTransform(
            transform_names=["horizontal_flip"],
            minority_classes={self.minority_class_id},
        )

        ds = ManifestVideoDataset(
            root=self.root_path,
            split="train",
            transform=transform,
            oversample_factor=2,
        )

        extracted_ids = [ds[i]["sample_id"] for i in range(len(ds))]

        self.assertIn("clip_002_aug0", extracted_ids)
        self.assertIn("clip_002_aug1", extracted_ids)
        self.assertIn("clip_000", extracted_ids)
        self.assertNotIn("clip_000_aug0", extracted_ids)


if __name__ == "__main__":
    unittest.main()