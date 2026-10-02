import os
import torch


class FeatureLoader:

    @staticmethod
    def load(pt_file_path: str, expected_dim: int = 197) -> list[dict]:
        if not os.path.exists(pt_file_path):
            print(f"[WARNING] Feature file '{pt_file_path}' not found. Generating mock features...")
            os.makedirs(os.path.dirname(pt_file_path), exist_ok=True)

            num_samples = 12 if "train" in pt_file_path else 6
            mock_samples = []

            for i in range(num_samples):
                T_i = int(torch.randint(8, 41, (1,)).item())
                mock_samples.append({
                    "id": f"clip_{i:06d}",
                    "data": torch.randn(T_i, expected_dim),
                    "label": i % 9,
                    "num_frames": T_i,
                })

            mock_payload = {
                "format_version": 1,
                "backbone": "MockBackbone",
                "feature_dim": expected_dim,
                "num_samples": num_samples,
                "samples": mock_samples,
            }

            torch.save(mock_payload, pt_file_path)
            print(f"[INFO] Created mock dataset at '{pt_file_path}' ({num_samples} samples, D={expected_dim}).")

        payload = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        print(f"[INFO] Loaded {payload['num_samples']} samples (D={payload['feature_dim']}) from '{pt_file_path}'")
        return payload["samples"]