"""
Implementations of spatial and spatio-temporal backbone modules.
"""
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseBackbone


class X3DSBackbone(BaseBackbone):
    """
    X3D-S spatio-temporal feature extractor.

    The pretrained X3D classification head is removed. The remaining X3D trunk
    produces a 5D feature map which is pooled only over the spatial dimensions,
    preserving the temporal axis for the downstream Temporal Neck.

    Tensor Shapes:
        - Input `x`:  RGB video tensor of shape (B, C, T, H, W)
        - Output:     Feature sequence of shape (B, T, out_dim)

    Notes:
        - PyTorchVideo X3D-S has a native final trunk width of 192 channels.
        - The project proposal uses D=512 for X3D-S features, therefore a small
          linear projection is used by default to map 192 -> 512.
        - X3D-S does not downsample the temporal axis in its stem/residual stages,
          so the output temporal length is preserved (T_out = T_in).
        - `freeze_x3d=True` freezes only the pretrained X3D trunk. The optional
          output projection remains trainable, which allows adapting the native
          X3D feature width to the project feature dimension.

    Args:
        out_dim (int, optional): Output feature dimension. Default: 512.
        pretrained (bool, optional): Load Kinetics-pretrained X3D-S weights.
            Default: True.
        freeze_x3d (bool, optional): Freeze the X3D trunk parameters and keep its
            BatchNorm layers in evaluation mode. Default: True.
        progress (bool, optional): Show the checkpoint download progress bar when
            pretrained weights are requested. Default: True.
        x3d_model (nn.Module, optional): Pre-built X3D-compatible model. Intended
            for dependency injection in tests or custom checkpoint loading. The
            module must expose a `blocks` attribute whose final block is the
            original classification head.
    """

    NATIVE_OUT_DIM = 192

    def __init__(
        self,
        out_dim: int = 512,
        pretrained: bool = True,
        freeze_x3d: bool = True,
        progress: bool = True,
        x3d_model: Optional[nn.Module] = None,
    ):
        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got {out_dim}.")

        super().__init__(out_dim=out_dim)

        model = x3d_model or self._load_x3d_s(pretrained=pretrained, progress=progress)
        self._validate_x3d_model(model)

        # PyTorchVideo X3D is structured as:
        # Stem -> Stage1 -> Stage2 -> Stage3 -> Stage4 -> Classification Head.
        # The head performs spatio-temporal pooling and classification, therefore
        # it must be removed to preserve the temporal feature sequence.
        self.trunk = nn.Sequential(*list(model.blocks[:-1]))

        self.projection = (
            nn.Identity()
            if out_dim == self.NATIVE_OUT_DIM
            else nn.Linear(self.NATIVE_OUT_DIM, out_dim)
        )

        self._x3d_frozen = False
        if freeze_x3d:
            self.freeze_x3d()

    @staticmethod
    def _load_x3d_s(pretrained: bool, progress: bool) -> nn.Module:
        try:
            from pytorchvideo.models.hub import x3d_s
        except ImportError as exc:
            raise ImportError(
                "X3DSBackbone requires PyTorchVideo. Install it on the training VM "
                "with `pip install pytorchvideo`."
            ) from exc

        return x3d_s(pretrained=pretrained, progress=progress)

    @staticmethod
    def _validate_x3d_model(model: nn.Module) -> None:
        if not hasattr(model, "blocks"):
            raise TypeError("The supplied X3D model must expose a `blocks` attribute.")
        if len(model.blocks) < 2:
            raise ValueError(
                "The supplied X3D model must contain at least one feature block "
                "and one classification head."
            )

    def freeze_x3d(self) -> None:
        """Freeze the pretrained X3D trunk while leaving the projection trainable."""
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False
        self.trunk.eval()
        self._x3d_frozen = True

    def unfreeze_x3d(self) -> None:
        """Unfreeze the X3D trunk for optional end-to-end fine-tuning."""
        for parameter in self.trunk.parameters():
            parameter.requires_grad = True
        self._x3d_frozen = False
        self.trunk.train(self.training)

    def freeze(self) -> None:
        """Freeze the entire backbone, including the output projection."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        self._x3d_frozen = True
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze the entire backbone for end-to-end fine-tuning."""
        for parameter in self.parameters():
            parameter.requires_grad = True
        self._x3d_frozen = False
        self.train()

    def train(self, mode: bool = True):
        """
        Set train/eval mode while keeping a frozen X3D trunk in evaluation mode.

        Freezing parameters alone is not enough for BatchNorm-based backbones:
        their running statistics would otherwise continue to change when the full
        pipeline is switched to training mode.
        """
        super().train(mode)
        if self._x3d_frozen:
            self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                "X3DSBackbone expects a 5D video tensor with shape "
                f"(B, C, T, H, W), got {tuple(x.shape)}."
            )
        if x.size(1) != 3:
            raise ValueError(
                f"X3DSBackbone expects RGB input with C=3, got C={x.size(1)}."
            )

        features = self.trunk(x)  # (B, 192, T, H', W')

        if features.dim() != 5 or features.size(1) != self.NATIVE_OUT_DIM:
            raise RuntimeError(
                "Unexpected X3D-S trunk output shape. Expected channel dimension "
                f"{self.NATIVE_OUT_DIM}, got {tuple(features.shape)}."
            )

        # Spatial pooling only: preserve the temporal sequence for the neck.
        features = features.mean(dim=(-2, -1))  # (B, 192, T)
        features = features.transpose(1, 2).contiguous()  # (B, T, 192)

        return self.projection(features)  # (B, T, out_dim)


from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXt2DBackbone(BaseBackbone):
    """
    Used for 2D spatial feature extraction (based on ConvNeXt-Tiny)
    Produces tensors of dimention 768 for each frame of the input sequence
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(out_dim=768)
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        
        self.features = model.features
        self.avgpool = model.avgpool

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        feat = self.features(x_2d)       # (B*T, 768, H', W')
        feat = self.avgpool(feat)        # (B*T, 768, 1, 1)
        feat = torch.flatten(feat, 1)    # (B*T, 768)
        
        return feat.view(B, T, self.out_dim)  # (B, T, 768)


class Hybrid2D3DBackbone(BaseBackbone):
    """
    This wrapper combines the two previous backbone outputs (size 960)
    """
    def __init__(self, backbone_3d: nn.Module, backbone_2d: nn.Module):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.out_dim = 192 + 768

    def freeze(self) -> None:
        if hasattr(self.backbone_3d, "freeze"):
            self.backbone_3d.freeze()
        else:
            for p in self.backbone_3d.parameters(): p.requires_grad = False
            self.backbone_3d.eval()

        if hasattr(self.backbone_2d, "freeze"):
            self.backbone_2d.freeze()
        else:
            for p in self.backbone_2d.parameters(): p.requires_grad = False
            self.backbone_2d.eval()
            
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f3d = self.backbone_3d(x)  # (B, T, 192)
        f2d = self.backbone_2d(x)  # (B, T, 768)
        return torch.cat([f3d, f2d], dim=-1)  # (B, T, 960)


import torch
import torch.nn as nn
import numpy as np


class BallTrackerBackbone(nn.Module):
    """
    Estrattore di feature geometriche della palla per sequenze video.
    Utilizza la rete YOLOv8n (COCO) in inferenza pura via AutoBackend,
    prevenendo al 100% l'attivazione dell'engine Trainer.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        from ultralytics.nn.autobackend import AutoBackend

        # Carichiamo l'AutoBackend nativo: e' un modulo PyTorch puro senza logica di training
        self.device = torch.device(device)
        self.model = AutoBackend(model=model_path, device=self.device)
        self.model.eval()

        # Congeliamo tutti i parametri
        for param in self.model.parameters():
            param.requires_grad = False

        self.conf_thresh = confidence_threshold
        self.ball_class_id = 32  # ID per 'sports ball' in COCO
        self.out_dim = 5  # [x_norm, y_norm, dx, dy, confidence]

    def freeze(self) -> None:
        """Imposta il modello in modalità eval."""
        self.eval()
        self.model.eval()

    @torch.no_grad()
    def _track_single_clip(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processa una singola clip (C, T, H, W) con valori float in [0, 1].
        """
        C, T, H, W = clip_tensor.shape
        
        # Prepara il batch di frame direttamente come tensore PyTorch: (T, C, 640, 640)
        # Permuta da (C, T, H, W) -> (T, C, H, W)
        batch_frames = clip_tensor.permute(1, 0, 2, 3).to(self.device)

        # Rescale automatico a 640x640 per YOLO
        if H != 640 or W != 640:
            batch_frames = torch.nn.functional.interpolate(
                batch_frames, size=(640, 640), mode="bilinear", align_corners=False
            )

        # Passaggio forward diretto su AutoBackend (Niente Trainer, niente callback)
        preds = self.model(batch_frames)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        coords = []
        prev_x, prev_y = 0.5, 0.5  # Fallback al centro se non rilevata

        # Processa le predizioni per ciascuno dei T frame
        for t in range(T):
            pred_frame = preds[t]  # (84, 8400)
            
            boxes = pred_frame[:4, :].transpose(0, 1)  # (8400, 4) -> [cx, cy, w, h] in pixel
            scores = pred_frame[4 + self.ball_class_id, :]  # (8400,) score classe 32

            mask = scores >= self.conf_thresh

            if mask.any():
                valid_scores = scores[mask]
                valid_boxes = boxes[mask]
                
                best_idx = torch.argmax(valid_scores)
                best_conf = float(valid_scores[best_idx].cpu().item())
                best_box = valid_boxes[best_idx].cpu().numpy()

                # Normalizzazione coordinate in [0, 1]
                curr_x = float(best_box[0] / 640.0)
                curr_y = float(best_box[1] / 640.0)
                max_conf = best_conf
            else:
                curr_x, curr_y = prev_x, prev_y
                max_conf = 0.0

            dx = curr_x - prev_x
            dy = curr_y - prev_y

            coords.append([curr_x, curr_y, dx, dy, max_conf])
            prev_x, prev_y = curr_x, curr_y

        return torch.tensor(coords, dtype=torch.float32)  # (T, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W)
        B = x.shape[0]
        batch_coords = [self._track_single_clip(x[b]) for b in range(B)]
        return torch.stack(batch_coords, dim=0).to(x.device)
    

from typing import List


class CompositeBackbone(nn.Module):
    """
    Raccoglie una lista di istanze di backbone già create e
    concatena le loro feature lungo l'ultimo asse (D).
    """
    def __init__(self, backbones: List[BaseBackbone]):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        
        # Somma le dimensioni delle singole backbone
        self.out_dim = sum(
            getattr(b, "out_dim", getattr(b, "NATIVE_OUT_DIM", 0)) 
            for b in self.backbones
        )

    def freeze(self) -> None:
        """Mette tutte le backbone in eval mode e congela i parametri."""
        for b in self.backbones:
            if hasattr(b, "freeze"):
                b.freeze()
            else:
                for p in b.parameters():
                    p.requires_grad = False
                b.eval()
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        features = [b(x) for b in self.backbones]
        return torch.cat(features, dim=-1)  # (B, T, D_totale)

