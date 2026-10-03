"""
Implementations of spatial and spatio-temporal backbone modules.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

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

from typing import Optional
import torch
import torch.nn as nn

from .base import BaseBackbone


class X3DLBackbone(BaseBackbone):
    NATIVE_OUT_DIM = 432

    def __init__(
        self,
        out_dim: int = 432,
        pretrained: bool = True,
        freeze_x3d: bool = True,
        progress: bool = True,
        x3d_model: Optional[nn.Module] = None,
    ):
        super().__init__(out_dim=out_dim)

        model = x3d_model or self._load_x3d_l(
            pretrained=pretrained, progress=progress
        )
        
        # 1. Trunk (Stage 0 -> Stage 4)
        self.trunk = nn.Sequential(*list(model.blocks[:-1]))
        
        # 2. Extract the head sub-layers that project 192 -> 432
        head_pool = model.blocks[5].pool
        self.head_proj = nn.Sequential(
            head_pool.pre_conv,
            head_pool.pre_norm,
            head_pool.pre_act
        )

        self.projection = (
            nn.Identity()
            if out_dim == self.NATIVE_OUT_DIM
            else nn.Linear(self.NATIVE_OUT_DIM, out_dim)
        )

        if freeze_x3d:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, 16, 312, 312)
        x_feat = self.trunk(x)             # (B, 192, 16, 10, 10)
        x_feat = self.head_proj(x_feat)     # (B, 432, 16, 10, 10)
        
        # Spatial Average Pooling only over H' and W' -> preserves T=16
        x_feat = x_feat.mean(dim=(-2, -1))  # (B, 432, 16)
        x_feat = x_feat.permute(0, 2, 1).contiguous() # (B, 16, 432)

        return self.projection(x_feat)

    @staticmethod
    def _load_x3d_l(pretrained: bool, progress: bool) -> nn.Module:
        try:
            from pytorchvideo.models.hub import x3d_l
        except ImportError as exc:
            raise ImportError(
                "X3DLBackbone requires PyTorchVideo. Install it with `pip"
                " install pytorchvideo`."
            ) from exc

        return x3d_l(pretrained=pretrained, progress=progress)

    @staticmethod
    def _validate_x3d_model(model: nn.Module) -> None:
        if not hasattr(model, "blocks"):
            raise ValueError(
                "Provided x3d_model does not have a 'blocks' attribute."
            )

    def freeze_x3d(self) -> None:
        """Freeze the X3D-L trunk parameters and set BatchNorm layers to eval."""
        for param in self.trunk.parameters():
            param.requires_grad = False
        self.trunk.eval()
        self._x3d_frozen = True

    def unfreeze_x3d(self) -> None:
        """Unfreeze the X3D-L trunk parameters for fine-tuning."""
        for param in self.trunk.parameters():
            param.requires_grad = True
        self._x3d_frozen = False

    def freeze(self) -> None:
        """Freeze the entire backbone."""
        for param in self.parameters():
            param.requires_grad = False
        self._x3d_frozen = True
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze the entire backbone."""
        for param in self.parameters():
            param.requires_grad = True
        self._x3d_frozen = False
        self.train()

    def train(self, mode: bool = True) -> "X3DLBackbone":
        super().train(mode)
        if self._x3d_frozen:
            self.trunk.eval()
        return self


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
    Geometric ball feature extractor for video sequences.
    Uses the YOLOv8n (COCO) network in pure inference via AutoBackend,
    100% preventing the activation of the Trainer engine.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        from ultralytics.nn.autobackend import AutoBackend

        # Load native AutoBackend: it is a pure PyTorch module without training logic
        self.device = torch.device(device)
        self.model = AutoBackend(model=model_path, device=self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.conf_thresh = confidence_threshold
        self.ball_class_id = 32  # ID for 'sports ball' in COCO
        self.out_dim = 5  # [x_norm, y_norm, dx, dy, confidence]

    def freeze(self) -> None:
        self.eval()
        self.model.eval()

    @staticmethod
    def flip_ball_tracking_coords(coords: torch.Tensor) -> torch.Tensor:
        """
        Flips tracking coordinates if the Video Flip augmentation has been applied.
        """
        coords_flipped = coords.clone()
        coords_flipped[:, 0] = 1.0 - coords_flipped[:, 0]
        coords_flipped[:, 2] = -coords_flipped[:, 2]
        return coords_flipped

    @torch.no_grad()
    def _track_single_clip(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes a single clip (C, T, H, W) with float values in [0, 1].
        """
        C, T, H, W = clip_tensor.shape
        
        # Prepare the batch of frames directly as a PyTorch tensor: (T, C, 640, 640)
        # Permute from (C, T, H, W) -> (T, C, H, W)
        batch_frames = clip_tensor.permute(1, 0, 2, 3).to(self.device)

        # Automatic rescale to 640x640 for YOLO
        if H != 640 or W != 640:
            batch_frames = torch.nn.functional.interpolate(
                batch_frames, size=(640, 640), mode="bilinear", align_corners=False
            )

        # Direct forward pass on AutoBackend
        preds = self.model(batch_frames)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        coords = []
        prev_x, prev_y = 0.5, 0.5  # Fallback to center if not detected

        # Process predictions for each of the T frames
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

                # Normalization of coordinates to [0, 1]
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
    Combines a list of already created backbone instances
    and concatenates their features along the last axis (D).
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
        """Put all backbones in eval mode and freeze their parameters."""
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

    def forward_dual(self, x_norm: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        features = []
        for b in self.backbones:
            if isinstance(b, BallTrackerBackbone):
                features.append(b(x_raw))
            else:
                features.append(b(x_norm))
        return torch.cat(features, dim=-1)


class LSTAModule(nn.Module):
    """
    Long Short-Term Adaptive (LSTA) Module.
    Applies adaptive temporal context to features extracted from the bottleneck.
    """
    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # 1. Gated Convolution Layer
        self.gate_conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gate_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.gate_conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
        # 2. Short-Term Branch (Channel & Time)
        red_dim = max(1, channels // reduction)
        self.short_term = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, red_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(red_dim, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Long-Term Branch (Dynamic temporal kernels)
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, red_dim),
            nn.ReLU(inplace=True),
            nn.Linear(red_dim, channels * kernel_size)
        )

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # x shape: (B*T, C, H, W)
        _, C, H, W = x.shape
        
        # Global Spatial Average Pooling -> (B, C, T)
        x_5d = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        P = x_5d.mean(dim=(-2, -1))                          # (B, C, T)
        
        # Gated Convolution
        gated = self.gate_conv3(self.gate_conv1(P) * self.gate_conv2(P))
        G = P + gated
        
        # Short-Term Branch
        V = self.short_term(G)  # (B, C, 1)
        Z = G * V              # (B, C, T)
        
        # Long-Term Branch (Adaptive Temporal Depthwise Convolution)
        kernels = self.kernel_gen(Z)                      # (B, C * K)
        kernels = kernels.view(B, C, self.kernel_size)    # (B, C, K)
        kernels = F.softmax(kernels, dim=-1)
        
        pad = self.kernel_size // 2
        Z_padded = F.pad(Z, (pad, pad), mode="reflect")
        Z_unfold = Z_padded.unfold(dimension=-1, size=self.kernel_size, step=1)  # (B, C, T, K)
        Y_temp = (Z_unfold * kernels.unsqueeze(2)).sum(dim=-1)                   # (B, C, T)
        
        # Rescaling and spatial re-weighting
        scale = Y_temp.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        out = x_5d * scale                          # (B, C, T, H, W)
        return out.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


class TSCIModule(nn.Module):
    """
    Triplet Spatial-Channel Interaction (TSCI) Module.
    Models cross-dimensional Space-Channel interactions.
    """
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv_channel = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # x shape: (B*T, C, H, W)
        BT, C, H, W = x.shape
        
        # 1. Spatial Attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        s_pooled = torch.cat([avg_s, max_s], dim=1)  # (B*T, 2, H, W)
        att_s = self.conv_spatial(s_pooled)          # (B*T, 1, H, W)
        out_s = x * att_s
        
        # 2. Channel Attention
        x_5d = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x_c_pool = x_5d.mean(dim=(-2, -1))                   # (B, C, T)
        avg_c = x_c_pool.mean(dim=-1, keepdim=True)          # (B, C, 1)
        max_c, _ = x_c_pool.max(dim=-1, keepdim=True)        # (B, C, 1)
        c_pooled = torch.cat([avg_c, max_c], dim=2).transpose(1, 2)  # (B, 2, C)
        att_c = self.conv_channel(c_pooled).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        out_c = (x_5d * att_c).permute(0, 2, 1, 3, 4).reshape(BT, C, H, W)
        
        return (out_s + out_c + x) / 3.0


class ACABottleneck(nn.Module):
    """
    Wrapper of the ResNet Bottleneck with the injection of the ACA module (LSTA + TSCI).
    """
    def __init__(self, original_bottleneck: Bottleneck):
        super().__init__()
        self.conv1 = original_bottleneck.conv1
        self.bn1 = original_bottleneck.bn1
        self.relu = original_bottleneck.relu
        
        in_channels = original_bottleneck.conv1.out_channels
        self.lsta = LSTAModule(channels=in_channels)
        self.tsci = TSCIModule(channels=in_channels)
        
        self.conv2 = original_bottleneck.conv2
        self.bn2 = original_bottleneck.bn2
        self.conv3 = original_bottleneck.conv3
        self.bn3 = original_bottleneck.bn3
        self.downsample = original_bottleneck.downsample

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # ACA-Module applied after the first 1x1 Conv
        out = self.lsta(out, B, T)
        out = self.tsci(out, B, T)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ACANetBackbone(BaseBackbone):
    """
    ACA-Net Backbone for spatio-temporal feature extraction from video.
    
    Args:
        out_dim (int): Output channel dimension (Default: 512).
        pretrained (bool): Load ImageNet pre-trained weights for ResNet50.
        freeze_resnet (bool): Freeze the trunk parameters if set to True.
    """
    NATIVE_OUT_DIM = 2048

    def __init__(
        self,
        out_dim: int = 512,
        pretrained: bool = True,
        freeze_resnet: bool = False,
    ):
        super().__init__(out_dim=out_dim)
        
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)
        
        # Initial Stem
        self.stem = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        
        # Bottleneck layers converted to ACA-Blocks
        self.layer1 = self._wrap_layer(base_resnet.layer1)
        self.layer2 = self._wrap_layer(base_resnet.layer2)
        self.layer3 = self._wrap_layer(base_resnet.layer3)
        self.layer4 = self._wrap_layer(base_resnet.layer4)
        
        self.avgpool = base_resnet.avgpool
        
        # Optional projection if out_dim != 2048
        self.projection = (
            nn.Identity()
            if out_dim == self.NATIVE_OUT_DIM
            else nn.Linear(self.NATIVE_OUT_DIM, out_dim)
        )
        
        if freeze_resnet:
            self.freeze()

    def _wrap_layer(self, layer: nn.Sequential) -> nn.ModuleList:
        wrapped_blocks = nn.ModuleList()
        for block in layer:
            wrapped_blocks.append(ACABottleneck(block))
        return wrapped_blocks

    def freeze(self) -> None:
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"ACANetBackbone expects a 5D tensor (B, C, T, H, W), got {tuple(x.shape)}"
            )
        
        B, C, T, H, W = x.shape
        # Permutation for batch frame processing: (B*T, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        feat = self.stem(x_2d)
        
        # Pass-through in the layers with temporal interaction between frames
        for block in self.layer1: feat = block(feat, B, T)
        for block in self.layer2: feat = block(feat, B, T)
        for block in self.layer3: feat = block(feat, B, T)
        for block in self.layer4: feat = block(feat, B, T)
            
        feat = self.avgpool(feat)           # (B*T, 2048, 1, 1)
        feat = torch.flatten(feat, 1)       # (B*T, 2048)
        feat = feat.view(B, T, 2048)        # (B, T, 2048)
        
        return self.projection(feat)        # (B, T, out_dim)


class VideoSwinBackbone(BaseBackbone):
    """
    Video Swin Transformer backbone (Swin3D-Tiny).
    Extracts spatio-temporal features based on 3D self-attention mechanisms (Shifted Windows).
    
    Plug-and-play implementation: 
    Supports video input (B, C, T, H, W) and temporally realigns (via 1D interpolation) 
    the output features to return the same length T as the input (B, T, out_dim).
    """
    NATIVE_OUT_DIM = 768  # Output channel dimension from swin3d_t feature block

    def __init__(
        self,
        out_dim: int = 512,
        pretrained: bool = True,
        freeze_swin: bool = False,
    ):
        super().__init__(out_dim=out_dim)
        
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        model = swin3d_t(weights=weights)
        
        # Keep only the feature extractor and the normalization layer
        self.features = model.features
        self.norm = model.norm
        
        # Optional projection to align the requested out_dim (default 512)
        self.projection = (
            nn.Identity()
            if out_dim == self.NATIVE_OUT_DIM
            else nn.Linear(self.NATIVE_OUT_DIM, out_dim)
        )
        
        if freeze_swin:
            self.freeze()
            
    def freeze(self) -> None:
        """Freezes the backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
    def unfreeze(self) -> None:
        """Unfreezes the parameters for training."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"VideoSwinBackbone expects a 5D video tensor (B, C, T, H, W), got {tuple(x.shape)}."
            )
            
        B, C, T_in, H, W = x.shape
        
        # Forward pass through the transformer: time might be compressed
        # Output shape: (B, 768, T', H', W')
        feat = self.features(x)  
        feat = self.norm(feat)
        
        # Spatial Global Average Pooling
        # Average only over H' and W' keeping T' (B, 768, T')
        feat = feat.mean(dim=(-2, -1))
        
        # Video Swin usually downsamples the temporal axis (e.g. T/2).
        # To maintain 1:1 compatibility with the Neck (B, T, out_dim), we interpolate the features
        T_out = feat.shape[-1]
        if T_out != T_in:
            feat = F.interpolate(feat, size=T_in, mode="linear", align_corners=False)
            
        # Permute to (B, T, D) and final projection
        feat = feat.transpose(1, 2).contiguous() # (B, T_in, 768)
        
        return self.projection(feat) # (B, T_in, out_dim)
