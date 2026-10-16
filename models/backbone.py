"""
Implementations of spatial and spatio-temporal backbone modules.
"""
import os
from typing import Optional, Dict, Type, List, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

from .base import BaseBackbone


class BackboneType(str, Enum):
    X3D_S = "X3D-S"
    X3D_L = "X3D-L"
    CONVNEXT_2D = "CONVNEXT-2D"
    ACA_NET = "ACA-NET"
    VIDEO_SWIN = "VIDEO-SWIN"
    UNIFORMER_XXS = "UNIFORMER-XXS"

    @classmethod
    def list_names(cls) -> List[str]:
        return [b.value for b in cls]


class BackboneFactory:
    _registry: Dict[str, Type[BaseBackbone]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: Type[BaseBackbone]):
            cls._registry[name.upper()] = subclass
            return subclass
        return decorator

    @classmethod
    def create(
        cls, 
        backbone_type: str | BackboneType, 
        pretrained: bool = True, 
        freeze_net: bool = True,
        **kwargs: Any
    ) -> BaseBackbone:
        """Instantiates and returns the requested backbone.
        Args:
            backbone_type: String or Enum of the backbone name.
            pretrained: Whether to load pretrained weights.
            freeze_net: Whether to freeze the network parameters.
            **kwargs: Additional arguments for advanced backbone configurations.
        """
        key = (backbone_type.value if isinstance(backbone_type, BackboneType) else str(backbone_type)).upper()

        if key not in cls._registry:
            valid_names = ", ".join(cls._registry.keys())
            raise ValueError(f"Backbone '{key}' in registry. Valid names: [{valid_names}]")

        backbone_cls = cls._registry[key]
        
        return backbone_cls(pretrained=pretrained, freeze_net=freeze_net, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))


@BackboneFactory.register(BackboneType.X3D_S)
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
        - X3D-S does not downsample the temporal axis in its stem/residual stages,
          so the output temporal length is preserved (T_out = T_in).

    Args:
        out_dim (int, optional): Output feature dimension. Default: 192.
        pretrained (bool, optional): Load Kinetics-pretrained X3D-S weights.
            Default: True.
        freeze_net (bool, optional): Freeze the X3D trunk parameters and keep its
            BatchNorm layers in evaluation mode. Default: True.
        progress (bool, optional): Show the checkpoint download progress bar when
            pretrained weights are requested. Default: True.
    """


    def __init__(
        self,
        pretrained: bool = True,
        freeze_net: bool = True,
        progress: bool = True,
    ):

        super().__init__(out_dim=192, pretrained=pretrained, freeze_net=freeze_net)

        model = self._load_x3d_s(pretrained=pretrained, progress=progress)
        self._validate_x3d_model(model)

        # PyTorchVideo X3D is structured as:
        # Stem -> Stage1 -> Stage2 -> Stage3 -> Stage4 -> Classification Head.
        # The head performs spatio-temporal pooling and classification, therefore
        # it must be removed to preserve the temporal feature sequence.
        self.trunk = nn.Sequential(*list(model.blocks[:-1]))

        if freeze_net:
            self.freeze()

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

        # Spatial pooling only: preserve the temporal sequence for the neck.
        features = features.mean(dim=(-2, -1))  # (B, 192, T)
        features = features.transpose(1, 2).contiguous()  # (B, T, 192)

        return features  # (B, T, out_dim)


@BackboneFactory.register(BackboneType.X3D_L)
class X3DLBackbone(BaseBackbone):

    def __init__(
        self,
        pretrained: bool = True,
        freeze_net: bool = True,
        progress: bool = True,
    ):
        super().__init__(out_dim=432, pretrained=pretrained, freeze_net=freeze_net)

        model = self._load_x3d_l(pretrained=pretrained, progress=progress)
        
        # 1. Trunk (Stage 0 -> Stage 4)
        self.trunk = nn.Sequential(*list(model.blocks[:-1]))
        
        # 2. Extract the head sub-layers that project 192 -> 432
        head_pool = model.blocks[5].pool
        self.head_proj = nn.Sequential(
            head_pool.pre_conv,
            head_pool.pre_norm,
            head_pool.pre_act
        )

        if freeze_net:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, 16, 312, 312)
        x_feat = self.trunk(x)             # (B, 192, 16, 10, 10)
        x_feat = self.head_proj(x_feat)     # (B, 432, 16, 10, 10)
        
        # Spatial Average Pooling only over H' and W' -> preserves T=16
        x_feat = x_feat.mean(dim=(-2, -1))  # (B, 432, 16)
        x_feat = x_feat.permute(0, 2, 1).contiguous() # (B, 16, 432)

        return x_feat

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


from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

@BackboneFactory.register(BackboneType.CONVNEXT_2D)
class ConvNeXt2DBackbone(BaseBackbone):
    """
    Used for 2D spatial feature extraction (based on ConvNeXt-Tiny)
    Produces tensors of dimention 768 for each frame of the input sequence
    """
    def __init__(self, pretrained: bool = True, freeze_net: bool = True):
        super().__init__(out_dim=768, pretrained=pretrained, freeze_net=freeze_net)
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        
        self.features = model.features
        self.avgpool = model.avgpool

        if freeze_net:
            self.freeze()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        feat = self.features(x_2d)       # (B*T, 768, H', W')
        feat = self.avgpool(feat)        # (B*T, 768, 1, 1)
        feat = torch.flatten(feat, 1)    # (B*T, 768)
        
        return feat.view(B, T, self.out_dim)  # (B, T, 768)


# Probably unused. To clean
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

# Probably unused. To refactor (other task)
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
    def __init__(self, kernel_size: int = 7):
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
        self.tsci = TSCIModule()
        
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


@BackboneFactory.register(BackboneType.ACA_NET)
class ACANetBackbone(BaseBackbone):
    """
    ACA-Net Backbone for spatio-temporal feature extraction from video.
    
    Args:
        out_dim (int): Output channel dimension (Default: 512).
        pretrained (bool): Load ImageNet pre-trained weights for ResNet50.
        freeze_resnet (bool): Freeze the trunk parameters if set to True.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_resnet: bool = True,
    ):
        super().__init__(out_dim=2048, pretrained=pretrained, freeze_net=freeze_resnet)
        
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

        if freeze_resnet:
            self.freeze()

    def _wrap_layer(self, layer: nn.Sequential) -> nn.ModuleList:
        wrapped_blocks = nn.ModuleList()
        for block in layer:
            wrapped_blocks.append(ACABottleneck(block))
        return wrapped_blocks

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
        
        return feat


@BackboneFactory.register(BackboneType.VIDEO_SWIN)
class VideoSwinBackbone(BaseBackbone):
    """
    Video Swin Transformer backbone (Swin3D-Tiny).
    Extracts spatio-temporal features based on 3D self-attention mechanisms (Shifted Windows).
    
    Plug-and-play implementation: 
    Supports video input (B, C, T, H, W) but halves the temporal resolution (T/2).
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_net: bool = True,
    ):
        super().__init__(out_dim=768, pretrained=pretrained, freeze_net=freeze_net)
        
        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        self.swin = swin3d_t(weights=weights)
        
        if freeze_net:
            self.freeze()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"VideoSwinBackbone expects a 5D video tensor (B, C, T, H, W), got {tuple(x.shape)}."
            )
            
        B, C, T_in, H, W = x.shape
        
        # (B, 3, T, H, W) -> (B, T_pad, H_pad, W_pad, 96)
        x = self.swin.patch_embed(x)
        x = self.swin.pos_drop(x)

        # (B, T_pad, H_pad, W_pad, 96) -> (B, T', H', W', 768)
        x = self.swin.features(x)

        x = self.swin.norm(x)

        # Average only over H' and W' keeping T' (B, T', 768)
        feat = x.mean(dim=(2, 3))
                    
        feat = feat.contiguous() # (B, T', 768)
        
        return feat # (B, T', out_dim)

"""
The following code implements the UniFormer-XXS Backbone natively adapted from Sense-X official implementation.
Adapted from:
    https://github.com/Sense-X/UniFormer/blob/main/video_classification/slowfast/models/uniformer_light.py
"""

UNIFORMER_XXS_160_K400 = "weights/uniformer_xxs16_160_k400.pth"

from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)
    
def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   

class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x    

class SpeicalPatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 56):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


# --- WRAPPER BACKBONE ---
@BackboneFactory.register(BackboneType.UNIFORMER_XXS)
class UniFormerXXSBackbone(BaseBackbone):
    """ 
    UniFormer-XXS Backbone natively adapted from Sense-X official implementation.
    Adapted from:
        https://github.com/Sense-X/UniFormer/blob/main/video_classification/slowfast/models/uniformer_light.py
    """
    def __init__(
        self, 
        pretrained: bool = True,
        weights_path: str = "weights/uniformer_xxs16_160_k400.pth",
        freeze_net: bool = True,
        in_channels: int = 3,
        img_size: int = 224,
        depths: list = [2, 5, 8, 2], 
        embed_dims: list = [56, 112, 224, 448]
    ):
        super().__init__(out_dim=embed_dims[-1], pretrained=pretrained, freeze_net=freeze_net)

        mlp_ratio = 3.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.

        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        
        self.patch_embed1 = SpeicalPatchEmbed(
            in_chans=in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        num_heads = [1, 2, 4, 8]
        
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depths[0])])
            
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depths[0]], norm_layer=norm_layer)
            for i in range(depths[1])])
            
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depths[0]+depths[1]], norm_layer=norm_layer)
            for i in range(depths[2])])
            
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depths[0]+depths[1]+depths[2]], norm_layer=norm_layer)
        for i in range(depths[3])])
        
        self.norm = bn_3d(embed_dims[-1])
        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if 't_attn.qkv.weight' in name or 't_attn.qkv.bias' in name or 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)

        if pretrained:
            self._load_pretrained_weights(weights_path)

        if freeze_net:
            self.freeze()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self, weights_path: str):
        """
        Weight adaptation for UniFormer-XXS pre-trained on Kinetics-400.
        Loads the weights from a local checkpoint file and removes the classification head parameters.
        """
        if not os.path.exists(weights_path):
            print(f"[WARNING] Local weights file not found at '{weights_path}'")
            return
            
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
        
        keys_to_delete = [
            k for k in state_dict.keys() 
            if k.startswith("head.") or k.startswith("pre_logits.") or k.startswith("cls_") or "cls" in k
        ]
        for k in keys_to_delete:
            del state_dict[k]
            
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[SUCCESS] Weights loaded successfully from {weights_path}")
        print(f"  - Missing keys: {len(missing)} (dovrebbe essere 0)")
        print(f"  - Unexpected keys: {len(unexpected)} (dovrebbe essere 0)")

        if len(missing) > 0:
            print("\n--- ESEMPIO MISSING KEYS ---")
            for k in missing:
                print(k)
                
        if len(unexpected) > 0:
            print("\n--- ESEMPIO UNEXPECTED KEYS ---")
            for k in unexpected:
                print(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        
        for blk in self.blocks1: x = blk(x)
        x = self.patch_embed2(x)
        
        for blk in self.blocks2: x = blk(x)
        x = self.patch_embed3(x)
        
        for blk in self.blocks3: x = blk(x)
        x = self.patch_embed4(x)
        
        for blk in self.blocks4: x = blk(x)
        
        x = self.norm(x) # [B, D, T, H', W']
        
        x = torch.mean(x, dim=(-2, -1)) # [B, D, T]
        
        x = x.permute(0, 2, 1).contiguous()
        return x