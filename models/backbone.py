"""
Implementations of spatial and spatio-temporal backbone modules.
"""
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
    Supports video input (B, C, T, H, W) and temporally realigns (via 1D interpolation) 
    the output features to return the same length T as the input (B, T, out_dim).
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_net: bool = True,
    ):
        super().__init__(out_dim=768, pretrained=pretrained, freeze_net=freeze_net)
        
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        model = swin3d_t(weights=weights)
        
        # Keep only the feature extractor and the normalization layer
        self.features = model.features
        self.norm = model.norm
        
        if freeze_net:
            self.freeze()
        
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
            
        # Permute to (B, T, D)
        feat = feat.transpose(1, 2).contiguous() # (B, T_in, 768)
        
        return feat # (B, T_in, out_dim)

#Checkpoint UniFormer-XXS addestrato su Kinetics-400 (16 frame)
UNIFORMER_XXS_K400_URL = "https://github.com/SenseTime-Research/UniFormer/releases/download/v1.0/uniformer_xxs_k400_16x4.pth"

class DynamicPositionEmbedding(nn.Module):
    """Dynamic Position Embedding (DPE) with 3D Depthwise Convolution."""
    def __init__(self, dim: int, kernel_size: tuple = (3, 3, 3)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=kernel_size, stride=1, padding=padding, groups=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        return x + self.dwconv(x)


class LocalMHRA(nn.Module):
    """Multi-Head Relation Aggregator for superficial stages (Local Convolutions)."""
    def __init__(self, dim: int, kernel_size: tuple = (3, 3, 3)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding, groups=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        out = self.conv(x)
        B, C, T, H, W = out.shape
        out = out.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        out = self.proj(out)
        return out.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]


class GlobalMHRA(nn.Module):
    """Multi-Head Relation Aggregator for deep stages (3D Self-Attention)."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        N = T * H * W

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # Use FlashAttention / Memory Efficient Attention
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


class FFN(nn.Module):
    """Feed Forward Network."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1)
        x_flat = self.fc1(x_flat)
        x_flat = self.act(x_flat)
        x_flat = self.fc2(x_flat)
        return x_flat.permute(0, 4, 1, 2, 3)


class UniFormerBlock(nn.Module):
    """Unified UniFormer block (Dynamic Position Embedding + Multi-Head Relation Aggregator + Feed Forward Network)."""
    def __init__(self, dim: int, num_heads: int, is_global: bool = False):
        super().__init__()
        self.dpe = DynamicPositionEmbedding(dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.mhra = GlobalMHRA(dim, num_heads=num_heads) if is_global else LocalMHRA(dim)
        self.norm2 = nn.BatchNorm3d(dim)
        self.ffn = FFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dpe(x)
        x = x + self.mhra(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    """Patch Embedding / Downsampling between the stages of the network."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(1, 4, 4), stride=(1, 4, 4)):
        super().__init__()
        padding = (kernel_size[0] // 2, 0, 0) if kernel_size[0] % 2 != 0 else (0, 0, 0)
        self.proj = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


@BackboneFactory.register(BackboneType.UNIFORMER_XXS)
class UniFormerXXSBackbone(BaseBackbone):
    """
    Backbone UniFormer-XXS.
    
    Architettural specifications:
    - Depths: [3, 4, 8, 3]
    - Embed Dims: [64, 128, 320, 512]
    - Number of Heads: [1, 2, 5, 8]
    - Parameters: ~10.2M - 10.4M
    """
    def __init__(
        self,
        pretrained: bool = True,
        freeze_net: bool = True,
        in_channels: int = 3,
    ):
        super().__init__(out_dim=512, pretrained=pretrained, freeze_net=freeze_net)

        # UniFormer-XXS configuration
        depths = [3, 4, 8, 3]
        embed_dims = [64, 128, 320, 512]
        num_heads = [1, 2, 5, 8]

        # Stage 1 (Stem + Local Blocks)
        self.patch_embed1 = PatchEmbed3D(in_channels, embed_dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.stage1 = nn.ModuleList([
            UniFormerBlock(embed_dims[0], num_heads[0], is_global=False) for _ in range(depths[0])
        ])

        # Stage 2 (Local Blocks)
        self.patch_embed2 = PatchEmbed3D(embed_dims[0], embed_dims[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.stage2 = nn.ModuleList([
            UniFormerBlock(embed_dims[1], num_heads[1], is_global=False) for _ in range(depths[1])
        ])

        # Stage 3 (Global Blocks)
        self.patch_embed3 = PatchEmbed3D(embed_dims[1], embed_dims[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.stage3 = nn.ModuleList([
            UniFormerBlock(embed_dims[2], num_heads[2], is_global=True) for _ in range(depths[2])
        ])

        # Stage 4 (Global Blocks)
        self.patch_embed4 = PatchEmbed3D(embed_dims[2], embed_dims[3], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.stage4 = nn.ModuleList([
            UniFormerBlock(embed_dims[3], num_heads[3], is_global=True) for _ in range(depths[3])
        ])

        self.norm = nn.BatchNorm3d(embed_dims[3])

        # Loading pre-trained weights
        if pretrained:
            self._load_pretrained_weights(UNIFORMER_XXS_K400_URL)

    def _load_pretrained_weights(self, url: str):
        print(f"Downloading/Loading pre-trained weights from {url}...")
        checkpoint = torch.hub.load_state_dict_from_url(
            url, progress=True, map_location='cpu'
        )

        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("backbone.", "").replace("module.", "")
            if new_key.startswith("head.") or new_key.startswith("fc."):
                continue
            
            for i in range(1, 5):
                new_key = new_key.replace(f"blocks{i}.", f"stage{i}.")
                
            cleaned_state_dict[new_key] = value

        missing, unexpected = self.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Weights loaded successfully! Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, T, C, H, W]
        Output: [B, T, D] 
        """
        # Conversion from [B, T, C, H, W] to [B, C, T, H, W] for 3D Conv
        if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Passing through the backbone stages
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        x = self.patch_embed4(x)
        for blk in self.stage4:
            x = blk(x)

        x = self.norm(x)  # Shape: [B, D, T, H', W'] -> es: [B, 512, 16, 7, 7]

        # Global Average Pooling on dimensions H and W
        x = torch.mean(x, dim=(-2, -1))  # Shape: [B, D, T]

        # Permutation for the sequential representation [B, T, D]
        x = x.permute(0, 2, 1).contiguous()
        return x