from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm>=0.9.16")

from .evotherm_adapter import build_evotherm_adapter


# ---------------------------------------------------------------------------
# EvoResNet50: ResNet-50 + EvoTherm-Adapter
# ---------------------------------------------------------------------------

class EvoResNet50(nn.Module):
    """
    ResNet-50 (pre-trained ImageNet) + EvoTherm-Adapter.

    Adapter is inserted AFTER the spatial feature map (Stage-4, 2048 channels)
    and BEFORE global average pooling + classification head.

    Shape flow:
      Input (B, 3, 224, 224)
        → ResNet-50 backbone (frozen stage 1-4)   : (B, 2048, 7, 7)
        → EvoThermAdapter (4D CNN mode)            : (B, 2048, 7, 7)
        → AdaptiveAvgPool2d(1)                     : (B, 2048)
        → ILC calibrated global pool               : (B, 2048)
        → Linear head                              : (B, num_classes)
    """

    BACKBONE_DIM = 2048

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 img_size: int = 224, drop_rate: float = 0.0,
                 num_heads: int = 8, num_evo_steps: int = 3,
                 gamma: float = 0.7, warm_epochs: int = 5):
        super().__init__()
        self.warm_epochs = warm_epochs
        self.img_size = img_size

        # Pre-trained backbone (global_pool='' to get spatial feature map)
        self.backbone = timm.create_model(
            'resnet50', pretrained=pretrained,
            num_classes=0, in_chans=3,
            drop_rate=drop_rate, global_pool=''
        )

        # EvoTherm-Adapter (operates on 2048-dim spatial feature map)
        self.adapter = build_evotherm_adapter(
            dim=self.BACKBONE_DIM,
            num_heads=num_heads,
            num_evo_steps=num_evo_steps,
            gamma=gamma,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.BACKBONE_DIM, num_classes)

        # Initialize head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def freeze_backbone(self):
        """Freeze backbone weights (Phase 1: train only adapter)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone weights (Phase 2: joint fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.backbone(x)          # (B, 2048, 7, 7)
        adapted = self.adapter(spatial)     # (B, 2048, 7, 7) — evolved features
        feat = self.pool(adapted).flatten(1)  # (B, 2048)
        return self.head(feat)              # (B, num_classes)

    def adapter_parameters(self):
        """Return only adapter + head parameters (for Phase 1 optimizer)."""
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def backbone_parameters(self):
        """Return only backbone parameters (for Phase 2 optimizer)."""
        return list(self.backbone.parameters())


# ---------------------------------------------------------------------------
# EvoDeiTTiny: DeiT-Tiny + EvoTherm-Adapter
# ---------------------------------------------------------------------------

class EvoDeiTTiny(nn.Module):
    """
    DeiT-Tiny (pre-trained ImageNet distillation) + EvoTherm-Adapter.

    Adapter is inserted AFTER DeiT's transformer blocks but BEFORE
    the classification head, operating on the [CLS] token context.

    Shape flow:
      Input (B, 3, 224, 224)
        → DeiT-Tiny backbone (frozen 12 transformer blocks): (B, N+1, 192)
        → Extract CLS + patch tokens
        → EvoThermAdapter (3D ViT mode on patch tokens)   : (B, N, 192)
        → Mean-pool evolved patch tokens                   : (B, 192)
        → ILC intrinsic calibration                        : (B, 192)
        → Linear head                                      : (B, num_classes)
    """

    BACKBONE_DIM = 192

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 img_size: int = 224, drop_rate: float = 0.0,
                 num_heads: int = 4, num_evo_steps: int = 3,
                 gamma: float = 0.7, warm_epochs: int = 5):
        super().__init__()
        self.warm_epochs = warm_epochs
        self.img_size = img_size

        self.backbone = timm.create_model(
            'deit_tiny_patch16_224', pretrained=pretrained,
            num_classes=0, in_chans=3, drop_rate=drop_rate,
        )
        # embed_dim = 192 for DeiT-Tiny
        dim = self.backbone.embed_dim  # 192

        self.adapter = build_evotherm_adapter(
            dim=dim,
            num_heads=num_heads,
            num_evo_steps=num_evo_steps,
            gamma=gamma,
        )

        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DeiT forward_features returns full token sequence: (B, N+1, D)
        tokens_full = self.backbone.forward_features(x)  # (B, 197, 192) for 224px input
        cls_token = tokens_full[:, 0]                    # (B, 192)
        patch_tokens = tokens_full[:, 1:]                # (B, 196, 192)

        # Adapter operates on patch tokens (3D mode: B, N, D)
        adapted_patches = self.adapter(patch_tokens)     # (B, 196, 192)

        # Fuse CLS + evolved patch mean (richer representation than CLS alone)
        patch_mean = adapted_patches.mean(dim=1)         # (B, 192)
        feat = 0.5 * cls_token + 0.5 * patch_mean       # (B, 192)

        return self.head(feat)                           # (B, num_classes)

    def adapter_parameters(self):
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def backbone_parameters(self):
        return list(self.backbone.parameters())


# ---------------------------------------------------------------------------
# EvoSwinV2Tiny: SwinV2-Tiny + EvoTherm-Adapter
# ---------------------------------------------------------------------------

class EvoSwinV2Tiny(nn.Module):
    """
    SwinV2-Tiny (pre-trained ImageNet) + EvoTherm-Adapter.

    SwinV2 outputs a global-pooled (B, 768) vector. The adapter treats
    this as a single-token sequence (2D mode).

    Shape flow:
      Input (B, 3, 256, 256)
        → SwinV2-Tiny backbone (frozen)               : (B, 768)
        → EvoThermAdapter (2D global mode)            : (B, 768)
        → ILC intrinsic calibration (already applied) : (B, 768)
        → Linear head                                 : (B, num_classes)
    """

    BACKBONE_DIM = 768

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 img_size: int = 256, drop_rate: float = 0.0,
                 num_heads: int = 8, num_evo_steps: int = 3,
                 gamma: float = 0.7, warm_epochs: int = 5):
        super().__init__()
        self.warm_epochs = warm_epochs
        self.img_size = img_size

        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256', pretrained=pretrained,
            num_classes=0, in_chans=3,
            drop_rate=drop_rate, img_size=img_size
        )
        dim = self.backbone.num_features  # 768

        self.adapter = build_evotherm_adapter(
            dim=dim,
            num_heads=num_heads,
            num_evo_steps=num_evo_steps,
            gamma=gamma,
        )

        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwinV2 with num_classes=0 returns global pooled (B, 768)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        feat = self.backbone(x)           # (B, 768)
        adapted = self.adapter(feat)      # (B, 768) — EvoTherm in 2D mode
        return self.head(adapted)         # (B, num_classes)

    def adapter_parameters(self):
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def backbone_parameters(self):
        return list(self.backbone.parameters())


# ---------------------------------------------------------------------------
# Registry + Factory
# ---------------------------------------------------------------------------

ADAPTED_BASELINE_REGISTRY = {
    'evo_resnet50':    EvoResNet50,
    'evo_deit_tiny':   EvoDeiTTiny,
    'evo_swinv2_tiny': EvoSwinV2Tiny,
}


def build_adapted_baseline(name: str, **kwargs):
    """
    Factory for adapted baseline models.

    Example:
        model = build_adapted_baseline('evo_resnet50', num_classes=2)
        model.freeze_backbone()  # Phase 1: train adapter only
    """
    assert name in ADAPTED_BASELINE_REGISTRY, (
        f"Unknown adapted model: {name}. Choose from {list(ADAPTED_BASELINE_REGISTRY)}"
    )
    return ADAPTED_BASELINE_REGISTRY[name](**kwargs)
