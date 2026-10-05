import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm>=0.9.16")


# ---------------------------------------------------------------------------
# Shared Detection Head (reused across all baselines)
# ---------------------------------------------------------------------------

class _DetHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.cls = nn.Linear(feat_dim, num_classes)
        self.box = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, 4), nn.Sigmoid()
        )

    def forward(self, feat):
        return self.cls(feat), self.box(feat)


# ---------------------------------------------------------------------------
# DeiT-Tiny Wrapper
# ---------------------------------------------------------------------------

class DeiTTiny(nn.Module):
    """
    DeiT-Tiny (patch16, 224) backbone with task-specific head.
    Uses timm's pretrained weights when pretrained=True.
    """

    def __init__(self, task: str = 'classification', num_classes: int = 2,
                 pretrained: bool = True, img_size: int = 224,
                 in_chans: int = 3, decoder=None,
                 num_classes_cls: int = 4, num_classes_loc: int = 3,
                 drop_rate: float = 0.0):
        super().__init__()
        self.task = task
        self.img_size = img_size

        self.backbone = timm.create_model(
            'deit_tiny_patch16_224', pretrained=pretrained,
            num_classes=0,          # remove classifier head
            in_chans=in_chans,
            drop_rate=drop_rate,
        )
        embed_dim = self.backbone.embed_dim  # 192

        if task == 'classification':
            self.head = nn.Linear(embed_dim, num_classes)
        elif task in ('binary_seg', 'multiclass_seg'):
            assert decoder is not None
            self.decoder = decoder
        elif task == 'detection':
            self.head = _DetHead(embed_dim, num_classes)
        elif task == 'joint':
            self.cls_head = nn.Linear(embed_dim, num_classes_cls)
            self.det_head = _DetHead(embed_dim, num_classes_loc)

    def _get_patch_spatial(self, x):
        """Extract patch tokens as spatial tensor (B, D, Hg, Wg)."""
        feats = self.backbone.forward_features(x)                       # (B, N+1, D)
        patch = feats[:, 1:]                                             # (B, N, D)
        Hg = Wg = self.img_size // self.backbone.patch_embed.patch_size[0]
        return feats[:, 0], patch, Hg, Wg

    def forward(self, x):
        cls, patch, Hg, Wg = self._get_patch_spatial(x)

        if self.task == 'classification':
            return self.head(cls)
        elif self.task in ('binary_seg', 'multiclass_seg'):
            sp = patch.transpose(1, 2).reshape(x.shape[0], -1, Hg, Wg)
            return self.decoder(sp)
        elif self.task == 'detection':
            return self.head(cls)
        elif self.task == 'joint':
            return self.cls_head(cls), *self.det_head(cls)


# ---------------------------------------------------------------------------
# SwinV2-Tiny Wrapper
# ---------------------------------------------------------------------------

class SwinV2Tiny(nn.Module):
    """SwinV2-Tiny backbone with task-specific head."""

    def __init__(self, task: str = 'classification', num_classes: int = 2,
                 pretrained: bool = True, img_size: int = 256,
                 in_chans: int = 3, decoder=None,
                 num_classes_cls: int = 4, num_classes_loc: int = 3,
                 drop_rate: float = 0.0):
        super().__init__()
        self.task = task
        self.img_size = img_size

        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256', pretrained=pretrained,
            num_classes=0, in_chans=in_chans, drop_rate=drop_rate,
            img_size=img_size
        )
        # SwinV2-Tiny final feature dim = 768
        embed_dim = self.backbone.num_features

        if task == 'classification':
            self.head = nn.Linear(embed_dim, num_classes)
        elif task in ('binary_seg', 'multiclass_seg'):
            assert decoder is not None
            self.decoder = decoder
        elif task == 'detection':
            self.head = _DetHead(embed_dim, num_classes)
        elif task == 'joint':
            self.cls_head = nn.Linear(embed_dim, num_classes_cls)
            self.det_head = _DetHead(embed_dim, num_classes_loc)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        # SwinV2 returns global pooled feature by default (num_classes=0)
        feat = self.backbone(x)                                          # (B, D)

        if self.task == 'classification':
            return self.head(feat)
        elif self.task in ('binary_seg', 'multiclass_seg'):
            # For segmentation: use forward_features which gives spatial maps
            sp = self.backbone.forward_features(x)                      # (B, H', W', D)
            sp = sp.permute(0, 3, 1, 2)                                 # (B, D, H', W')
            return self.decoder(sp)
        elif self.task == 'detection':
            return self.head(feat)
        elif self.task == 'joint':
            return self.cls_head(feat), *self.det_head(feat)


# ---------------------------------------------------------------------------
# ResNet-50 Wrapper
# ---------------------------------------------------------------------------

class ResNet50(nn.Module):
    """ResNet-50 backbone with task-specific head."""

    def __init__(self, task: str = 'classification', num_classes: int = 2,
                 pretrained: bool = True, img_size: int = 224,
                 in_chans: int = 3, decoder=None,
                 num_classes_cls: int = 4, num_classes_loc: int = 3,
                 drop_rate: float = 0.0):
        super().__init__()
        self.task = task
        self.img_size = img_size

        self.backbone = timm.create_model(
            'resnet50', pretrained=pretrained, num_classes=0,
            in_chans=in_chans, drop_rate=drop_rate, global_pool=''
        )
        # ResNet-50 layer4 output: 2048 channels
        embed_dim = 2048

        self.pool = nn.AdaptiveAvgPool2d(1)

        if task == 'classification':
            self.head = nn.Linear(embed_dim, num_classes)
        elif task in ('binary_seg', 'multiclass_seg'):
            assert decoder is not None
            self.decoder = decoder
        elif task == 'detection':
            self.head = _DetHead(embed_dim, num_classes)
        elif task == 'joint':
            self.cls_head = nn.Linear(embed_dim, num_classes_cls)
            self.det_head = _DetHead(embed_dim, num_classes_loc)

    def forward(self, x):
        spatial = self.backbone(x)                                       # (B, 2048, H', W')
        feat = self.pool(spatial).flatten(1)                             # (B, 2048)

        if self.task == 'classification':
            return self.head(feat)
        elif self.task in ('binary_seg', 'multiclass_seg'):
            return self.decoder(spatial)
        elif self.task == 'detection':
            return self.head(feat)
        elif self.task == 'joint':
            return self.cls_head(feat), *self.det_head(feat)


# ---------------------------------------------------------------------------
# Unified Baseline Factory
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    'deit_tiny':    DeiTTiny,
    'swinv2_tiny':  SwinV2Tiny,
    'resnet50':     ResNet50,
}


def build_baseline(name: str, **kwargs):
    """
    Factory to build any baseline model.
    Example:
        model = build_baseline('deit_tiny', task='classification', num_classes=4)
    """
    assert name in BASELINE_REGISTRY, f"Unknown baseline: {name}. Choose from {list(BASELINE_REGISTRY)}"
    return BASELINE_REGISTRY[name](**kwargs)
