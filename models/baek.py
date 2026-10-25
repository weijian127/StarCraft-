"""Baek et al.'s 3D ResNet with the embedded-Gaussian non-local block.

Reimplemented from InsungBaek/StarCraft2, commit 664b17a, specifically
models/resnet.py::ResNet3D_non_local and its embedding/residual/non-local layers.
Paper: https://doi.org/10.1371/journal.pone.0264550 (Tables 3 and 4).

The canonical input is three categorical layers [B, 50, 128, 128]. Residual
convolutions use stride 1, as actually executed by the reference BasicBlock3D;
only the stem and pooling layers downsample. The final adaptive average pool is
identical to the reference (5, 4, 4) pool at canonical resolution, while allowing
shorter clips for debugging. Raw uint16 unit IDs use a fixed 65,536-row embedding
instead of depending on the installed PySC2 unit list. No SC2 import is needed.
"""

import torch
from torch import nn
from torch.nn import functional as F


FEATURES = ("visibility_map", "player_relative", "unit_type")
EMBEDDING_DIMS = (10, 10, 100)


class CategoricalEmbedding(nn.Module):
    def __init__(self, categories, dimensions, *, freeze=False):
        super().__init__()
        self.embed_fn = nn.Embedding(categories, dimensions, padding_idx=0)
        self.embed_fn.weight.requires_grad_(not freeze)

    def forward(self, x):
        # [B,T,H,W,E] -> [B,T,E,H,W], matching the reference channel layout.
        return self.embed_fn(x).permute(0, 1, 4, 2, 3)


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv_skip = nn.Conv3d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        residual = self.conv_skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + residual)


class EmbeddedGaussianNonLocal3D(nn.Module):
    """Unscaled softmax(theta(x) @ phi(x)), with spatially pooled keys/values.

    This is the reference embedded Gaussian, not scaled dot-product attention.
    The output BN starts at zero, so the block initially acts as the identity.
    """

    def __init__(self, channels):
        super().__init__()
        self.inter_channels = max(1, channels // 2)
        self.g = nn.Sequential(
            nn.Conv3d(channels, self.inter_channels, 1),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.theta = nn.Conv3d(channels, self.inter_channels, 1)
        self.phi = nn.Sequential(
            nn.Conv3d(channels, self.inter_channels, 1),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.W = nn.Sequential(
            nn.Conv3d(self.inter_channels, channels, 1),
            nn.BatchNorm3d(channels),
        )
        nn.init.zeros_(self.W[1].weight)
        nn.init.zeros_(self.W[1].bias)

    def forward(self, x):
        values = self.g(x).flatten(2).transpose(1, 2)
        queries = self.theta(x).flatten(2).transpose(1, 2)
        keys = self.phi(x).flatten(2)
        attention = (queries @ keys).softmax(dim=-1)
        y = (attention @ values).transpose(1, 2).reshape(
            x.shape[0], self.inter_channels, *x.shape[2:]
        )
        return x + self.W(y)


class BaekResNet3D(nn.Module):
    """Two logits: index 0 = Protoss wins; index 1 = Terran wins.

    Embeddings are trainable by default, matching the released code. Set
    freeze_embeddings=True to follow the paper's statement that they are fixed.
    """

    def __init__(self, *, unit_type_vocab_size=65536, freeze_embeddings=False):
        super().__init__()
        if not 1 <= unit_type_vocab_size <= 65536:
            raise ValueError("unit_type_vocab_size must be in [1, 65536]")
        self.config = {
            "unit_type_vocab_size": unit_type_vocab_size,
            "freeze_embeddings": freeze_embeddings,
        }
        categories = (4, 5, unit_type_vocab_size)
        self.embeddings = nn.ModuleDict({
            name: CategoricalEmbedding(size, dim, freeze=freeze_embeddings)
            for name, size, dim in zip(FEATURES, categories, EMBEDDING_DIMS)
        })
        self.conv1 = nn.Conv3d(120, 32, 7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.block1 = BasicBlock3D(32, 64)
        self.pool1 = nn.MaxPool3d((5, 4, 4))
        self.block2 = BasicBlock3D(64, 128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.block3 = BasicBlock3D(128, 256)
        self.non_block2 = EmbeddedGaussianNonLocal3D(256)
        self.pool3 = nn.AdaptiveAvgPool3d(1)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(256, 2)

    def forward_features(self, inputs):
        if set(inputs) != set(FEATURES):
            raise ValueError(f"Expected exactly these categorical inputs: {FEATURES}")
        shape = inputs[FEATURES[0]].shape
        if len(shape) != 4 or min(shape[0:2]) < 1 or shape[1] < 9 or min(shape[2:]) < 64:
            raise ValueError("Expected [B,T,H,W] with B>=1, T>=9, H/W>=64 (normally 50x128x128)")
        embedded = []
        # A fixed order makes predictions independent of the input dict's order.
        for name in FEATURES:
            layer = inputs[name]
            if layer.shape != shape or layer.dtype != torch.long:
                raise ValueError(f"{name} must be an aligned int64 [B,T,H,W] tensor")
            size = self.embeddings[name].embed_fn.num_embeddings
            if torch.any(layer < 0) or torch.any(layer >= size):
                raise ValueError(f"{name} category IDs must be in [0, {size - 1}]")
            embedded.append(self.embeddings[name](layer))
        x = torch.cat(embedded, dim=2).permute(0, 2, 1, 3, 4)
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop_out(self.pool1(self.block1(x)))
        x = self.drop_out(self.pool2(self.block2(x)))
        return self.non_block2(self.block3(x))

    def forward(self, inputs):
        x = self.forward_features(inputs)
        x = self.drop_out(self.pool3(x)).flatten(1)
        return self.linear(x)
