import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Backbone
class PretrainedViT_B16(nn.Module):
    def __init__(self, image_size: int = 224, in_channels: int = 1):
        super(PretrainedViT_B16, self).__init__()

        if in_channels not in (1, 3):
            raise ValueError(f"in_channels must be 1 or 3, got {in_channels}")

        self.requested_image_size = image_size
        self.patch_size = 16
        self.model_image_size = int(math.ceil(image_size / self.patch_size) * self.patch_size)

        # Build model with target image size, then load pretrained weights.
        # For non-224 sizes, interpolate positional embeddings.
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=None, image_size=self.model_image_size)

        if in_channels == 1:
            old_conv = self.base.conv_proj
            self.base.conv_proj = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        state_dict = weights.get_state_dict(progress=True)
        state_dict = self._adapt_input_projection_if_needed(
            state_dict=state_dict,
            in_channels=in_channels,
            conv_proj_weight_key="conv_proj.weight",
        )
        state_dict = self._resize_positional_embedding_if_needed(
            state_dict=state_dict,
            target_image_size=self.model_image_size,
            patch_size=self.patch_size,
            pos_embed_key="encoder.pos_embedding",
        )
        self.base.load_state_dict(state_dict, strict=True)

        # Freeze all pretrained parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # Remove original classification head
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        # Custom FC layers (和 PretrainedGoogLeNet 风格一致)
        self.extra_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            Swish(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            Swish(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)   # binary output
        )

    @staticmethod
    def _resize_positional_embedding_if_needed(
        state_dict: dict,
        target_image_size: int,
        patch_size: int,
        pos_embed_key: str,
    ) -> dict:
        if pos_embed_key not in state_dict:
            return state_dict

        pos_embed = state_dict[pos_embed_key]
        old_seq_len = pos_embed.shape[1]
        old_grid = int(math.sqrt(old_seq_len - 1))
        new_grid = target_image_size // patch_size
        new_seq_len = new_grid * new_grid + 1

        if new_seq_len == old_seq_len:
            return state_dict

        cls_token = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        hidden_dim = patch_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, old_grid, old_grid, hidden_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, hidden_dim)

        state_dict[pos_embed_key] = torch.cat([cls_token, patch_pos_embed], dim=1)
        return state_dict

    @staticmethod
    def _adapt_input_projection_if_needed(
        state_dict: dict,
        in_channels: int,
        conv_proj_weight_key: str,
    ) -> dict:
        if in_channels != 1 or conv_proj_weight_key not in state_dict:
            return state_dict

        # Keep behavior equivalent to duplicating one grayscale channel to RGB.
        # y = sum_c(W_c * x) when x is repeated on 3 channels.
        conv_w = state_dict[conv_proj_weight_key]
        state_dict[conv_proj_weight_key] = conv_w.sum(dim=1, keepdim=True)
        return state_dict

    def forward(self, x):
        # Keep original content unchanged; only pad border to satisfy ViT patch grid.
        _, _, h, w = x.shape
        target_h = int(math.ceil(h / self.patch_size) * self.patch_size)
        target_w = int(math.ceil(w / self.patch_size) * self.patch_size)
        if target_h != h or target_w != w:
            pad_h = target_h - h
            pad_w = target_w - w
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        x = self.base(x)        # ViT backbone features
        x = self.extra_fc(x)    # custom classifier
        return x
