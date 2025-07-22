"""
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
"""

import math

import torch
import torch as th
import torch.nn as nn


def xf_convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x, key_padding_mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, key_padding_mask)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv, key_padding_mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards

        if key_padding_mask is not None:
            weight = weight.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1)
                float('-inf'),
            )
        wdtype = weight.dtype
        weight = th.softmax(weight.float(), dim=-1).type(wdtype)
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: th.Tensor, key_padding_mask=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        return x


class LayoutTransformerEncoder(nn.Module):
    def __init__(
            self,
            layout_length: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            num_heads: int,
 


            resolution_to_attention=[],

    ):
        super().__init__()
   


        self.transform = Transformer(
            n_ctx=layout_length, # 28
            width=hidden_dim, # model_channel, 
            layers=num_layers, # 4
            heads=num_heads # 8
        )


        self.transformer_proj = nn.Linear(hidden_dim, output_dim)

        self.obj_bbox_embedding = nn.Linear(5, hidden_dim)

        self.final_ln = LayerNorm(hidden_dim)

        self.dtype = torch.float32

        self.resolution_to_attention = resolution_to_attention
        self.image_patch_bbox_embedding = {}
        for resolution in self.resolution_to_attention:
            interval = 1.0 / resolution
            self.image_patch_bbox_embedding['resolution{}'.format(resolution)] = torch.FloatTensor(
                [(interval * j, interval * i, interval * (j + 1), interval * (i + 1)) for i in range(resolution) for j in range(resolution)],
            ).cuda()  # (L, 4)



    def forward(self, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, image_patch_bbox=None):
        assert (obj_class is not None) or (obj_bbox is not None) or (obj_mask is not None)
        outputs = {}

        xf_in = None
        obj_bbox_embedding = self.obj_bbox_embedding(obj_bbox.to(self.dtype))

        if xf_in is None:
            xf_in = obj_bbox_embedding
        else:
            xf_in = xf_in + obj_bbox_embedding


        outputs['obj_bbox_embedding'] = obj_bbox_embedding.permute(0, 2, 1)
        for resolution in self.resolution_to_attention:
            outputs['image_patch_bbox_embedding_for_resolution{}'.format(resolution)] = torch.repeat_interleave(
                input=self.obj_bbox_embedding(
                    self.image_patch_bbox_embedding['resolution{}'.format(resolution)].to(self.dtype)
                ).unsqueeze(0),
                repeats = obj_bbox_embedding.shape[0],
                dim=0
            ).permute(0, 2, 1)


        key_padding_mask = None

        xf_out = self.transform(xf_in.to(self.dtype), key_padding_mask)  # NLC
      
        xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, 0])  # NC
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL
        outputs['xf_proj'] = xf_proj
        outputs['xf_out'] = xf_out

        return outputs
