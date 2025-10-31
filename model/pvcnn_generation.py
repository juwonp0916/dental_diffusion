import functools
import time
import math
import torch.nn as nn
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint
from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention, Swish
from model import Transformer, LayerNorm

def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8, out_channels), Swish())

def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    c = 0
    for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(r * out_channels)
        for p in range(num_blocks):
            attention = k % 2 == 0 and k > 0 and p == 0
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                          attention=attention,
                                          with_se=with_se, normalize=normalize, eps=eps)

            if c == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels + embed_dim, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
            c += 1
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=1e-6,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3



    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and c > 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                              attention=attention,
                                              dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_blocks.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels


        num_centers, radius, num_neighbors, out_channels = sa_configs

        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(
            block(in_channels=extra_feature_channels + (embed_dim if k == 0 else 0), out_channels=out_channels,
                  include_coordinates=True))
        c += 1
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                                dropout=0.1,
                                with_se=False, normalize=True, eps=1e-6,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim,
                             out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = c % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                              attention=attention,
                                              dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True, normalize=normalize,
                                              eps=eps)
                
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels

class PVCNN2Base(nn.Module):

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1,
                 use_checkpoint=True):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.extra_feature_channels = extra_feature_channels
        self.in_channels = extra_feature_channels + 3
        self.use_checkpoint = use_checkpoint  # Enable gradient checkpointing to save memory

        # embedding to uniquely identify fdi position (not existence - use tooth_exists_indicator for that)
        self.fdi_embedding = nn.Embedding(num_embeddings=28, embedding_dim=8)

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)

        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = Attention(channels_sa_features, 8, D=1)

        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)

        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)


        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(embed_dim, embed_dim))
        


        self.bound_embedding = nn.Linear(5, embed_dim)

        self.bound_transformer = Transformer(
            n_ctx=28,
            width = embed_dim,
            layers = 4,
            heads = 8
        )

        self.bound_final_ln = LayerNorm(embed_dim)


    def get_timestep_embedding(self, timesteps, device):

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)

        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb


    def forward(self, xt, t, x0, fdi_indices, l_mask, o_mask, bound, original_missing_mask=None, debug_nan=False):

        # xt: (B, 28, 3, 1024)
        # x0: (B, 28, 3, 1024)
        # fdi_indices: (B, 28) - embedding indices [0-27] mapped from FDI tooth positions
        # o_mask: (B, 28, 1, 1)
        # l_mask: (B, 28, 1, 1)
        # bound (B, 28, 5) or None
        # original_missing_mask: (B, 28) - True for naturally missing teeth (use this for tooth existence check)
        # debug_nan: If True, print detailed NaN tracking through the network

        B, nT, nD, nP = xt.shape
        t = t.view(B, 1).expand(B, nT).reshape(B*nT)

        if debug_nan:
            print(f"\n{'='*80}")
            print(f"[NaN TRACE] Starting forward pass")
            print(f"[NaN TRACE] Input xt has NaN: {torch.isnan(xt).any()}")
            print(f"[NaN TRACE] Input x0 has NaN: {torch.isnan(x0).any()}")
            print(f"{'='*80}")

        # Create attention mask for global attention from original_missing_mask
        # Shape: (B*28,) - True for positions to mask out in attention
        if original_missing_mask is not None:
            # Expand from (B, 28) to (B*28,) by flattening
            attn_mask = original_missing_mask.reshape(B*nT)
        else:
            attn_mask = None

        # Use the actual FDI embedding indices passed from dataset (already mapped [0-27])
        fdi_embeddings = self.fdi_embedding(fdi_indices) # (B, 28, 8)

        # Handle missing bounding cylinder data
        # Instead of using zero bounds (which cause numerical instability in transformer),
        # use learned positional embeddings or skip the bound transformer entirely
        temb_raw = self.embedf(self.get_timestep_embedding(t, xt.device))

        if bound is not None:
            # Only use bound transformer when actual bound data is available
            bound_embedding = self.bound_embedding(bound)
            bound_embedding_transformed = self.bound_transformer(bound_embedding)
            bound_embedding_transformed = self.bound_final_ln(bound_embedding_transformed).reshape(B*nT, self.embed_dim)
            temb_raw = temb_raw + bound_embedding_transformed
        # else: Skip bound embedding when not available - use only timestep embedding

        temb = temb_raw[:, :, None].expand(-1, -1, xt.shape[-1])

        # Create binary indicator: 1 if tooth naturally exists, 0 if missing
        # Use original_missing_mask to identify missing teeth (not fdi_indices anymore)
        if original_missing_mask is not None:
            tooth_exists_mask = (~original_missing_mask).float().unsqueeze(-1).unsqueeze(-1)  # (B, 28, 1, 1)
        else:
            tooth_exists_mask = torch.ones((B, nT, 1, 1), device=xt.device)

        obs_indicator = o_mask.expand(-1, -1, 1, nP)  # (B, 28, 1, 1024) 
        fdi_embeddings = fdi_embeddings.unsqueeze(3).expand(-1, -1, -1, nP)  # (B, 28, 8, 1024) t
        tooth_exists_indicator = tooth_exists_mask.expand(-1, -1, 1, nP)  # (B, 28, 1, 1024) 

        x = torch.cat([
            xt*l_mask + x0*o_mask,
            fdi_embeddings,           # 8 channels - FDI-based positional embedding (tooth identity)
            obs_indicator,            # 1 channel  - is this tooth used as context?
            tooth_exists_indicator    # 1 channel  - does this tooth naturally exist in patient?
        ], dim=2)
        # x: (B, 28, (3+8+1+1), 1024) = (B, 28, 13, 1024)
        x = x.reshape(B*nT, nD+self.extra_feature_channels, nP)

        if debug_nan:
            print(f"[NaN TRACE] After input concatenation, x has NaN: {torch.isnan(x).any()}")
            if torch.isnan(x).any():
                print(f"[NaN TRACE] Breaking down components:")
                print(f"  - xt*l_mask + x0*o_mask has NaN: {torch.isnan(xt*l_mask + x0*o_mask).any()}")
                print(f"  - fdi_embeddings has NaN: {torch.isnan(fdi_embeddings).any()}")
                print(f"  - temb has NaN: {torch.isnan(temb).any()}")

        coords, features = x[:, :3, :].contiguous(), x
        coords_list, in_features_list = [], []

        # Set abstraction layer with gradient checkpointing to save memory
        for i, sa_blocks in enumerate(self.sa_layers):  # 4 layers
            in_features_list.append(features)
            coords_list.append(coords)

            # Prepare input
            if i == 0:
                input_tuple = (features, coords, temb)
            else:
                input_tuple = (torch.cat([features, temb], dim=1), coords, temb)

            # Handle Sequential blocks
            # NOTE: Do NOT pass attn_mask to SA layers - they operate on POINTS within each tooth (spatial),
            # not on TEETH across the dentition (semantic). The mask is tooth-level, not point-level.
            if isinstance(sa_blocks, nn.Sequential):
                output = input_tuple
                for layer_idx, layer in enumerate(sa_blocks):
                    # Enable detailed debug for first PVConv block in first SA layer
                    if debug_nan and i == 0 and hasattr(layer, 'forward') and 'debug' in layer.forward.__code__.co_varnames:
                        print(f"[NaN TRACE] SA layer {i}, block {layer_idx} ({type(layer).__name__})")
                        if self.use_checkpoint and self.training:
                            output = checkpoint(layer, output, True, use_reentrant=False)  # debug=True
                        else:
                            output = layer(output, debug=True)
                    else:
                        if self.use_checkpoint and self.training:
                            output = checkpoint(layer, output, use_reentrant=False)
                        else:
                            output = layer(output)
                features, coords, temb = output
            else:
                # Single module case
                if self.use_checkpoint and self.training:
                    features, coords, temb = checkpoint(sa_blocks, input_tuple, use_reentrant=False)
                else:
                    features, coords, temb = sa_blocks(input_tuple)

            if debug_nan:
                print(f"[NaN TRACE] After SA layer {i}, features has NaN: {torch.isnan(features).any()}")
                if torch.isnan(features).any():
                    print(f"[NaN TRACE] *** NaN first appeared in SA layer {i} ***")

        in_features_list[0] = x[:, 3:, :].contiguous()

        if self.global_att is not None:
            debug_attention = debug_nan  # Use same debug flag
            # IMPORTANT: Disable checkpointing for global attention to ensure correct mask passing
            # The checkpoint mechanism has issues with keyword arguments in closures
            features = self.global_att(features, attn_mask, debug=debug_attention)

            if debug_nan:
                print(f"[NaN TRACE] After global attention, features has NaN: {torch.isnan(features).any()}")
                if torch.isnan(features).any():
                    print(f"[NaN TRACE] *** NaN first appeared in global attention ***")

        # Feature propagation layer with gradient checkpointing
        for fp_idx, fp_blocks in enumerate(self.fp_layers):  # 4 layers

            jump_coords = coords_list[-1 - fp_idx]
            fump_feats = in_features_list[-1 - fp_idx]

            # Handle Sequential blocks
            # NOTE: Do NOT pass attn_mask to FP layers - same reason as SA layers
            # FP layer PVConv attention is point-level (spatial), not tooth-level (semantic)
            if isinstance(fp_blocks, nn.Sequential):
                output = (jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb)

                for layer_idx, layer in enumerate(fp_blocks):
                    if layer_idx == 0:
                        # First layer is PointNetFPModule
                        if self.use_checkpoint and self.training:
                            output = checkpoint(layer, output, use_reentrant=False)
                        else:
                            output = layer(output)
                        features, coords, temb = output
                    else:
                        # Subsequent layers (e.g., PVConv)
                        input_tuple = (features, coords, temb)
                        if self.use_checkpoint and self.training:
                            output = checkpoint(layer, input_tuple, use_reentrant=False)
                        else:
                            output = layer(input_tuple)
                        features, coords, temb = output
            else:
                # Single module case (just PointNetFPModule, no attn_mask needed)
                if self.use_checkpoint and self.training:
                    features, coords, temb = checkpoint(fp_blocks, (jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb), use_reentrant=False)
                else:
                    features, coords, temb = fp_blocks((jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb))

            if debug_nan:
                print(f"[NaN TRACE] After FP layer {fp_idx}, features has NaN: {torch.isnan(features).any()}")
                if torch.isnan(features).any():
                    print(f"[NaN TRACE] *** NaN first appeared in FP layer {fp_idx} ***")

        out = self.classifier(features)

        if debug_nan:
            print(f"[NaN TRACE] After classifier, out has NaN: {torch.isnan(out).any()}")
            if torch.isnan(out).any():
                print(f"[NaN TRACE] *** NaN first appeared in classifier ***")
            print(f"{'='*80}\n")

        out = out.view(B, nT, nD, nP)


        return out, None



