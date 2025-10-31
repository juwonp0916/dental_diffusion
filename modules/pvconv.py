import torch.nn as nn
import torch
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d

__all__ = ['PVConv', 'Attention', 'Swish', 'PVConvReLU']


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super(Attention, self).__init__()
        assert in_ch % num_groups == 0
        self.in_ch = in_ch
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)
        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)

        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)

    def forward(self, x, attn_mask=None, debug=False):
        """
        Args:
            x: Input features (B, C, ...)
            attn_mask: Optional attention mask (B,) - True for positions to MASK OUT (missing teeth)
                       Will be expanded to match attention matrix dimensions
            debug: If True, print detailed debugging information when NaN is detected
        """
        B, C = x.shape[:2]

        # CRITICAL: Check if input has NaN and handle it
        if torch.isnan(x).any():
            if debug:
                print(f"[Attention DEBUG] NaN in INPUT x: {torch.isnan(x).sum().item()} elements")
                print(f"  x shape: {x.shape}")
                print(f"  WARNING: Input to attention is corrupted! Replacing NaN with zeros.")
            # Replace NaN with zeros to prevent propagation
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        h = x

        # DEBUG: Check input for NaN
        if debug and torch.isnan(x).any():
            print(f"[Attention DEBUG] NaN in INPUT x: {torch.isnan(x).sum().item()} elements")
            print(f"  x shape: {x.shape}, x range: [{x.min():.4f}, {x.max():.4f}]")

        q = self.q(h).reshape(B, C, -1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1)

        # DEBUG: Check Q, K, V for NaN
        if debug and (torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any()):
            print(f"[Attention DEBUG] NaN in Q/K/V projections!")
            print(f"  Q has NaN: {torch.isnan(q).any()}, range: [{q.min():.4f}, {q.max():.4f}]")
            print(f"  K has NaN: {torch.isnan(k).any()}, range: [{k.min():.4f}, {k.max():.4f}]")
            print(f"  V has NaN: {torch.isnan(v).any()}, range: [{v.min():.4f}, {v.max():.4f}]")

        # Apply scaling for numerical stability (standard scaled dot-product attention)
        scale = (int(C) ** (-0.5))
        qk = torch.matmul(q.permute(0, 2, 1), k) * scale  # (B, seq_len, seq_len)

        # DEBUG: Check attention scores before masking
        if debug:
            print(f"[Attention DEBUG] Attention scores (before masking):")
            print(f"  qk shape: {qk.shape}, range: [{qk.min():.4f}, {qk.max():.4f}]")
            print(f"  qk has NaN: {torch.isnan(qk).any()}, has Inf: {torch.isinf(qk).any()}")
            if attn_mask is not None:
                print(f"  attn_mask shape: {attn_mask.shape}, num_masked: {attn_mask.sum().item()}/{attn_mask.numel()}")

        # Apply attention mask if provided
        # Compute softmax BEFORE masking to avoid NaN in backward pass
        # This is critical: masking with -inf before softmax causes softmax([-inf, -inf, ...]) = NaN
        # which breaks gradient computation even if we clean NaN in forward pass
        w = self.sm(qk)

        # DEBUG: Check softmax output before masking
        if debug:
            print(f"[Attention DEBUG] Softmax output (before masking):")
            print(f"  w shape: {w.shape}, range: [{w.min():.4f}, {w.max():.4f}]")
            print(f"  w has NaN: {torch.isnan(w).any()}, has Inf: {torch.isinf(w).any()}")

        # NOW apply mask by zeroing out attention weights for missing teeth
        # This approach avoids NaN entirely (no softmax of all -inf)
        if attn_mask is not None:
            # attn_mask shape: (B*28,) with True for missing teeth
            # w shape: (B*28, seq_len, seq_len)

            # Expand mask to cover all attention weights
            mask_expanded = attn_mask.view(-1, 1, 1)  # (B*28, 1, 1) -> broadcast to (B*28, seq_len, seq_len)

            # Zero out attention weights for missing teeth
            # This is mathematically sound and avoids gradient issues
            w = w.masked_fill(mask_expanded, 0.0)

            # DEBUG: Check after masking
            if debug:
                num_masked = (w == 0).sum().item()
                print(f"[Attention DEBUG] After masking: {num_masked} attention weights zeroed")
                print(f"  w has NaN: {torch.isnan(w).any()}, has Inf: {torch.isinf(w).any()}")

        # Sanity check - should never have NaN with this approach
        if debug and torch.isnan(w).any():
            print(f"[Attention DEBUG] UNEXPECTED NaN in attention weights!")
            print(f"  This should not happen with the new masking approach")

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B, C, *x.shape[2:])

        # DEBUG: Check after attention application
        if debug and torch.isnan(h).any():
            print(f"[Attention DEBUG] UNEXPECTED NaN in attention output h!")

        h = self.out(h)

        # DEBUG: Check after output projection
        if debug and torch.isnan(h).any():
            print(f"[Attention DEBUG] UNEXPECTED NaN after output projection!")

        # Residual connection
        x = h + x

        # DEBUG: Check after residual
        if debug and torch.isnan(x).any():
            print(f"[Attention DEBUG] UNEXPECTED NaN after residual connection!")

        x = self.nonlin(self.norm(x))

        # DEBUG: Check final output
        if debug and torch.isnan(x).any():
            print(f"[Attention DEBUG] UNEXPECTED NaN in FINAL OUTPUT!")
            print(f"  NaN count: {torch.isnan(x).sum().item()}/{x.numel()}")

        return x


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.use_attention = attention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Build voxel processing layers, extracting Attention for separate handling
        voxel_layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
                        nn.GroupNorm(num_groups=8, num_channels=out_channels),
                        Swish()]

        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []

        voxel_layers += [nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
                         nn.GroupNorm(num_groups=8, num_channels=out_channels)]

        # Store main voxel layers without Attention/Swish
        self.voxel_layers = nn.Sequential(*voxel_layers)

        # Store Attention separately so it can receive attn_mask, or use Swish
        if attention:
            self.voxel_attention = Attention(out_channels, 8)
        else:
            self.voxel_attention = Swish()

        # Store SE separately if needed
        if with_se:
            self.voxel_se = SE3d(out_channels, use_relu=with_se_relu)
        else:
            self.voxel_se = None

        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs, attn_mask=None, debug=False):
        features, coords, temb = inputs

        if debug:
            print(f"[PVConv DEBUG] Input - features NaN: {torch.isnan(features).any()}, coords NaN: {torch.isnan(coords).any()}")
            print(f"  coords stats: min={coords.min().item():.6f}, max={coords.max().item():.6f}, mean={coords.mean().item():.6f}")

        voxel_features, voxel_coords = self.voxelization(features, coords)

        if debug:
            print(f"[PVConv DEBUG] After voxelization - voxel_features NaN: {torch.isnan(voxel_features).any()}")
            if torch.isnan(voxel_features).any():
                print(f"  *** NaN in voxelization output! ***")
                return torch.zeros_like(features), coords, temb  # Emergency fallback

        # Apply voxel convolution layers
        voxel_features = self.voxel_layers(voxel_features)

        if debug:
            print(f"[PVConv DEBUG] After voxel_layers - NaN: {torch.isnan(voxel_features).any()}")

        # Apply attention with mask if using attention, otherwise apply activation
        if self.use_attention and attn_mask is not None:
            voxel_features = self.voxel_attention(voxel_features, attn_mask, debug=debug)
        else:
            voxel_features = self.voxel_attention(voxel_features)

        if debug:
            print(f"[PVConv DEBUG] After attention/activation - NaN: {torch.isnan(voxel_features).any()}")

        # Apply SE if present
        if self.voxel_se is not None:
            voxel_features = self.voxel_se(voxel_features)

        if debug:
            print(f"[PVConv DEBUG] After SE - NaN: {torch.isnan(voxel_features).any()}")

        # Devoxelize back to point locations
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        if debug:
            print(f"[PVConv DEBUG] After devoxelization - NaN: {torch.isnan(voxel_features).any()}")

        fused_features = voxel_features + self.point_features(features)

        if debug:
            print(f"[PVConv DEBUG] After fusion - NaN: {torch.isnan(fused_features).any()}")

        return fused_features, coords, temb


class PVConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False, leak=0.2,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.use_attention = attention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Build voxel processing layers, extracting Attention for separate handling
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(leak, True)
        ]
        voxel_layers += [nn.Dropout(dropout)] if dropout is not None else []
        voxel_layers += [
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels)
        ]

        # Store main voxel layers without Attention/LeakyReLU
        self.voxel_layers = nn.Sequential(*voxel_layers)

        # Store Attention separately so it can receive attn_mask, or use LeakyReLU
        if attention:
            self.voxel_attention = Attention(out_channels, 8)
        else:
            self.voxel_attention = nn.LeakyReLU(leak, True)

        # Store SE separately if needed
        if with_se:
            self.voxel_se = SE3d(out_channels, use_relu=with_se_relu)
        else:
            self.voxel_se = None

        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs, attn_mask=None):
        features, coords, temb = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)

        # Apply voxel convolution layers
        voxel_features = self.voxel_layers(voxel_features)

        # Apply attention with mask if using attention, otherwise apply activation
        if self.use_attention and attn_mask is not None:
            voxel_features = self.voxel_attention(voxel_features, attn_mask)
        else:
            voxel_features = self.voxel_attention(voxel_features)

        # Apply SE if present
        if self.voxel_se is not None:
            voxel_features = self.voxel_se(voxel_features)

        # Devoxelize back to point locations
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, temb
