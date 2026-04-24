"""PyTorch CNN autoencoders (Models 3 and 4).

Model 3: 1 conv layer + 1 FC layer (encoder), mirrored decoder
Model 4: 2 conv layers + 2 FC layers (encoder), mirrored decoder

Both are parameterised by the number of conv and FC layers so the same
class handles both architectures -- satisfying the assignment requirement
of not writing separate code per architecture.

Input to the network is a 2-D spectrogram stored as a flat vector (shape: N, D).
Internally each batch is reshaped to (N, 1, F, T) for convolution.

AdaptiveAvgPool2d is applied after the conv stack to collapse the spatial
dimensions to a fixed pool_size before the FC layers. Without it the FC
input would be hundreds of thousands of values and the CNNs would have more
parameters than the MLPs, defeating the purpose. The decoder mirrors this
with nn.Upsample before the ConvTranspose stack.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with configurable depth.

    Parameters
    ----------
    input_shape : (int, int)
        (freq_bins, time_frames) -- the 2-D spectrogram shape before flattening.
    conv_channels : list of int
        Number of output channels for each conv layer in the encoder.
        len == 1 -> Model 3, len == 2 -> Model 4.
    fc_dims : list of int
        Hidden FC dimensions between the pooled conv output and the bottleneck.
        len == 1 -> Model 3, len == 2 -> Model 4.
    bottleneck_dim : int
        Size of the latent code.
    activation : str
        'relu', 'tanh', or 'sigmoid'.
    kernel_size : int
        Convolution kernel size (same for all layers).
    stride : int
        Convolution stride (controls spatial downsampling in encoder).
    pool_size : (int, int) or None
        Target spatial size after AdaptiveAvgPool2d in the encoder.
        Keeps FC parameter count independent of input resolution.
        The decoder uses nn.Upsample to reverse this before ConvTranspose.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_channels: List[int],
        fc_dims: List[int],
        bottleneck_dim: int,
        activation: str = "relu",
        kernel_size: int = 3,
        stride: int = 2,
        pool_size: Optional[Tuple[int, int]] = (4, 4),
    ):
        super().__init__()

        self.input_shape    = input_shape
        self.conv_channels  = conv_channels
        self.fc_dims        = fc_dims
        self.bottleneck_dim = bottleneck_dim
        self.kernel_size    = kernel_size
        self.stride         = stride

        _acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        Act = _acts[activation]

        # ── Encoder conv stack ────────────────────────────────────────────────
        enc_conv_layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            enc_conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2),
                Act(),
            ]
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc_conv_layers)

        # Pool to a fixed spatial size so FC input dim is independent of resolution.
        # Without this, FC(flatten -> fc_dims[0]) would be huge for large spectrograms.
        self.enc_pool = nn.AdaptiveAvgPool2d(pool_size) if pool_size else nn.Identity()

        # Compute shapes via a dry run (no gradients needed)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            conv_out  = self.enc_conv(dummy)
            self.pre_pool_shape = conv_out.shape[1:]      # (C, H_conv, W_conv)
            pooled    = self.enc_pool(conv_out)
            self.conv_out_shape = pooled.shape[1:]        # (C, pool_h, pool_w)
            flat_dim  = pooled.numel()

        # Mirror the pool with an upsample in the decoder
        if pool_size:
            spatial = (self.pre_pool_shape[1], self.pre_pool_shape[2])
            self.dec_upsample: nn.Module = nn.Upsample(size=spatial, mode="bilinear", align_corners=False)
        else:
            self.dec_upsample = nn.Identity()

        # ── Encoder FC stack ──────────────────────────────────────────────────
        enc_fc_layers: List[nn.Module] = []
        in_dim = flat_dim
        for h_dim in fc_dims:
            enc_fc_layers += [nn.Linear(in_dim, h_dim), Act()]
            in_dim = h_dim
        enc_fc_layers += [nn.Linear(in_dim, bottleneck_dim)]  # no activation on bottleneck
        self.enc_fc = nn.Sequential(*enc_fc_layers)

        # ── Decoder FC stack (mirror of encoder FC) ───────────────────────────
        dec_fc_layers: List[nn.Module] = []
        in_dim = bottleneck_dim
        for h_dim in reversed(fc_dims):
            dec_fc_layers += [nn.Linear(in_dim, h_dim), Act()]
            in_dim = h_dim
        dec_fc_layers += [nn.Linear(in_dim, flat_dim), Act()]
        self.dec_fc = nn.Sequential(*dec_fc_layers)

        # ── Decoder conv stack (mirror using ConvTranspose2d) ─────────────────
        # Encoder goes: [1, ch0, ch1, ...].  Decoder reverses: [ch_last, ..., ch0, 1].
        all_channels = [1] + conv_channels
        dec_conv_layers: List[nn.Module] = []
        n_conv = len(conv_channels)
        for i in range(n_conv - 1, -1, -1):
            in_ch  = all_channels[i + 1]
            out_ch = all_channels[i]
            is_last = (i == 0)
            dec_conv_layers.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch, kernel_size, stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                )
            )
            if not is_last:
                dec_conv_layers.append(Act())
            # last layer: linear output, no activation
        self.dec_conv = nn.Sequential(*dec_conv_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, F*T) -> latent: (N, bottleneck_dim)."""
        N = x.size(0)
        h = x.view(N, 1, *self.input_shape)
        h = self.enc_conv(h)
        h = self.enc_pool(h)
        h = h.view(N, -1)
        return self.enc_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, bottleneck_dim) -> x_hat: (N, F*T)."""
        N = z.size(0)
        h = self.dec_fc(z)
        h = h.view(N, *self.conv_out_shape)
        h = self.dec_upsample(h)                              # restore conv spatial dims
        h = self.dec_conv(h)
        # Crop to exactly input_shape (ConvTranspose2d can be off by 1)
        h = h[:, :, : self.input_shape[0], : self.input_shape[1]]
        return h.reshape(N, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE, shape (N,). No gradient."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
