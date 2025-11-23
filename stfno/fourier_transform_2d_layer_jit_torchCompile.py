import torch
import torch.nn as nn


class SpectralConv2d_jit_torchCompile(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_jit_torchCompile, self).__init__()
        """
        2D Fourier layer compatible with torch.compile.
        It uses view_as_real to perform complex multiplication using float32 types
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        # Initialize weights as real tensors with an extra dimension of size 2 (for real/imag parts)
        # This replaces dtype=torch.cfloat which causes compile errors.
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                2,
                dtype=torch.float32,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                2,
                dtype=torch.float32,
            )
        )

    def compl_mul2d_as_real(self, input, weights):
        # input: (batch, in_channel, x, y, 2)  [last dim 0=real, 1=imag]
        # weights: (in_channel, out_channel, x, y, 2)
        # returns: (batch, out_channel, x, y, 2)

        # Complex multiplication: (a+bi)(c+di) = (ac - bd) + (ad + bc)i

        # a = input[..., 0]
        # b = input[..., 1]
        # c = weights[..., 0]
        # d = weights[..., 1]

        # We perform 4 separate einsums on float32 data

        # ac
        ac = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 0])
        # bd
        bd = torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 1])
        # ad
        ad = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 1])
        # bc
        bc = torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 0])

        # Result real part: ac - bd
        # Result imag part: ad + bc
        return torch.stack([ac - bd, ad + bc], dim=-1)

    def forward(self, x):
        batchsize = x.shape[0]

        # 1. FFT (Returns complex)
        x_ft = torch.fft.rfft2(x)

        # 2. Convert to real view.
        # Shape becomes (Batch, In, X, Y, 2). Dtype is float32.
        x_ft = torch.view_as_real(x_ft)

        # 3. Prepare output buffer as float32
        # x.size(-1)//2 + 1 is the width in Fourier domain
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            2,
            dtype=torch.float32,
            device=x.device,
        )

        # 4. Perform multiplication using real arithmetic
        # Note: We slice the spatial dimensions, but keep the last dimension (size 2)
        out_ft[:, :, : self.modes1, : self.modes2, :] = self.compl_mul2d_as_real(
            x_ft[:, :, : self.modes1, : self.modes2, :], self.weights1
        )

        out_ft[:, :, -self.modes1 :, : self.modes2, :] = self.compl_mul2d_as_real(
            x_ft[:, :, -self.modes1 :, : self.modes2, :], self.weights2
        )

        # 5. Convert back to complex view for IFFT
        out_ft = torch.view_as_complex(out_ft)

        # 6. IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
