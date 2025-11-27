# export_fno_aot.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch_mlir
import torch.nn as nn
import torch.nn.functional as F
import types


class CompileFriendlyInstanceNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[-2, -1], keepdim=True)
        var = ((x - mean).pow(2)).mean(dim=[-2, -1], keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


# Apply Patch 1
torch.nn.InstanceNorm2d = CompileFriendlyInstanceNorm2d


def compile_friendly_interpolate_bilinear(x, size):
    """
    Performs bilinear interpolation (align_corners=False) using basic PyTorch
    ops (indexing, arithmetic) that decompose cleanly to MLIR Linalg.
    """
    B, C, H, W = x.shape
    target_h, target_w = size

    # 1. Calculate source coordinates (align_corners=False logic)
    # formula: src = (dst + 0.5) * (src_len / dst_len) - 0.5
    y_coords = torch.arange(target_h, device=x.device, dtype=torch.float32)
    x_coords = torch.arange(target_w, device=x.device, dtype=torch.float32)

    y_src = (y_coords + 0.5) * (H / target_h) - 0.5
    x_src = (x_coords + 0.5) * (W / target_w) - 0.5

    # Clamp to ensure we don't index out of bounds
    y_src = torch.clamp(y_src, 0, H - 1)
    x_src = torch.clamp(x_src, 0, W - 1)

    # 2. Get integer indices for the four corners
    y0 = torch.floor(y_src).long()
    x0 = torch.floor(x_src).long()
    y1 = torch.clamp(y0 + 1, 0, H - 1)
    x1 = torch.clamp(x0 + 1, 0, W - 1)

    # 3. Calculate interpolation weights
    # Reshape weights for broadcasting: (1, 1, H_out, 1) and (1, 1, 1, W_out)
    y_weight = (y_src - y0).view(1, 1, -1, 1)
    x_weight = (x_src - x0).view(1, 1, 1, -1)

    # 4. Gather pixels using advanced indexing
    # We slice dimensions. x[..., y0, :] selects rows, result is (B, C, H_out, W)
    # Then we slice columns from that result.

    # Ia: Top-Left (y0, x0)
    Ia = x[..., y0, :][..., x0]
    # Ib: Top-Right (y0, x1)
    Ib = x[..., y0, :][..., x1]
    # Ic: Bottom-Left (y1, x0)
    Ic = x[..., y1, :][..., x0]
    # Id: Bottom-Right (y1, x1)
    Id = x[..., y1, :][..., x1]

    # 5. Interpolate
    # Horizontal interpolation
    top = Ia + (Ib - Ia) * x_weight
    bottom = Ic + (Id - Ic) * x_weight

    # Vertical interpolation
    result = top + (bottom - top) * y_weight

    return result


import stfno.fourier_transform_2d_layer_jit_torchCompile as original_module


class SafeSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)

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
        ac = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 0])
        bd = torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 1])
        ad = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 1])
        bc = torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 0])
        return torch.stack([ac - bd, ad + bc], dim=-1)

    def forward(self, x):
        batchsize = x.shape[0]

        target_width = x.size(-1) // 2 + 1

        x_res = compile_friendly_interpolate_bilinear(
            x, size=(x.size(-2), target_width)
        )

        x_ft = torch.stack([x_res, x_res], dim=-1)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            target_width,
            2,
            dtype=torch.float32,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2, :] = self.compl_mul2d_as_real(
            x_ft[:, :, : self.modes1, : self.modes2, :], self.weights1
        )

        out_ft[:, :, -self.modes1 :, : self.modes2, :] = self.compl_mul2d_as_real(
            x_ft[:, :, -self.modes1 :, : self.modes2, :], self.weights2
        )

        out_real = out_ft[..., 0] - out_ft[..., 1]

        x = compile_friendly_interpolate_bilinear(
            out_real, size=(x.size(-2), x.size(-1))
        )

        return x


original_module.SpectralConv2d_jit_torchCompile = SafeSpectralConv2d
print(
    "Patched SpectralConv2d to use Safe Implementation (No-Complex + Manual Interpolation)."
)


from stfno.stfno_2d import FNO2d_global

# 1. Configuration
S = 64
modes = 12
width = 20
T_in = 1
T_out = 1
total_vector_a_elements_i = 1
total_vector_u_elements_i = 1
number_of_layers = 4
input_parameter_order = [[0]]
mWidth_input_parameters = [1]
nWidth_output_parameters = [1]

# 2. Instantiate
print("Instantiating Model...")
model = FNO2d_global(
    modes1=modes,
    modes2=modes,
    width=width,
    T_in=T_in,
    total_vector_a_elements_i=total_vector_a_elements_i,
    T=T_out,
    total_vector_u_elements_i=total_vector_u_elements_i,
    number_of_layers=number_of_layers,
    input_parameter_order0=input_parameter_order,
    mWidth_input_parameters0=mWidth_input_parameters,
    nWidth_output_parameters0=nWidth_output_parameters,
    if_model_jit_torchCompile=True,
)
model.eval()

# 3. Dummy Input
input_channels = (T_in * total_vector_a_elements_i) + 2
dummy_input = torch.randn(1, S, S, input_channels)
print(f"Input Shape: {dummy_input.shape}")

# 4. Compile
print("Compiling to MLIR (Linalg on Tensors)...")
module = torch_mlir.compile(
    model,
    example_args=[dummy_input],
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True,
)

# 5. Save
with open("stfno.mlir", "w") as f:
    f.write(str(module))

print("Success! Saved stfno.mlir")
