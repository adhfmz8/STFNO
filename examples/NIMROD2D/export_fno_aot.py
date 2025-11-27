# export_fno_aot.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch_mlir
import torch.nn as nn
import torch.nn.functional as F
import types


# ==============================================================================
# PATCH 1: InstanceNorm2d (Fixes 'aten.instance_norm' error)
# ==============================================================================
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

        x_res = F.interpolate(
            x, size=(x.size(-2), target_width), mode="bilinear", align_corners=False
        )

        x_ft = torch.stack([x_res, x_res], dim=-1)  # Shape: (B, C, H, W//2+1, 2)

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

        x = F.interpolate(
            out_real,
            size=(x.size(-2), x.size(-1)),
            mode="bilinear",
            align_corners=False,
        )

        return x


original_module.SpectralConv2d_jit_torchCompile = SafeSpectralConv2d
print("Patched SpectralConv2d to use Safe (No-Complex) Implementation.")


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
