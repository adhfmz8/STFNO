# export_fno_aot.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch_mlir
import torch.nn as nn


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
        # Note: The user's code uses affine=False (default).
        # If affine=True were used, we would need to handle self.weight/bias here.

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        # Calculate mean and variance over spatial dims (-2, -1)
        mean = x.mean(dim=[-2, -1], keepdim=True)
        # Var = Mean((x - mean)^2)
        var = ((x - mean).pow(2)).mean(dim=[-2, -1], keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


print("Patching InstanceNorm2d for compilation compatibility...")
torch.nn.InstanceNorm2d = CompileFriendlyInstanceNorm2d

from stfno.stfno_2d import FNO2d_global

# --- 1. CONFIGURATION ---
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

# --- 2. INSTANTIATE ---
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

# --- 3. CREATE DUMMY INPUT ---
# Channels = (T_in * total_vector_a_elements_i) + 2 (grid)
input_channels = (T_in * total_vector_a_elements_i) + 2
dummy_input = torch.randn(1, S, S, input_channels)
print(f"Input Shape: {dummy_input.shape}")

# --- 4. COMPILE TO MLIR ---
print("Compiling to MLIR (Linalg on Tensors)...")

module = torch_mlir.compile(
    model,
    example_args=[dummy_input],
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True,
)

# --- 5. SAVE ---
with open("stfno.mlir", "w") as f:
    f.write(str(module))

print("Success! Saved stfno.mlir")
