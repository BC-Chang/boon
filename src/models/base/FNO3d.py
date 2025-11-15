import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# This code is borrowed from FNO git repository: https://github.com/zongyi-li/fourier_neural_operator


################################################################
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1.0 / (in_channels * out_channels)

        # Use REAL parameters for real/imag parts (no complex params!)
        def init_parts():
            shape = (in_channels, out_channels, self.modes1, self.modes2, self.modes3)
            real = nn.Parameter(self.scale * torch.randn(*shape, dtype=torch.float32))
            imag = nn.Parameter(self.scale * torch.randn(*shape, dtype=torch.float32))
            return real, imag

        self.w1_real, self.w1_imag = init_parts()
        self.w2_real, self.w2_imag = init_parts()
        self.w3_real, self.w3_imag = init_parts()
        self.w4_real, self.w4_imag = init_parts()

    def compl_mul3d(self, input, weights):
        # (B, C_in, X, Y, F_T) x (C_in, C_out, X, Y, F_T) -> (B, C_out, X, Y, F_T)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def _complex_weights_fp32(self):
        # Build complex64 weights from fp32 real/imag parts (inside autocast-disabled region)
        w1 = torch.complex(self.w1_real.float(), self.w1_imag.float())
        w2 = torch.complex(self.w2_real.float(), self.w2_imag.float())
        w3 = torch.complex(self.w3_real.float(), self.w3_imag.float())
        w4 = torch.complex(self.w4_real.float(), self.w4_imag.float())
        return w1, w2, w3, w4

    def forward(self, x):
        B, C_in, X, Y, T = x.shape

        # FFTs and complex ops in FP32 to avoid cuFFT FP16 restrictions
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x.float()
            x_ft = torch.fft.rfftn(x32, dim=(-3, -2, -1))  # complex64

            out_ft = torch.zeros(B, self.out_channels, X, Y, T//2 + 1,
                                 dtype=torch.complex64, device=x.device)

            w1, w2, w3, w4 = self._complex_weights_fp32()

            out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], w1)
            out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], w2)
            out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], w3)
            out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], w4)

            y32 = torch.fft.irfftn(out_ft, s=(X, Y, T))  # float32

        return y32.to(dtype=x.dtype)
'''
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        with torch.amp.autocast('cuda', enabled=False):
            x32  = x.float()
            #Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfftn(x32, dim=[-3,-2,-1])

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.complex64, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
            out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
            out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

            #Return to physical space
            y32 = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
            
        return y32.to(dtype=x.dtype)
'''
class FNO3d(nn.Module):
    def __init__(self, 
        modes1, 
        modes2, 
        modes3, 
        width,
        lb=0,
        ub=1):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.lb = lb # lower value of the domain
        self.ub = ub # upper value of the domain
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(self.lb, self.ub, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(self.lb, self.ub, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
