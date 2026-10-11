"""3-layer CNN; each convolution receptive-field dot product has <=64 inputs.

conv1: 3x3x1=9 -> 4
conv2: 3x3x4=36 -> 6
conv3: 3x3x6=54 -> 8.  Last one may be replaced by a VLS backend.
"""
import torch
from torch import nn
import torch.nn.functional as F


class TinyVLSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 6, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(6, 8, 3, stride=2, padding=1)
        self.fc = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.conv3(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def forward_vls(self, x, backend):
        """Replace conv3 only. Backend(mat [N,54]) -> [N,8] real outputs.
        N is batch*spatial positions. Optical hardware must provide actual batched
        or multiplexed throughput; Python call batching does not create it.
        """
        h = self.features(x)
        patches = F.unfold(h, kernel_size=3, stride=2, padding=1)
        b, _, locations = patches.shape
        vectors = patches.transpose(1, 2).reshape(-1, 54)
        outputs = backend(vectors)
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.as_tensor(outputs, dtype=vectors.dtype, device=vectors.device)
        if outputs.shape != (b * locations, 8):
            raise ValueError(f'VLS backend must return {(b * locations, 8)}, got {tuple(outputs.shape)}')
        out = outputs.reshape(b, locations, 8).transpose(1, 2)
        out = F.relu(out).mean(dim=-1)
        return self.fc(out)


class SimulatedVLSBackend:
    """Exact mathematical conv3 matrix multiplication, NOT optical hardware."""
    def __init__(self, conv):
        self.weight = conv.weight.detach().reshape(8, 54)
        self.bias = conv.bias.detach()

    def __call__(self, vectors):
        return F.linear(vectors, self.weight.to(vectors.device), self.bias.to(vectors.device))
