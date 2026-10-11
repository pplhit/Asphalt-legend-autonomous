import torch
from asphalt_vls.model import TinyVLSCNN,SimulatedVLSBackend


def test_shape_and_vls_parity():
    torch.manual_seed(10)
    model=TinyVLSCNN().eval()
    x=torch.randn(2,1,96,160)
    with torch.no_grad():
        baseline=model(x)
        optical_sim=model.forward_vls(x,SimulatedVLSBackend(model.conv3))
    assert baseline.shape == (2,3)
    assert torch.allclose(baseline,optical_sim,atol=1e-6)


def test_vector_dimensions():
    model=TinyVLSCNN()
    assert [getattr(model,k).weight[0].numel() for k in ('conv1','conv2','conv3')] == [9,36,54]
