import argparse
import torch
from asphalt_vls.model import TinyVLSCNN,SimulatedVLSBackend


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/track1/best.pt')
    a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=TinyVLSCNN().eval()
    model.load_state_dict(state['model'])
    x=torch.randn(2,1,96,160)
    with torch.inference_mode():
        a=model(x)
        b=model.forward_vls(x,SimulatedVLSBackend(model.conv3))
    err=(a-b).abs().max().item()
    print('Digital vs im2col software max error:',err)
    assert err < 1e-5


if __name__=='__main__':
    main()
