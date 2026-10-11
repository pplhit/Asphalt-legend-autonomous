"""Export convolution weights as channel-first flattened im2col matrices.
These are real-valued matrices; translating them into physical VLS encodings and
signed outputs requires a calibrated hardware implementation.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from asphalt_vls.model import TinyVLSCNN


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/track1/best.pt')
    p.add_argument('--out',default='runs/track1/vls_weights.npz')
    a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=TinyVLSCNN()
    model.load_state_dict(ckpt['model'])
    arrays={}
    for name in ('conv1','conv2','conv3'):
        layer=getattr(model,name)
        w=layer.weight.detach().numpy()
        arrays[name+'_weight']=w.reshape(w.shape[0],-1)
        arrays[name+'_bias']=layer.bias.detach().numpy()
        print(f'{name}: {arrays[name+"_weight"].shape[1]} inputs -> {w.shape[0]} outputs')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.out,**arrays)
    print('Exported',a.out)


if __name__=='__main__':
    main()
