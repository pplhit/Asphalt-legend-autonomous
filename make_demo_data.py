"""Generate SYNTHETIC samples for pipeline smoke tests ONLY.
Never train a real Asphalt driver using these images.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='demo_data')
    p.add_argument('--samples-per-class',type=int,default=40)
    a=p.parse_args()
    root=Path(a.out)
    rng=np.random.default_rng(7)
    for session in range(3):
        directory=root/'sessions'/f'synthetic_{session:02d}'
        (directory/'frames').mkdir(parents=True,exist_ok=True)
        with (directory/'labels.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['image','timestamp','steer','drift','nitro','throttle'])
            counter=0
            for steer in (-1,0,1):
                for n in range(a.samples_per_class):
                    im=rng.integers(35,75,size=(96,160),dtype=np.uint8)
                    curvature=int(steer*35)
                    cv2.line(im,(75,95),(78+curvature,0),200,3)
                    cv2.line(im,(110,95),(111+curvature,0),200,3)
                    filename=f'frames/{counter:06d}.png'
                    cv2.imwrite(str(directory/filename),im)
                    writer.writerow([filename,1000+session*1000+n*0.1+counter*0.1,steer,0,0,1])
                    counter+=1
    print('Generated synthetic smoke-test dataset:',root)


if __name__=='__main__':
    main()
