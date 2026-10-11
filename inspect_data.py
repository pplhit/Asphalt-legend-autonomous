"""Show a mosaic of recorded preprocessed training inputs."""
import argparse
import cv2
import numpy as np
from asphalt_vls.dataset import read_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data')
    p.add_argument('--out',default='dataset_preview.png')
    a=p.parse_args()
    rows=read_rows(a.data)
    indexes=np.linspace(0,len(rows)-1,num=min(24,len(rows)),dtype=int)
    frames=[]
    for i in indexes:
        r=rows[i]
        img=cv2.imread(str(r['path']),cv2.IMREAD_GRAYSCALE)
        img=cv2.cvtColor(cv2.resize(img,(320,192)),cv2.COLOR_GRAY2BGR)
        cv2.putText(img,f'steer={r["steer"]:+d}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
        frames.append(img)
    pad=np.zeros_like(frames[0])
    while len(frames)%4: frames.append(pad.copy())
    mosaic=np.vstack([np.hstack(frames[i:i+4]) for i in range(0,len(frames),4)])
    cv2.imwrite(a.out,mosaic)
    print('Wrote',a.out)


if __name__=='__main__':
    main()
