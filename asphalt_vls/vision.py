import cv2
import numpy as np


def preprocess(frame_bgr, config):
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = config['game']['crop']
    xa, xb = int(w*x0), int(w*x1)
    ya, yb = int(h*y0), int(h*y1)
    if xa >= xb or ya >= yb:
        raise ValueError('Invalid crop in config.yaml')
    crop = frame_bgr[ya:yb, xa:xb]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (config['game']['image_width'], config['game']['image_height']), interpolation=cv2.INTER_AREA)
    return gray


def to_tensor(gray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(gray)).float()[None, None] / 255.0
