import csv
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset


class RaceDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv2.imread(str(row['path']), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(row['path'])
        x = torch.from_numpy(image.copy()).float()[None] / 255.
        label = int(row['steer']) + 1  # -1/0/+1 -> 0/1/2
        if self.augment:
            # Mild photometric jitter, never horizontal flip (would swap steering).
            x = (x * (0.90 + 0.20*torch.rand(())) + (torch.rand(())-0.5)*0.06).clamp(0, 1)
        return x, label


def read_rows(dataset_dir):
    root = Path(dataset_dir).resolve()
    rows = []
    for csv_file in sorted(root.glob('sessions/*/labels.csv')):
        with csv_file.open(newline='', encoding='utf-8') as f:
            for item in csv.DictReader(f):
                path = csv_file.parent / item['image']
                if path.exists():
                    rows.append(dict(path=path, steer=int(item['steer']),
                                     timestamp=float(item['timestamp']),
                                     session=csv_file.parent.name))
    if not rows:
        raise RuntimeError(f'No dataset found under {root}/sessions')
    return rows


def split_rows(rows, val_fraction, block_seconds, seed):
    """Block split; prefer full-session validation if >=2 sessions exist."""
    import random
    rng = random.Random(seed)
    sessions = sorted(set(r['session'] for r in rows))
    if len(sessions) >= 2:
        rng.shuffle(sessions)
        n = max(1, min(len(sessions)-1, round(len(sessions)*val_fraction)))
        val_sessions = set(sessions[:n])
        train = [r for r in rows if r['session'] not in val_sessions]
        val = [r for r in rows if r['session'] in val_sessions]
    else:
        groups = {}
        for row in rows:
            key = (row['session'], int(row['timestamp']//block_seconds))
            groups.setdefault(key, []).append(row)
        keys = sorted(groups)
        if len(keys) < 2:
            raise RuntimeError('Record >=2 sessions or >=2 time blocks for validation')
        rng.shuffle(keys)
        n = max(1, min(len(keys)-1, round(len(keys)*val_fraction)))
        val_keys = set(keys[:n])
        train = [r for k, grp in groups.items() if k not in val_keys for r in grp]
        val = [r for k, grp in groups.items() if k in val_keys for r in grp]
    return train, val
