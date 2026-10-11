import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from asphalt_vls.config import load_config
from asphalt_vls.dataset import RaceDataset, read_rows, split_rows
from asphalt_vls.model import TinyVLSCNN


def evaluate(model, loader, device):
    model.eval()
    conf = np.zeros((3, 3), dtype=np.int64)
    loss_sum = 0.
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            pred = logits.argmax(dim=1)
            for target, guess in zip(y.cpu().tolist(), pred.cpu().tolist()):
                conf[target, guess] += 1
    n = max(1, conf.sum())
    per_class = [float(v) for v in (conf.diagonal()/np.maximum(1, conf.sum(axis=1))).tolist()]
    return dict(loss=float(loss_sum/n), accuracy=float(conf.trace()/n),
                balanced_accuracy=float(np.mean(per_class)),
                class_recall=per_class, confusion=conf.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', default='runs/track1')
    args = parser.parse_args()
    cfg = load_config(args.config)
    opts = cfg['train']
    seed = opts['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data)
    train_rows, val_rows = split_rows(rows, opts['val_fraction'], opts['val_block_seconds'], seed)
    train_loader = DataLoader(RaceDataset(train_rows, augment=True), batch_size=opts['batch_size'], shuffle=True,
                              num_workers=opts['num_workers'])
    val_loader = DataLoader(RaceDataset(val_rows), batch_size=opts['batch_size'], shuffle=False,
                            num_workers=opts['num_workers'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyVLSCNN().to(device)
    frequencies = np.bincount([int(r['steer'])+1 for r in train_rows], minlength=3)
    if (frequencies == 0).any():
        raise RuntimeError(f'Missing steering class in training: {-1,0,1}, counts={frequencies.tolist()}. Collect more data.')
    weights = 1.0/np.sqrt(frequencies.astype(float))
    weights = torch.tensor(weights/weights.mean(), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts['learning_rate'], weight_decay=opts['weight_decay'])
    best = -1.
    for epoch in range(opts['epochs']):
        model.train()
        running = 0.
        for x,y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{opts["epochs"]}'):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits,y)
            loss.backward()
            optimizer.step()
            running += loss.item()*len(x)
        metrics = evaluate(model,val_loader,device)
        print(f'Epoch {epoch+1:03d}: train_loss={running/len(train_rows):.4f} '
              f'val_acc={metrics["accuracy"]:.3f} val_balanced={metrics["balanced_accuracy"]:.3f} '
              f'val_recall={np.round(metrics["class_recall"],3)}')
        if metrics['balanced_accuracy'] > best:
            best = metrics['balanced_accuracy']
            torch.save({'model':model.cpu().state_dict(), 'config':cfg,
                        'epoch':epoch, 'val_metrics':metrics}, out/'best.pt')
            model.to(device)
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Best checkpoint:', out/'best.pt', 'balanced_accuracy:',best)


if __name__=='__main__':
    main()
