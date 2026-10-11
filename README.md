# Asphalt VLS AutoDrive

Lightweight three-convolution-layer CNN for fixed-track autonomous steering in the Windows PC version of Asphalt Legends, with a software im2col/VLS-compatible Conv3 backend and a placeholder for real SLM–VLS–CMOS hardware.

> Experimental prototype. A trained real-game checkpoint and physical optical backend are not included. Start with offline/private permitted game testing. The synthetic dataset generator is only a pipeline smoke test.

## Setup (Windows / Python 3.11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

```powershell
python make_demo_data.py --out demo_data
python train.py --data demo_data --out runs/smoke
python compare_backends.py --checkpoint runs/smoke/best.pt
python record.py --data data
python train.py --data data --out runs/track1
python drive.py --checkpoint runs/track1/best.pt --backend digital
python export_vls.py --checkpoint runs/track1/best.pt
```

In the Windows game, set **Manual Drive / TouchDrive OFF**. `F8` toggles recording or arms autonomous steering (depending on script), `End` stops. Game window must be focused. `drive.py` is disarmed by default. Check `config.yaml` for crop, performance and safety settings.

## Architecture

Input `1×96×160` → Conv(1→4,3×3,s2) → Conv(4→6,3×3,s2) → Conv(6→8,3×3,s2) → global average pool → MLP(8→16→3) → left / straight / right.

Each patch transforms 9→4, 36→6, and 54→8. `sim-vls` verifies the exact im2col matrix mapping for Conv3, but **does not execute on optical hardware**. Real deployment requires implementing `examples/my_vls_backend.py:infer` and calibrating signed outputs, throughput and SLM/CMOS synchronization.

## Files

- `record.py`: collect human gameplay frames and steering labels.
- `train.py`: train the small CNN from real demonstrations.
- `drive.py`: focus-gated Windows keyboard control, watchdog and timeout checks.
- `export_vls.py`: export flattened convolution matrices.
- `compare_backends.py`: compare ordinary Conv3 against software im2col.
- `asphalt_vls/`: CNN, dataset, window capture, control and preprocessing.
- `examples/my_vls_backend.py`: physical-backend integration stub.

No promise of successful autonomous completion is made without collecting track-specific data and measuring closed-loop performance.
