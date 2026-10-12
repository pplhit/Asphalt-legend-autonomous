"""Windows local-game auto-steering. NOT an unrestricted racing bot.

Starts DISARMED. F8 toggle control, END emergency stop. Only send keys while
Asphalt game window is foreground. OFFLINE / private permitted testing only.
"""
import argparse
import importlib
import time
import numpy as np
import torch
from asphalt_vls.config import load_config
from asphalt_vls.model import TinyVLSCNN, SimulatedVLSBackend
from asphalt_vls.vision import preprocess, to_tensor
from asphalt_vls.windows import WindowCapture, game_focused, pressed
from asphalt_vls.control import KeyController, MouseController, KeyboardWatchdog


def load_plugin(spec):
    module, name = spec.split(':', 1)
    return getattr(importlib.import_module(module),name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config.yaml')
    p.add_argument('--checkpoint',default='runs/track1/best.pt')
    p.add_argument('--backend',choices=['digital','sim-vls','hardware'],default='digital')
    p.add_argument('--vls-plugin',default='')
    p.add_argument('--confidence',type=float,default=0.53)
    p.add_argument('--control',choices=['keyboard','mouse'],default='keyboard')
    p.add_argument('--throttle',action='store_true',help='Hold W while model is armed; otherwise steering-only')
    p.add_argument('--preview',action='store_true',help='DO NOT use while controlling (takes game focus)')
    a=p.parse_args()
    if a.backend=='hardware' and not a.vls_plugin:
        p.error('--backend hardware requires --vls-plugin module:function')
    if a.preview:
        p.error('Preview UI intentionally disabled for focus-safe control; use record.py for visual calibration')
    cfg=load_config(a.config)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=TinyVLSCNN()
    model.load_state_dict(ckpt['model'])
    model.eval()
    sim_backend=SimulatedVLSBackend(model.conv3)
    plugin=load_plugin(a.vls_plugin) if a.backend=='hardware' else None

    def infer(x):
        with torch.inference_mode():
            if a.backend=='digital':
                logits=model(x)
            elif a.backend=='sim-vls':
                logits=model.forward_vls(x,sim_backend)
            else:
                def hardware(vectors):
                    arrays=vectors.detach().cpu().numpy().astype('float32')
                    output=np.asarray(plugin(arrays),dtype=np.float32)
                    if not np.isfinite(output).all():
                        raise ValueError('Optical backend returned non-finite outputs')
                    return torch.from_numpy(output)
                logits=model.forward_vls(x,hardware)
            probs=logits.softmax(-1).squeeze(0).cpu().numpy()
        return probs

    if a.control=='mouse' and a.throttle:
        p.error('--throttle is available only in keyboard mode')
    controller=MouseController(cfg) if a.control=='mouse' else KeyController(keys=(cfg['game']['steer_left_key'],cfg['game']['steer_right_key'],cfg['game']['throttle_key']))
    watchdog=KeyboardWatchdog(controller, timeout=0.5)
    cap=WindowCapture()
    armed=False; previous_f8=False
    last_action='not armed'
    period=1./cfg['game']['fps_control']
    print('DISARMED. Focus game window, Manual Drive/TouchDrive OFF. F8 arm/disarm. END STOP.')
    print('Backend:',a.backend, '| control:',a.control, '| throttle:',a.throttle)
    try:
        while True:
            if pressed('end'):
                print('END: emergency stop'); break
            focused=game_focused(cfg)
            f8=pressed('f8')
            if f8 and not previous_f8:
                if focused:
                    armed=not armed
                    print('ARMED' if armed else 'DISARMED')
                else:
                    print('F8 ignored: focus the game')
            previous_f8=f8
            if not focused:
                if armed: print('Focus lost, DISARMED')
                armed=False
                controller.release()
                watchdog.ping()
                time.sleep(0.05)
                continue
            if not armed:
                controller.release()
                watchdog.ping()
                time.sleep(0.03)
                continue
            start=time.monotonic()
            frame=cap.grab_game_window()
            capture_ts=time.monotonic()
            x=to_tensor(preprocess(frame,cfg))
            probs=infer(x)
            age=time.monotonic()-capture_ts
            elapsed=time.monotonic()-start
            if age > cfg['game']['max_frame_age_seconds'] or elapsed > cfg['game']['max_infer_seconds']:
                controller.release()
                armed=False
                print('TIMEOUT, DISARMED',f'{elapsed*1000:.0f}ms')
                watchdog.ping()
                continue
            choice=int(probs.argmax())
            confidence=float(probs[choice])
            # If classifier is uncertain, do NOT turn blindly.
            desired=set()
            if confidence >= a.confidence:
                if choice==0: desired.add('left' if a.control=='mouse' else cfg['game']['steer_left_key'])
                if choice==2: desired.add('right' if a.control=='mouse' else cfg['game']['steer_right_key'])
            if a.throttle:
                desired.add(cfg['game']['throttle_key'])
            # Max steering hold is effectively bounded by loop period and watchdog.
            controller.set_keys(desired)
            watchdog.ping()
            label=['LEFT','STRAIGHT','RIGHT'][choice]
            if label != last_action:
                print(f'{label:>8} p={np.round(probs,2)} infer+capture={elapsed*1000:.1f}ms')
                last_action=label
            time.sleep(max(0,period-(time.monotonic()-start)))
    except KeyboardInterrupt:
        print('CTRL-C: emergency stop')
    finally:
        controller.release()
        watchdog.stop()
        cap.close()


if __name__=='__main__':
    main()
