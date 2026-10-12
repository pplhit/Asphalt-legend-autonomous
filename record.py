"""Record human driving demonstrations. F8 toggles collection, END quits.

Run from a terminal, then focus the Asphalt game window.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import cv2
from asphalt_vls.config import load_config
from asphalt_vls.vision import preprocess
from asphalt_vls.windows import WindowCapture, game_focused, pressed, mouse_steering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--data', default='data')
    parser.add_argument('--control', choices=['keyboard','mouse'], default='keyboard')
    args = parser.parse_args()
    cfg = load_config(args.config)
    session = Path(args.data) / 'sessions' / datetime.now().strftime('%Y%m%d_%H%M%S')
    (session/'frames').mkdir(parents=True, exist_ok=True)
    interval = 1.0 / cfg['game']['fps_record']
    cap = WindowCapture()
    recording, last_f8, count = False, False, 0
    print('Focus game; F8 toggles recording; END quits. Control:', args.control)
    print('Start before race, record multiple complete laps and recovery maneuvers.')
    try:
        with (session/'labels.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image','timestamp','steer','drift','nitro','throttle'])
            next_time = time.monotonic()
            while True:
                if pressed('end'):
                    break
                f8 = pressed('f8')
                if f8 and not last_f8:
                    recording = not recording
                    print('RECORDING' if recording else 'PAUSED', 'samples:', count)
                last_f8 = f8
                if not recording or not game_focused(cfg):
                    time.sleep(0.02)
                    next_time = time.monotonic()
                    continue
                now = time.monotonic()
                if now < next_time:
                    time.sleep(min(next_time-now, 0.02))
                    continue
                next_time = now + interval
                frame = cap.grab_game_window()
                gray = preprocess(frame, cfg)
                if args.control == 'mouse':
                    steer = mouse_steering(cfg)
                else:
                    left = pressed(cfg['game']['steer_left_key'])
                    right = pressed(cfg['game']['steer_right_key'])
                    steer = int(right) - int(left)
                relative = f'frames/{count:07d}.png'
                cv2.imwrite(str(session / relative), gray)
                writer.writerow([relative, f'{now:.6f}', steer,
                                 int(pressed('s')), int(pressed('space')), int(pressed('w'))])
                if count % 50 == 0:
                    f.flush()
                    print('Saved', count, 'samples')
                count += 1
    finally:
        cap.close()
    print('Saved session:', session, 'frames:', count)


if __name__ == '__main__':
    main()
