"""Windows-only foreground-safe game capture and manual key state."""
import sys
import ctypes
import ctypes.wintypes
import mss
import cv2
import numpy as np


VK = {'a': 0x41, 'd': 0x44, 'w': 0x57, 's': 0x53, 'space': 0x20,
      'f8': 0x77, 'end': 0x23}


def check_windows():
    if sys.platform != 'win32':
        raise RuntimeError('Screen/control tools require Windows. Training and tests work cross-platform.')


def pressed(name):
    check_windows()
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK[name]) & 0x8000)


def active_title():
    check_windows()
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def game_focused(config):
    title = active_title().casefold()
    return any(k.casefold() in title for k in config['game']['title_keywords'])


class WindowCapture:
    def __init__(self):
        check_windows()
        self.sct = mss.mss()

    def grab_game_window(self):
        """Capture foreground client rectangle (no menu/titlebar)."""
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        rect = ctypes.wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError('GetClientRect failed')
        point = ctypes.wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
            raise RuntimeError('ClientToScreen failed')
        bounds = dict(left=point.x, top=point.y,
                      width=rect.right-rect.left, height=rect.bottom-rect.top)
        if bounds['width'] <= 0 or bounds['height'] <= 0:
            raise RuntimeError('Invalid game client area')
        return cv2.cvtColor(np.asarray(self.sct.grab(bounds)), cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()
