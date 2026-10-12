import time
import threading


class KeyController:
    def __init__(self, keys=('a','d','w')):
        import pydirectinput
        self.api = pydirectinput
        self.state = {k:False for k in keys}
        self.lock = threading.RLock()

    def set_keys(self, desired):
        with self.lock:
            for key in self.state:
                want = key in desired
                if want != self.state[key]:
                    if want: self.api.keyDown(key)
                    else: self.api.keyUp(key)
                    self.state[key] = want

    def release(self):
        self.set_keys(set())


class KeyboardWatchdog:
    """Independent watchdog releases keys when main loop stops updating heartbeat.
    It cannot protect against OS hangs or loss of host input permissions.
    """
    def __init__(self, controller, timeout=0.5):
        self.controller = controller
        self.timeout = timeout
        self.heartbeat = time.monotonic()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def ping(self):
        self.heartbeat = time.monotonic()

    def _run(self):
        while self.running:
            if time.monotonic() - self.heartbeat > self.timeout:
                self.controller.release()
            time.sleep(0.05)

    def stop(self):
        self.running = False
        self.controller.release()
        self.thread.join(timeout=1)


class MouseController:
    """Hold left mouse button at calibrated client-relative steering points."""
    def __init__(self, config):
        import pydirectinput
        from asphalt_vls.windows import game_client_rect
        self.api = pydirectinput
        self.rect = game_client_rect
        self.config = config
        self.lock = threading.RLock()
        self.held = False
        self.direction = None

    def set_keys(self, desired):
        # Compatible with KeyboardWatchdog's release/set_keys interface.
        direction = 'left' if 'left' in desired else ('right' if 'right' in desired else None)
        with self.lock:
            if direction is None:
                self.release()
                return
            if self.held and direction == self.direction:
                return
            self.release()
            x0, y0, width, height = self.rect()
            fraction = self.config['game'].get(
                'mouse_left_point' if direction == 'left' else 'mouse_right_point',
                [0.25, 0.75] if direction == 'left' else [0.75, 0.75])
            fx, fy = map(float, fraction)
            if not (0 <= fx <= 1 and 0 <= fy <= 1):
                raise ValueError('Mouse point must use normalized coordinates [0,1]')
            self.api.moveTo(x0 + round(width * fx), y0 + round(height * fy))
            self.api.mouseDown(button='left')
            self.held = True
            self.direction = direction

    def release(self):
        with self.lock:
            if self.held:
                self.api.mouseUp(button='left')
                self.held = False
            self.direction = None
