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
