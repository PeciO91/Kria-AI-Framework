import re
import subprocess
import threading
import time

from kria_ai.common.board.config import KV260


def read_power_mw(command=None):
    command = tuple(command or KV260.power_command)
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        return 0.0
    match = re.search(r"SOM total power\s+:\s+(\d+)\s+mW", output)
    return float(match.group(1)) if match else 0.0


class PowerMonitor(threading.Thread):
    def __init__(self, sample=read_power_mw, interval=0.2):
        super().__init__(daemon=True)
        self.sample = sample
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            value = self.sample() / 1000.0
            if value > 0:
                self.samples.append(value)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=max(1.0, self.interval * 2))

    def average(self, fallback=0.0):
        return sum(self.samples) / len(self.samples) if self.samples else fallback
