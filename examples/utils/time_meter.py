import sys
import time

from tqdm import trange

__all__ = [
    'TimeMeter',
    'timeit',
]


class TimeMeter:
    def __init__(self):
        self.n = self.avg = self._start = self._end = self.last_elapsed_time = 0

    def reset(self):
        self.n = self.avg = self._start = self._end = self.last_elapsed_time = 0

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.time()
        self.last_elapsed_time = self._end - self._start
        self.avg = (self.avg * self.n + self.last_elapsed_time) / (self.n + 1)
        self.n += 1

    @property
    def fps(self):
        return 1 / self.avg if self.avg else float('nan')


def timeit(func, args=(), kwargs={}, name=None, number=100, warmup=False):
    name = name if name is not None else func.__name__
    time_meter = TimeMeter()
    pbar = trange(number, desc=f'[{name}]', file=sys.stdout)
    for _ in pbar:
        if warmup and _ == 1:
            time_meter.reset()
        with time_meter:
            func(*args, **kwargs)
        pbar.set_description(f'[{name}] fps={time_meter.fps:.03f}')
