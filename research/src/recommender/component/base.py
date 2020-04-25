import time
import logging


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    timers = dict()

    def __init__(
            self,
            name=None,
            text="{}: Elapsed time: {:0.4f} seconds",
            log=print,
    ):
        self._start_time = None
        self.name = name
        self.text = text
        self.log = log

        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)

    def start(self):
        """Start a new timer"""

        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.log:
            self.log(self.text.format(self.name, elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time


class DataProcessor:

    def __init__(self, logger=None, timer=None):
        self.logger = logger
        self._timer = Timer(name=timer if timer is not None else type(self).__name__,
                            log=self.logger.debug if self.logger else print)

    def print(self, msg, level='print'):
        if not self.logger:
            print(msg)
        else:
            if level == 'error':
                self.logger.error(msg)
            elif level == 'debug':
                self.logger.debug(msg)
            else:
                self.logger.info(msg)

    def _process_inner(self, data):
        return data

    def _process_inner_with_time(self, data):
        self._timer.start()
        result = self._process_inner(data)
        self._timer.stop()
        return result

    def process(self, data, timer=False):
        process_func = self._process_inner if not timer else self._process_inner_with_time
        if isinstance(data, list):
            return [process_func(item) for item in data]
        return process_func(data)
