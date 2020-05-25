import time
import logging


def create_logger(name, level, handlers) -> logging.Logger:
    """Creates a new logger with name, level and handlers.

    Args:
        name (str): name of the logger
        level (int|str): specification of logging level
        handlers (list): list of logging handlers

    Returns:
        logger: new logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers
    return logger


class Component:
    """Base component providing logging and time measure functions.

    All other project components inherits from this class to support basic functions.

    Attributes:
        logger: a logger object to be copied for new instance logger
    """
    def __init__(self, logger=None, timer=None):
        """
        Args:
            logger (Logger): a logger object to be copied for new instance
            timer (str): name of the timer
        """
        self.logger = create_logger(self.__class__.__name__, logger.level, logger.handlers) if logger else None
        self._timer = Timer(name=timer if timer is not None else type(self).__name__,
                            log=self.logger.debug if self.logger else print)

    def print(self, msg, level='print'):
        """Method for printing or logging.

        If member logger is specified, this method uses it to log the message,
        otherwise prints to std.output.

        Args:
            msg: message to be printed
            level (int|str): specification of logging level
        """
        if not self.logger:
            print(msg)
        else:
            if level == 'error':
                self.logger.error(msg)
            elif level == 'debug':
                self.logger.debug(msg)
            else:
                self.logger.info(msg)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    """Class providing time measure interface.

    Keeps accumulated times for each named timer.

    Attributes:
        timers (dict of str: int):  static accumulation of elapsed time of named timers
        name (str): name of current timer
        text (str): string format for printing
        log (function): printing/logging function
    """
    timers = dict()

    def __init__(
            self,
            name=None,
            text="{}: {} Elapsed time: {:0.4f} seconds",
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

    def stop(self, msg=None):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.log:
            self.log(self.text.format(self.name, msg, elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time


class DataProcessor(Component):
    """Base data processing component.

    Provides interface to process both single item or a collection of items.
    """

    def _process_inner(self, data):
        return data

    def _process_inner_with_time(self, data):
        self._timer.start()
        result = self._process_inner(data)
        self._timer.stop()
        return result

    def process(self, data, timer=False):
        """Main processing method.

        Recognizes whether to process all the collection or a single item.

        Args:
            data: in case of list the processing function is done item by item
            timer (bool): if true, the processing time is measured

        Returns:
            single result or a list of results
        """
        process_func = self._process_inner if not timer else self._process_inner_with_time
        if isinstance(data, list):
            return [process_func(item) for item in data]
        return process_func(data)
