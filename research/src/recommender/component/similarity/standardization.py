import numpy


class Standardizer:

    @staticmethod
    def compute(val):
        return 1 - val


class Log10Standardizer:

    @staticmethod
    def compute(val):
        if val <= 0:
            return 1
        res = 1 / numpy.log1p(val)
        if res > 1:
            res = 1
        return res