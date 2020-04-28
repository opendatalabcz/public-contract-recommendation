import numpy


class Standardizer:

    @staticmethod
    def compute(val):
        return 1-val


class CosineStandardizer:

    @staticmethod
    def compute(val):
        return (2-val)/2


class Log10Standardizer(Standardizer):

    @staticmethod
    def compute(val):
        if val <= 0:
            return 1
        res = 1 / numpy.log1p(val)
        if res > 1:
            res = 1
        return res


class WeightedStandardizer(Standardizer):

    def __init__(self, weight=0.1):
        self.weight = weight

    def compute(self, val):
        val *= self.weight
        val += (1 - self.weight)
        return val