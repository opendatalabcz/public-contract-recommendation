import numpy


class Standardizer:

    @staticmethod
    def compute(val):
        return min(1, 1/val)


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
        return min(1, res)


class WeightedStandardizer(Standardizer):

    def __init__(self, weight=0.1):
        self.weight = weight

    def compute(self, val):
        val *= self.weight
        return val


class InverseWeightedStandardizer(Standardizer):

    def __init__(self, weight=0.1):
        self.weight = weight

    def compute(self, val):
        val *= self.weight
        val += (1 - self.weight)
        return val


class UpperBoundStandardizer(Standardizer):

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def compute(self, val):
        val /= self.upper_bound
        return val