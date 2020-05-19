import numpy


class Standardizer:
    """Basic inverse standardizer"""

    @staticmethod
    def compute(val: float) -> float:
        """Computes standardized value as the inversion capped by 1.

        Args:
            val (float): value to standardize

        Returns:
            float: standardized value
        """
        return min(1, 1 / val)


class CosineStandardizer(Standardizer):
    """Cosine standardizer"""

    @staticmethod
    def compute(val):
        """Computes standardized value from cosine distance.

        Args:
            val (float): value to standardize

        Returns:
            float: standardized value
        """
        return (2 - val) / 2


class Log1pStandardizer(Standardizer):
    """Log1p standardizer"""

    @staticmethod
    def compute(val):
        """Computes standardized value as natural logarithm of x+1 capped by 1.

        Args:
            val (float): value to standardize

        Returns:
            float: standardized value
        """
        if val <= 0:
            return 1
        res = 1 / numpy.log1p(val)
        return min(1, res)


class WeightedStandardizer(Standardizer):
    """Weighting standardizer

    Attributes:
        weight (float): weight used for standardization
    """

    def __init__(self, weight=0.1):
        self.weight = weight

    def compute(self, val):
        """Computes standardized value as multiplication by specified weight.

        Args:
            val (float): value to standardize

        Returns:
            float: standardized value
        """
        val *= self.weight
        return val


class UpperBoundStandardizer(Standardizer):
    """Upper bound standardizer

    Attributes:
        upper_bound (float): bound specifying max value
    """

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def compute(self, val):
        """Computes standardized value as division by specified upper bound.

        Args:
            val (float): value to standardize

        Returns:
            float: standardized value
        """
        val /= self.upper_bound
        return val


class RandomStandardizer(Standardizer):
    """Random standardizer

    Attributes:
        rate (float): the rate of random influence
    """

    def __init__(self, rate=1.0):
        self.rate = rate

    def compute(self, val):
        """Computes standardized value with a random bias.

        Args:
            val (float): value to standardize

        Returns:
            float: standardized value
        """
        val *= numpy.random.uniform(1 - self.rate, 1.0)
        return val
