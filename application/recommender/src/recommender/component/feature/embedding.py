from abc import abstractmethod

import numpy
import time
import fasttext
import os

from recommender.component.base import DataProcessor


class Embedder(DataProcessor):
    """Base embedding computation class"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def embed(self, text):
        """Computes the embedding of the text

        Args:
            text (str): text to compute embedding for

        Returns:
            ndarray: embedding vector
        """
        pass

    def _process_inner(self, text):
        return self.embed(text)


class RandomEmbedder(Embedder):
    """Computes random embedding"""

    def __init__(self, model=300, seed=None, **kwargs):
        """
        Args:
            model (int): dimension of the result embedding
            seed (int): random machine initialization seed
        """
        super().__init__(**kwargs)
        self._model = model
        if seed is not None:
            self._seed = seed
            numpy.random.seed(self._seed)

    def embed(self, token):
        """Computes random embedding for token.

        Args:
            token (str): token to be embedded

        Returns:
            ndarray: random embedding
        """
        return numpy.random.rand(self._model)


class FastTextEmbedder(Embedder):
    """FastText embedder

    Provides FastText embedding computation
    """
    def __init__(self, model, **kwargs):
        """
        Args:
            model: may be either path (str) to the binary FastText model of following or the loaded model itself
        """
        super().__init__(**kwargs)
        if isinstance(model, str):
            path_to_model = os.path.abspath(model)
            self.print('Loading FastText model from: ' + path_to_model)
            start = time.time()
            model = fasttext.load_model(path_to_model)
            end = time.time()
            self.print('Model loaded in: ' + str(end - start) + ' sec')
        if not isinstance(model, fasttext.FastText._FastText):
            raise ValueError('Model must be ' + fasttext.FastText._FastText + ' or a path to fasttext binary model')
        self._model = model

    def embed(self, token):
        """Computes FastText embedding for token.

        The embedding algorithm is used the sentence vector from FastText model.
        See also: get_sentence_vector in https://github.com/facebookresearch/fastText/tree/master/python#model-object

        Args:
            token (str): token to be embedded

        Returns:
            ndarray: vector representation of the token
        """
        embedding = self._model.get_sentence_vector(token)
        return embedding


class EmbedderFactory:

    @staticmethod
    def build():
        pass