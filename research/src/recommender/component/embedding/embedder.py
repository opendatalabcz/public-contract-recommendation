from abc import abstractmethod

import numpy
import time
import fasttext
import os

from recommender.component.base import DataProcessor


class Embedder(DataProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def embed(self, text):
        pass

    def _process_inner(self, text):
        return self.embed(text)


class RandomEmbedder(Embedder):

    def __init__(self, model=300, seed=None, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        if seed is not None:
            self._seed = seed
            numpy.random.seed(self._seed)

    def embed(self, token):
        return numpy.random.rand(self._model)


class FastTextEmbedder(Embedder):

    def __init__(self, model, **kwargs):
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
        embedding = self._model.get_sentence_vector(token)
        return embedding


class EmbedderFactory:

    @staticmethod
    def build():
        pass