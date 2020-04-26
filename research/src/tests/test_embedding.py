import pytest
import numpy

from recommender.component.feature.embedding import RandomEmbedder, FastTextEmbedder


def test_random_embedder():
    embedder = RandomEmbedder(seed=42)
    token = 'koloběžka'
    result = embedder.embed(token)
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 300
    assert result[0] == 0.3745401188473625


def test_random_embedder_list():
    embedder = RandomEmbedder(seed=42)
    tokens = ['koloběžka', 'bicykl']
    result = embedder.process(tokens)
    assert isinstance(result, list)
    assert len(result) == 2
    assert numpy.isclose(result[0][0], 0.3745401188473625)


def test_fasttext_embedder(context):
    embedder = context.get_fasttext_embedder()
    token = 'koloběžka'
    result = embedder.process(token)
    assert isinstance(result, numpy.ndarray)
    assert len(result) == 300
    assert numpy.isclose(result[0], 0.042146936)
