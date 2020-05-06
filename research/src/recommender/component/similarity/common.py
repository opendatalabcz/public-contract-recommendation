from random import random

import numpy
import pandas
import textdistance

from recommender.component.similarity.standardization import WeightedStandardizer, UpperBoundStandardizer
from recommender.component.similarity.vector_space import AggregatedItemSimilarityComputer
from recommender.component.similarity.geospatial import AggregatedLocalSimilarityComputer
from recommender.component.base import Component


class SimilarityMachine:
    _text = None
    _reference = None
    _model = None

    def __init__(self, model=None):
        self._model = model

    def init_case(self, text, reference):
        self._text = text
        self._reference = reference

    def preprocess(self):
        None

    def compute(self, text, reference):
        self.init_case(text, reference)
        self.preprocess()
        return self._inner_compute()

    def _inner_compute(self):
        return 0.5


class RandomSimilarityMachine(SimilarityMachine):

    def _inner_compute(self):
        return random.random()


class JaccardSimilarityMachine(SimilarityMachine):

    def preprocess(self):
        if isinstance(self._text, str):
            self._text = self._text.split()
        if isinstance(self._reference, str):
            self._reference = self._reference.split()

    def _inner_compute(self):
        return textdistance.jaccard(self._text, self._reference)


class ComplexSimilarityComputer(Component):

    def __init__(self, df_contracts, similarity_computers=None, **kwargs):
        super().__init__(**kwargs)
        self._df_contracts = df_contracts
        self._similarity_computers = similarity_computers if similarity_computers is not None \
            else [
            (AggregatedItemSimilarityComputer(self._df_contracts), WeightedStandardizer(1)),
            (AggregatedLocalSimilarityComputer(self._df_contracts), WeightedStandardizer(0.1))
        ]

    def compute_most_similar(self, df_query, num_results=1):
        similarities = {}
        for similarity_computer, standardizer in self._similarity_computers:
            most_similar = similarity_computer.compute_most_similar(df_query, num_results=numpy.iinfo(numpy.int32).max)
            for qid in most_similar:
                for contract_res in most_similar[qid]:
                    cid = contract_res['contract_id']
                    qs = similarities.get(qid, {})
                    qs[cid] = qs.get(cid, 0) + standardizer.compute(contract_res['similarity'])
                    similarities[qid] = qs
        if not similarities:
            return {}
        df_similar_contracts = pandas.DataFrame.from_dict(similarities, orient='index')
        s_similar_contracts = df_similar_contracts.stack()
        s_similar_contracts = s_similar_contracts.rename('similarity')
        standardizer = UpperBoundStandardizer(sum([sc[1].weight for sc in self._similarity_computers]))
        s_similar_contracts = s_similar_contracts.apply(standardizer.compute)
        s_similar_contracts = s_similar_contracts.rename_axis(['query_id', 'contract_id'])
        df_nlargest = s_similar_contracts.groupby(level=0).nlargest(num_results).reset_index(drop=True,
                                                                                             level=1).reset_index()
        result = {query: g[['contract_id', 'similarity']].to_dict('record')
                  for query, g in df_nlargest.groupby('query_id')}
        return result
