from random import random

import numpy
import pandas
import textdistance

from recommender.component.similarity.standardization import WeightedStandardizer, UpperBoundStandardizer
from recommender.component.similarity.vector_space import AggregatedItemSimilarityComputer
from recommender.component.similarity.geospatial import AggregatedLocalSimilarityComputer
from recommender.component.base import Component


class SimilarityMachine:
    """Base similarity computation class"""
    _text = None
    _reference = None
    _model = None

    def __init__(self, model=None):
        self._model = model

    def _init_case(self, text, reference):
        self._text = text
        self._reference = reference

    def _preprocess(self):
        None

    def _inner_compute(self):
        return 0.5

    def compute(self, text, reference) -> float:
        """Computes the similarity of the text and reference

        Args:
            text (str): text to compute similarity for
            reference (str): reference text to compute similarity against

        Returns:
            float: the similarity ratio
        """
        self._init_case(text, reference)
        self._preprocess()
        return self._inner_compute()


class RandomSimilarityMachine(SimilarityMachine):
    """Computes random 'similarity'"""

    def _inner_compute(self):
        return random.random()


class JaccardSimilarityMachine(SimilarityMachine):
    """Computes Jaccard index as a similarity measure for the texts"""

    def _preprocess(self):
        if isinstance(self._text, str):
            self._text = self._text.split()
        if isinstance(self._reference, str):
            self._reference = self._reference.split()

    def _inner_compute(self):
        return textdistance.jaccard(self._text, self._reference)


class ComplexSimilarityComputer(Component):
    """Computes complex similarities of queries and contracts.

    Using specific similarity computers to compute the similarities between two pandas DataFrame.
    Results of all similarity computers are aggregated with weighted average.
    """

    def __init__(self, df_contracts, similarity_computers=None, **kwargs):
        """
        Args:
            df_contracts (DataFrame): reference dataframe
            similarity_computers (list of tuple): list of similarity computers and their weighing standardizers
        """
        super().__init__(**kwargs)
        self._df_contracts = df_contracts
        self._similarity_computers = similarity_computers if similarity_computers is not None \
            else [
            (AggregatedItemSimilarityComputer(self._df_contracts), WeightedStandardizer(1)),
            (AggregatedLocalSimilarityComputer(self._df_contracts), WeightedStandardizer(0.1))
        ]

    def compute_most_similar(self, df_query, num_results=1):
        """Computes the pairwise similarities of queries to reference contracts.

        Uses the member similarity computers to compute the partial similarities between the records.
        Uses the member weighing standardizers to compute the complex aggregated similarities.

        Args:
            df_query (DataFrame): query dataframe
            num_results (int): number of most similar contracts to return, default is only the most similar one

        Returns:
            dict: list of n most similar reference contracts for each query

            The mapping is:
            query_id:
                list of reference contracts:
                    contract_id
                    similarity
        """
        similarities = {}
        # compute partial similarities using computers
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
        # aggregate the partial results
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
