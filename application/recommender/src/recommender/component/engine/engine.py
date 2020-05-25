from typing import Tuple, Dict, List

import pandas

from recommender.component.base import Component
from recommender.component.feature.geospatial import APIGeocoder
from recommender.component.feature.embedding import RandomEmbedder
from recommender.component.similarity.common import ComplexSimilarityComputer
from recommender.component.similarity.standardization import WeightedStandardizer
from recommender.component.similarity.vector_space import AggregatedItemSimilarityComputer, ItemDistanceComputer
from recommender.component.similarity.geospatial import AggregatedLocalSimilarityComputer


class SearchEngine(Component):
    """Represent the engine (core) of the searching respectively recommending system.

    Processes diverse queries against reference dataset with the purpose of finding the most similar records
    from the dataset to the query.

    Attributes:
        embedder (Component): embedding component for transformation of text input to vector representation
        geocoder (Component): geocoding component for transformation of input address to GPS representation
        num_results (int): number of results to be found for query
        df_contracts (DataFrame): reference dataframe representing the dataset
    """

    def __init__(self, df_contracts, embedder=None, geocoder=None, num_results=1, random_bias_rate=0.0, **kwargs):
        """
        Args:
            df_contracts (DataFrame): reference dataframe for the dataset
            embedder (Component): embedding component for transformation of text input to vector representation
            geocoder (Component): geocoding component for transformation of input address to GPS representation
            num_results (int): number of results to be found for query
            random_bias_rate (float): the rate of random bias, 0 means no bias, 1 means total random bias
        """
        super().__init__(**kwargs)
        self.embedder = embedder if embedder is not None else RandomEmbedder(logger=self.logger)
        self.geocoder = geocoder if geocoder is not None else APIGeocoder(logger=self.logger)
        self.num_results = num_results
        self.df_contracts = df_contracts
        self.random_bias_rate = random_bias_rate
        self._similarity_computers = {
            'subject': {
                'sc': AggregatedItemSimilarityComputer(self.df_contracts, logger=self.logger),
                'weight': 1,
                'cols': ('items', 'embeddings')},
            'locality': {
                'sc': AggregatedLocalSimilarityComputer(self.df_contracts, logger=self.logger),
                'weight': 0.1,
                'cols': ('address', 'gps')},
            'entity_subject': {
                'sc': AggregatedItemSimilarityComputer(self.df_contracts,
                                                       distance_computer=ItemDistanceComputer(df_contracts,
                                                                                              cols=('entity_embeddings',
                                                                                                    'entity_items'),
                                                                                              logger=self.logger),
                                                       logger=self.logger),
                'weight': 0.1,
                'cols': ('items', 'embeddings')},
        }
        self._full_similarity_computer = \
            ComplexSimilarityComputer(self.df_contracts,
                                      similarity_computers=[
                                          (sc['sc'], WeightedStandardizer(sc['weight']))
                                          for sc in self._similarity_computers.values()],
                                      random_bias_rate=self.random_bias_rate)

    def prepare_query_items(self, query_type, query_string) -> Tuple:
        """Prepares the raw query input to the inner representation.

        Regarding the query type uses relevant processing component:
            for textual subject items input uses the member embedder to transform to vector representation
            for address input uses the member geocoder to transform to gps coordinations

        Args:
            query_type (str): type of query input
            query_string (str): textual query input

        Returns:
            tuple: tuple of preprocessed query input string and its inner representation
        """
        if query_type in ['subject', 'entity_subject']:
            query_items = query_string.split('\n')
            query_embeddings = self.embedder.process(query_items)
            return query_items, query_embeddings
        if query_type == 'locality':
            gps = self.geocoder.gps_for_address(query_string)
            return query_string, gps

    def prepare_query(self, query_type, query_string) -> Dict[str, any]:
        """Prepares the query.

        Regarding the query type builds the query with relevant columns and query items.

        Args:
            query_type (str): type of query input
            query_string (str): textual query input

        Returns:
            dict: dictionary containing query_id, and relevant key-val pairs for specific query_type
        """
        cols = self._similarity_computers[query_type]['cols']
        query_items = self.prepare_query_items(query_type, query_string)
        return {'query_id': 1, cols[0]: query_items[0], cols[1]: query_items[1]}

    def run_query(self, query_type, query_string):
        """Runs specific query against the reference dataset.

        Prepares query and runs the similarity computation.

        Args:
            query_type (str): type of query input
            query_string (str): textual query input

        Returns:
            result: the result of similarity computer
        """
        query = self.prepare_query(query_type, query_string)
        df_query = pandas.DataFrame([query])
        sc = self._similarity_computers[query_type]['sc']
        result = sc.compute_most_similar(df_query, self.num_results)
        return result

    def query_by_subject(self, query_string):
        """Runs the query of subject query type.

        Args:
            query_string (str): textual query input

        Returns:
            result: the result of similarity computer
        """
        return self.run_query('subject', query_string)

    def query_by_address(self, query_string):
        """Runs the query of locality query type.

        Args:
            query_string (str): textual query input

        Returns:
            result: the result of similarity computer
        """
        return self.run_query('locality', query_string)

    def query_by_entity_subject(self, query_string):
        """Runs the query of entity subject query type.

        Args:
            query_string (str): textual query input

        Returns:
            result: the result of similarity computer
        """
        return self.run_query('entity_subject', query_string)

    def prepare_similarity_computers(self, query_params) -> List:
        """Prepares similarity computers for query.

        Regarding the query type collects the relevant similarity computers.

        Args:
            query_params (dict of str:str): query params containing query_type

        Returns:
            list: list of tuples of similarity computers and their standardizers
        """
        similarity_computers = []
        for query_type in query_params:
            sc = self._similarity_computers[query_type]['sc']
            standardizer = WeightedStandardizer(self._similarity_computers[query_type]['weight'])
            similarity_computers.append((sc, standardizer))
        return similarity_computers

    def merge_queries(self, queries):
        """Merges specific types of queries to single query

        Args:
            queries (list): list of partial queries

        Returns:
            dict: dictionary containing query_id, and merged relevant key-val pairs for specific query_type
        """
        query = {'query_id': 1}
        for subquery in queries:
            for field in subquery:
                val = query.get(field, None)
                if val and isinstance(val, list):
                    val.extend(subquery[field])
                else:
                    val = subquery[field]
                query[field] = val
        return query

    def query_by_user_profile(self, df_user_profile, query_params=None):
        """Runs query represented by user profile against the reference dataset.

        Prepares similarity computers regarding query parameters.
        Builds the query from user profile.
        Runs the query using prepared computers.

        Args:
            df_user_profile (DataFrame): query dataframe representing user profile with user_id, interest_items,
                                        embeddings, address and gps columns
            query_params (set of str): query params containing query_type

        Returns:
            result: the result of similarity computer
        """
        if query_params:
            similarity_computers = self.prepare_similarity_computers(query_params)
            sc = ComplexSimilarityComputer(self.df_contracts,
                                           similarity_computers=similarity_computers,
                                           random_bias_rate=self.random_bias_rate)
        else:
            sc = self._full_similarity_computer
        df_query = df_user_profile.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})
        result = sc.compute_most_similar(df_query, self.num_results)
        return result

    def query(self, query_params):
        """Runs generic query against the reference dataset.

        Prepares similarity computers regarding query parameters.
        Builds the query from query parameters.
        Runs the query using prepared computers.

        Args:
            query_params (dict of str: str): query params containing query_type and query_string

        Returns:
            result: the result of similarity computer
        """
        query = self.merge_queries([self.prepare_query(qt, qs) for qt, qs in query_params.items()])
        df_query = pandas.DataFrame([query])
        similarity_computers = self.prepare_similarity_computers(query_params)
        complex_similarity_computer = ComplexSimilarityComputer(self.df_contracts,
                                                                similarity_computers=similarity_computers,
                                                                random_bias_rate=self.random_bias_rate)
        result = complex_similarity_computer.compute_most_similar(df_query, self.num_results)
        return result
