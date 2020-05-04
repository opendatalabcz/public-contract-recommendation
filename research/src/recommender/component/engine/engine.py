import pandas

from recommender.component.base import Component
from recommender.component.feature.geospatial import APIGeocoder
from recommender.component.feature.embedding import RandomEmbedder
from recommender.component.similarity.common import ComplexSimilarityComputer
from recommender.component.similarity.standardization import WeightedStandardizer
from recommender.component.similarity.vector_space import AggregatedItemSimilarityComputer, ItemDistanceComputer
from recommender.component.similarity.geospatial import AggregatedLocalSimilarityComputer


class SearchEngine(Component):

    def __init__(self, df_contracts, embedder=None, geocoder=None, num_results=1, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder if embedder is not None else RandomEmbedder(logger=self.logger)
        self.geocoder = geocoder if geocoder is not None else APIGeocoder(logger=self.logger)
        self.num_results = num_results
        self.df_contracts = df_contracts
        self._similarity_computers = {
            'subject': {
                'sc': AggregatedItemSimilarityComputer(self.df_contracts, logger=self.logger),
                'weight': 1,
                'cols': ('items', 'embeddings')},
            'locality': {
                'sc': AggregatedLocalSimilarityComputer(self.df_contracts, logger=self.logger),
                'weight': 0.2,
                'cols': ('address', 'gps')},
            'entity_subject': {
                'sc': AggregatedItemSimilarityComputer(self.df_contracts,
                                                       distance_computer=ItemDistanceComputer(df_contracts,
                                                                                              cols=('entity_embeddings',
                                                                                                    'entity_items')),
                                                       logger=self.logger),
                'weight': 0.2,
                'cols': ('items', 'embeddings')},
        }
        self._full_similarity_computer = \
            ComplexSimilarityComputer(self.df_contracts,
                                      similarity_computers=[
                                          (sc['sc'], WeightedStandardizer(sc['weight']))
                                          for sc in self._similarity_computers.values()])

    def prepare_query_items(self, query_type, query_string):
        if query_type in ['subject', 'entity_subject']:
            query_items = query_string.split('\n')
            query_embeddings = self.embedder.process(query_items)
            return query_items, query_embeddings
        if query_type == 'locality':
            gps = self.geocoder.gps_for_address(query_string)
            return query_string, gps

    def prepare_query(self, query_type, query_string):
        cols = self._similarity_computers[query_type]['cols']
        query_items = self.prepare_query_items(query_type, query_string)
        return {'query_id': 1, cols[0]: query_items[0], cols[1]: query_items[1]}

    def run_query(self, query_type, query_string):
        query = self.prepare_query(query_type, query_string)
        df_query = pandas.DataFrame([query])
        sc = self._similarity_computers[query_type]['sc']
        result = sc.compute_most_similar(df_query, self.num_results)
        return result

    def query_by_subject(self, query_string):
        return self.run_query('subject', query_string)

    def query_by_address(self, query_string):
        return self.run_query('locality', query_string)

    def query_by_entity_subject(self, query_string):
        return self.run_query('entity_subject', query_string)

    def prepare_similarity_computers(self, query_params):
        similarity_computers = []
        for query_type in query_params:
            sc = self._similarity_computers[query_type]['sc']
            standardizer = WeightedStandardizer(self._similarity_computers[query_type]['weight'])
            similarity_computers.append((sc, standardizer))
        return similarity_computers

    def merge_queries(self, queries):
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
        if query_params:
            similarity_computers = self.prepare_similarity_computers(query_params)
            sc = ComplexSimilarityComputer(self.df_contracts,
                                           similarity_computers=similarity_computers)
        else:
            sc = self._full_similarity_computer
        df_query = df_user_profile.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})
        result = sc.compute_most_similar(df_query, self.num_results)
        return result

    def query(self, query_params):
        query = self.merge_queries([self.prepare_query(qt, qs) for qt, qs in query_params.items()])
        df_query = pandas.DataFrame([query])
        similarity_computers = self.prepare_similarity_computers(query_params)
        complex_similarity_computer = ComplexSimilarityComputer(self.df_contracts,
                                                                similarity_computers=similarity_computers)
        result = complex_similarity_computer.compute_most_similar(df_query, self.num_results)
        return result
