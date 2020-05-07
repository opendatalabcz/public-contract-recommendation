import numpy
import pandas

from geopy.distance import distance as geo_distance

from recommender.component.base import Component
from recommender.component.similarity.standardization import Log10Standardizer
from recommender.component.similarity.vector_space import DistanceVectorComputer


class GeodesicDistanceComputer(DistanceVectorComputer):

    @staticmethod
    def _compute_matrix(target, vectors):
        result = []
        for t in target:
            res = []
            for v in vectors:
                d = geo_distance(t, v).kilometers
                res.append(d)
            result.append(res)
        if not result:
            return numpy.empty((0, 2), dtype=numpy.float32)
        return numpy.array(result, dtype=numpy.float32)

    def _compute_sorted_distances(self, target, vectors):
        distances = self._compute_matrix(target, vectors)
        sorted_index = numpy.argsort(distances)
        return distances, sorted_index


class LocalityDistanceComputer(Component):

    def __init__(self, df_contract_locations, distance_computer=None, **kwargs):
        super().__init__(**kwargs)
        self._df_contract_locations = df_contract_locations
        self._nvectors, self._nvec_to_contr = self._count_mapping(df_contract_locations)
        self._distance_computer = distance_computer if distance_computer is not None else GeodesicDistanceComputer()

    def _count_mapping(self, df_locations):
        vectors = []
        vec_to_entity = []
        for index, row in df_locations.iterrows():
            if not isinstance(row['gps'], tuple):
                continue
            vectors.append(list(row['gps']))
            vec_to_entity.append(index)

        nvectors = numpy.array(vectors, dtype=numpy.float32)
        nvec_to_entity = numpy.array(vec_to_entity, dtype=numpy.float32)
        return nvectors, nvec_to_entity

    def compute_nearest(self, df_query_locations, num_results=1):
        target_nvectors, nvec_to_query_locations = self._count_mapping(df_query_locations)
        most_similar_vecs = self._distance_computer.compute_nearest(target_nvectors, self._nvectors, num_results)

        results = {}

        for index_target_loc, target_loc_row in enumerate(most_similar_vecs):
            index_df_query_locations = nvec_to_query_locations[index_target_loc]
            query_id = df_query_locations.loc[index_df_query_locations, 'query_id']
            query_address = df_query_locations.loc[index_df_query_locations, 'address']

            query_results = results.get(query_id, {})
            results[query_id] = query_results
            loc_results = query_results.get(query_address, [])
            query_results[query_address] = loc_results

            for index_contr_loc, distance in target_loc_row:
                index_df_contr = self._nvec_to_contr[index_contr_loc]
                contr_id = self._df_contract_locations.loc[index_df_contr, 'contract_id']
                address = self._df_contract_locations.loc[index_df_contr, 'address']

                loc_results.append({'contract_id': contr_id, 'address': address, 'distance': distance})

        return results


class SimilarLocalityComputer(Component):

    def __init__(self, df_contract_locality, distance_computer=None, standardizer=None, **kwargs):
        super().__init__(**kwargs)
        self._distance_computer = distance_computer if distance_computer is not None else LocalityDistanceComputer(
            df_contract_locality, logger=self.logger)
        self._standardizer = standardizer if standardizer is not None else Log10Standardizer()

    def compute_most_similar(self, df_query_locations, num_results=1):
        nearest = self._distance_computer.compute_nearest(df_query_locations, num_results)
        for query in nearest.values():
            for loc in query.values():
                for contract in loc:
                    contract['similarity'] = self._standardizer.compute(contract['distance'])
        return nearest


class AggregatedLocalSimilarityComputer(SimilarLocalityComputer):

    def compute_most_similar(self, df_query_locations, num_results=1):
        similar_addresses = super().compute_most_similar(df_query_locations, num_results=numpy.iinfo(numpy.int32).max)
        similar_addresses_flat = []
        for qid in similar_addresses:
            for address in list(similar_addresses[qid].values())[0]:
                address['query_id'] = qid
                similar_addresses_flat.append(address)
        if not similar_addresses_flat:
            return {}
        df_similar_addresses = pandas.DataFrame(similar_addresses_flat)
        s_aggregated = df_similar_addresses.set_index(['query_id', 'contract_id'])['similarity']
        df_nlargest = s_aggregated.groupby('query_id').apply(
            lambda x: x.reset_index(drop=True, level=0).nlargest(num_results)).reset_index()
        result = {query: g[['contract_id', 'similarity']].to_dict('record')
                  for query, g in df_nlargest.groupby('query_id')}
        return result
