import numpy
import pandas

from geopy.distance import distance as geo_distance

from recommender.component.base import Component
from recommender.component.similarity.standardization import Log10Standardizer


class GeodesicDistanceComputer(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _compute_matrix(target, vectors):
        result = []
        for t in target:
            res = []
            for v in vectors:
                d = geo_distance(t, v).kilometers
                res.append(d)
            result.append(res)
        return numpy.array(result, dtype=numpy.float32)

    def compute_distances(self, target, vectors, num_results=1):
        distances = self._compute_matrix(target, vectors)
        sorted_index = numpy.argsort(distances)
        selected_index = sorted_index[:, :num_results]
        return [[(index, distances[i, index]) for index in row] for i, row in enumerate(selected_index)]


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
            vectors.append(list(row['gps']))
            vec_to_entity.append(index)

        nvectors = numpy.array(vectors, dtype=numpy.float32)
        nvec_to_entity = numpy.array(vec_to_entity, dtype=numpy.float32)
        return nvectors, nvec_to_entity

    def compute_nearest(self, df_user_profile, num_results=1):
        target_nvectors, nvec_to_user_profile = self._count_mapping(df_user_profile)
        most_similar_vecs = self._distance_computer.compute_distances(target_nvectors, self._nvectors, num_results)

        results = {}

        for index_target_loc, target_loc_row in enumerate(most_similar_vecs):
            index_df_user_profile = nvec_to_user_profile[index_target_loc]
            user_id = df_user_profile.loc[index_df_user_profile, 'user_id']
            user_address = df_user_profile.loc[index_df_user_profile, 'address']

            user_results = results.get(user_id, {})
            results[user_id] = user_results
            loc_results = user_results.get(user_address, [])
            user_results[user_address] = loc_results

            for index_contr_loc, distance in target_loc_row:
                index_df_contr = self._nvec_to_contr[index_contr_loc]
                contr_id = self._df_contract_locations.loc[index_df_contr, 'contract_id']
                address = self._df_contract_locations.loc[index_df_contr, 'address']

                loc_results.append({'contract_id': contr_id, 'address': address, 'distance': distance})

        return results


class SimilarLocalityComputer(Component):

    def __init__(self, df_contract_locality, distance_domputer=None, standardizer=None, **kwargs):
        super().__init__(**kwargs)
        self._distance_computer = distance_domputer if distance_domputer is not None else LocalityDistanceComputer(df_contract_locality)
        self._standardizer = standardizer if standardizer is not None else Log10Standardizer()

    def compute_most_similar(self, df_user_profile, num_results=1):
        nearest = self._distance_computer.compute_nearest(df_user_profile, num_results)
        for user in nearest.values():
            for loc in user.values():
                for contract in loc:
                    contract['similarity'] = self._standardizer.compute(contract['distance'])
        return nearest


class AggregatedLocalSimilarityComputer(SimilarLocalityComputer):

    def compute_most_similar(self, df_user_profile, num_results=1):
        similar_addresses = super().compute_most_similar(df_user_profile, num_results=numpy.iinfo(numpy.int32).max)
        similar_addresses_flat = []
        for uid in similar_addresses:
            for address in list(similar_addresses[uid].values())[0]:
                address['user'] = uid
                similar_addresses_flat.append(address)

        df_similar_addresses = pandas.DataFrame(similar_addresses_flat)
        s_aggregated = df_similar_addresses.set_index(['user', 'contract_id'])['similarity']
        df_nlargest = s_aggregated.groupby('user').apply(
            lambda x: x.reset_index(drop=True, level=0).nlargest(num_results)).reset_index()
        result = {user: g[['contract_id', 'similarity']].to_dict('record')
                  for user, g in df_nlargest.groupby('user')}
        return result