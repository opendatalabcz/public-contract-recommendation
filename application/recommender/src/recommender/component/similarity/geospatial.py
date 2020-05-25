from typing import Dict, List

import numpy
import pandas

from geopy.distance import distance as geo_distance

from recommender.component.base import Component
from recommender.component.similarity.standardization import Log1pStandardizer
from recommender.component.similarity.vector_space import DistanceVectorComputer


class GeodesicDistanceComputer(DistanceVectorComputer):
    """Computes pairwise geodesic distance between two GPS coordination sets.

    Uses geodesic distance as the metric. (geopy implementation)
    """

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


class ApproximatedGeoDistanceComputer(DistanceVectorComputer):
    """Computes pairwise approximation of geographic distance between two GPS coordination sets.

    Uses euclidean distance as the metric. (implementation from DistanceVectorComputer).
    The euclidean distance computation is feasible due to the approximation of GPS coordinations in vector space.

    Attributes:
        DEF_LATITUDE_TO_KILOMETERS (float): default translation ratio of one degree of latitude to kilometers
        DEF_LONGITUDE_TO_KILOMETERS (float): default translation ratio of one degree of longitude to kilometers
        lattokm (float): translation ratio of one degree of latitude to kilometers
        lontokm (float): translation ratio of one degree of longitude to kilometers
        euclidean_vectors (ndarray): approximated reference GPS coordinations in vector space
    """

    DEF_LATITUDE_TO_KILOMETERS = 71.941
    DEF_LONGITUDE_TO_KILOMETERS = 111.319

    def __init__(self, vectors, lattokm=DEF_LATITUDE_TO_KILOMETERS, lontokm=DEF_LONGITUDE_TO_KILOMETERS, **kwargs):
        """
        Args:
            vectors: reference collection of GPS coordinations
            lattokm: translation ratio of one degree of latitude to kilometers
            lontokm: translation ratio of one degree of longitude to kilometers
        """
        super().__init__(**kwargs)
        vectors = numpy.array(vectors)
        # reference GPS coordination approximation
        self.lattokm = lattokm
        self.lontokm = lontokm
        self.euclidean_vectors = numpy.empty(vectors.shape)
        self.euclidean_vectors[:, 0] = vectors[:, 0] * self.lontokm
        self.euclidean_vectors[:, 1] = vectors[:, 1] * self.lattokm

    def _compute_sorted_distances(self, target, vectors):
        # target GPS coordination approximation
        euclidean_target = numpy.empty(target.shape)
        euclidean_target[:, 0] = target[:, 0] * self.lontokm
        euclidean_target[:, 1] = target[:, 1] * self.lattokm
        return super()._compute_sorted_distances(euclidean_target, self.euclidean_vectors)


class LocalityDistanceComputer(Component):
    """Computes pairwise distances between two dataframe columns.

    Using (geographic) distance computer computes the distances between two collections of locations
    given by address and gps column of pandas DataFrame.
    Uses index mapping to flatten the representation of location collections.
    """

    def __init__(self, df_contract_locations, distance_computer=None, **kwargs):
        """
        Args:
            df_contract_locations (DataFrame): reference dataframe with address and gps columns
            distance_computer (DistanceVectorComputer): (geographic) distance computer
        """
        super().__init__(**kwargs)
        self._df_contract_locations = df_contract_locations
        # pre-compute reference locations mapping
        self._nvectors, self._nvec_to_contr = self._count_mapping(df_contract_locations)
        self._contract_ids = list(self._df_contract_locations['contract_id'])
        self._addresses = list(self._df_contract_locations['address'])
        self._distance_computer = distance_computer or ApproximatedGeoDistanceComputer(self._nvectors)

    def _count_mapping(self, df_locations):
        vectors = []
        vec_to_entity = []
        for index, row in df_locations.iterrows():
            if not isinstance(row['gps'], tuple):
                continue
            gps = numpy.array(row['gps'], dtype=numpy.float64)
            if gps.shape != (2,) or numpy.all((gps == 0)) or numpy.any(numpy.isnan(gps)):
                continue
            vectors.append(gps)
            vec_to_entity.append(index)

        nvectors = numpy.array(vectors, dtype=numpy.float32)
        nvec_to_entity = numpy.array(vec_to_entity, dtype=numpy.int)
        return nvectors, nvec_to_entity

    def compute_nearest(self, df_query_locations, num_results=1) -> Dict[int, Dict[int, List[Dict[str, any]]]]:
        """Computes the pairwise distances of query(target) locations to reference locations.

         Uses the member distance computer to compute the distances
         between the flat collections of target locations to flat collection of member reference locations.
         Transforms the source hierarchical representation of locations to a flat representation using index mapping.

        Args:
            df_query_locations (DataFrame): target dataframe with address and gps columns
            num_results (int): number of nearest locations to return, default is only the nearest one

        Returns:
            dict: list of n nearest items (with mapping to its contract and distance) for each item of each query

            The mapping is:
            query_id:
                query_location:
                    list of reference locations:
                        contract_id
                        address
                        distance
        """
        target_nvectors, nvec_to_query_locations = self._count_mapping(df_query_locations)
        most_similar_vecs = self._distance_computer.compute_nearest(target_nvectors, self._nvectors, num_results)

        results = {}

        # backward mapping of indexes to locations
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
                contr_id = self._contract_ids[index_df_contr]
                address = self._addresses[index_df_contr]

                loc_results.append({'contract_id': contr_id, 'address': address, 'distance': distance})

        return results


class SimilarLocalityComputer(Component):
    """Computes pairwise similarities between items from two dataframes.

    Using specific (geographic) distance computer computes the distances between two collections of locations
    and transforms the distances to similarities using specific standardizer.
    """
    def __init__(self, df_contract_locality, distance_computer=None, standardizer=None, **kwargs):
        """
        Args:
            df_contract_locality (DataFrame): reference dataframe with address and gps columns
            distance_computer (LocalityDistanceComputer): locality distance computer
            standardizer (Standardizer): standardizer of distance value to similarity value
        """
        super().__init__(**kwargs)
        self._distance_computer = distance_computer if distance_computer is not None else LocalityDistanceComputer(
            df_contract_locality, logger=self.logger)
        self._standardizer = standardizer if standardizer is not None else Log1pStandardizer()

    def compute_most_similar(self, df_query_locations, num_results=1) -> Dict[int, Dict[int, List[Dict[str, any]]]]:
        """Computes the pairwise similarities of query locations to reference locations.

        Uses the member distance computer to compute the distances between the locations.
        Uses the member standardizer to transform computed distances to similarites.

        Args:
            df_query_locations (DataFrame): target dataframe with address and gps columns
            num_results (int): number of most similar locations to return, default is only the most similar one

        Returns:
            dict: list of n most similar reference locations for each location of each query

            The mapping is:
            query_id:
                query_location:
                    list of reference locations:
                        contract_id
                        address
                        distance
                        similarity
        """
        nearest = self._distance_computer.compute_nearest(df_query_locations, num_results)
        for query in nearest.values():
            for loc in query.values():
                for contract in loc:
                    contract['similarity'] = self._standardizer.compute(contract['distance'])
        return nearest


class AggregatedLocalSimilarityComputer(SimilarLocalityComputer):
    """Computes similarities of queries and contracts.

    Extends pairwise locality similarity computation with aggregation by queries and contracts.
    """

    def compute_most_similar(self, df_query_locations, num_results=1) -> Dict[int, List[Dict[str, float]]]:
        """Computes the similarities of queries to reference contracts.

        Extends pairwise locality similarity computation with aggregation by queries and contracts.

        Args:
            df_query_locations (DataFrame): target dataframe with address and gps columns
            num_results (int): number of most similar locations to return, default is only the most similar one

        Returns:
            dict: list of n most similar reference contracts for each query

            The mapping is:
            query_id:
                list of reference contracts:
                    contract_id
                    similarity
        """
        similar_addresses = super().compute_most_similar(df_query_locations, num_results=numpy.iinfo(numpy.int32).max)
        similar_addresses_flat = []
        # flatten similar items
        for qid in similar_addresses:
            for address in list(similar_addresses[qid].values())[0]:
                address['query_id'] = qid
                similar_addresses_flat.append(address)
        if not similar_addresses_flat:
            return {}
        df_similar_addresses = pandas.DataFrame(similar_addresses_flat)
        s_aggregated = df_similar_addresses.set_index(['query_id', 'contract_id'])['similarity']
        # filter n most similar
        df_nlargest = s_aggregated.groupby('query_id').apply(
            lambda x: x.reset_index(drop=True, level=0).nlargest(num_results)).reset_index()

        # transform to dict of list
        result = {query: g[['contract_id', 'similarity']].to_dict('record')
                  for query, g in df_nlargest.groupby('query_id')}
        return result
