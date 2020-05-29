from typing import List, Tuple, Dict

import hdbscan
import numpy
import pandas
from pandas import DataFrame
from scipy.spatial import distance as spatial_distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import Normalizer

from recommender.component.base import Component
from recommender.component.similarity.standardization import CosineStandardizer


class DistanceVectorComputer(Component):
    """Computes pairwise vector distance between two vector sets.

    Uses euclidean distance as the metric. (sklearn implementation)
    """

    def _compute_sorted_distances(self, target, vectors):
        distances = euclidean_distances(target, vectors)
        sorted_index = numpy.argsort(distances)
        return distances, sorted_index

    def compute_nearest(self, target, vectors, nresults=1) -> List[List[Tuple[int, float]]]:
        """Computes the pairwise distances and returns the nearest pairs.

        For each of target vectors computes the distances to all reference vectors
        and filters the n nearest.

        Args:
            target: target collection of vectors (2D matrix)
            vectors: reference collection of vectors (2D matrix)
            nresults (int): number of nearest vectors to return, default is only the nearest one

        Returns:
            list: list of n nearest vectors (tuple of index and distance) for each of target vectors
        """
        distances, sorted_index = self._compute_sorted_distances(target, vectors)
        if sorted_index.size == 0:
            return []
        # filter sorted indexes
        selected_index = sorted_index[:, :nresults]
        # merge selected indexes with distances
        return [[(index, distances[i, index]) for index in row] for i, row in enumerate(selected_index)]


class CosineDistanceVectorComputer(DistanceVectorComputer):
    """Computes pairwise vector distance between two vector sets.

    Uses cosine distance as the metric. (scipy implementation)
    """

    def _compute_sorted_distances(self, target, vectors):
        distances = spatial_distance.cdist(target, vectors, 'cosine')
        sorted_index = numpy.argsort(distances)
        return distances, sorted_index


class OptimalizedCosineDistanceVectorComputer(DistanceVectorComputer):
    """Computes pairwise vector distance between two vector sets.

    Uses cosine distance as the metric implemented as matrix multiplication. (numpy implementation)
    The computation is optimized using partial pre-computing of reference vectors.

    Attributes:
        vectors (ndarray): reference vectors
        vectors_norm (ndarray): pre-computed reference vectors norms
    """

    def __init__(self, vectors, **kwargs):
        """
        Args:
            vectors: reference collection of vectors
        """
        super().__init__(**kwargs)
        vectors = numpy.array(vectors)
        self.vectors = vectors
        # pre-compute reference vector norms
        vectors_extended = vectors.T.reshape((1, vectors.shape[1], vectors.shape[0]))
        self.vectors_norm = numpy.linalg.norm(vectors_extended, axis=1)

    def _compute_matrix(self, target):
        target = numpy.array(target)
        if len(target.shape) != 2:
            return numpy.empty(0)
        # compute target vector norms
        target_extended = target.reshape((target.shape[0], target.shape[1], 1))
        target_norm = numpy.linalg.norm(target_extended, axis=1)
        dot_product = target @ self.vectors.T
        norm = target_norm @ self.vectors_norm
        return 1 - (dot_product / norm)

    def _compute_sorted_distances(self, target, _):
        distances = self._compute_matrix(target)
        sorted_index = numpy.argsort(distances)
        return distances, sorted_index


class ItemDistanceComputer(Component):
    """Computes pairwise distances between two dataframe columns.

    Using specific distance computer computes the distances between two collections of items
    given by items and embeddings column of pandas DataFrame.
    Uses index mapping to flatten the representation of items collections.

    Attributes:
        DEFAULT_COLUMNS (tuple of str: str): default names of items and embeddings columns
    """

    DEFAULT_COLUMNS = ('embeddings', 'items')

    def __init__(self, df_contract_items: DataFrame, distance_computer=None, cols=DEFAULT_COLUMNS, **kwargs):
        """
        Args:
            df_contract_items (DataFrame): reference dataframe with items and embedding columns
            distance_computer (DistanceVectorComputer): vector distance computer
            cols (tuple of str: str): names of items and embeddings columns
        """
        super().__init__(**kwargs)
        self._df_contract_items = df_contract_items
        self._df_embeddings_col, self._df_items_col = cols
        # pre-compute reference items mapping
        self._nvectors, self._nvec_to_contr = self._count_mapping(df_contract_items)
        self._contract_ids = list(self._df_contract_items['contract_id'])
        self._contract_items = list(self._df_contract_items[self._df_items_col])
        self._distance_vec_comp = distance_computer or OptimalizedCosineDistanceVectorComputer(self._nvectors)

    def _count_mapping(self, df_items):
        embeddings_col = self._df_embeddings_col if self._df_embeddings_col in df_items.columns else \
            self.DEFAULT_COLUMNS[0]
        vectors = []
        vec_to_entity = []
        for index, row in df_items.iterrows():
            if not isinstance(row[embeddings_col], list):
                continue
            for i, e in enumerate(row[embeddings_col]):
                if numpy.all((e == 0)) or numpy.any(numpy.isnan(e)):
                    continue
                vectors.append(e)
                vec_to_entity.append((index, i))
        nvectors = numpy.array(vectors, dtype=numpy.float32)
        nvec_to_entity = numpy.array(vec_to_entity, dtype=numpy.int)
        return nvectors, nvec_to_entity

    def compute_nearest(self, df_query_items: DataFrame, num_results=1) -> Dict[int, Dict[int, List[Dict[str, any]]]]:
        """Computes the pairwise distances of query(target) items to reference items.

         Uses the member distance computer to compute the distances
         between the flat collections of target items to flat collection of member reference items.
         Transforms the source hierarchical representation of items to a flat representation using index mapping.

        Args:
            df_query_items (DataFrame): target dataframe with items and embeddings columns
            num_results (int): number of nearest items to return, default is only the nearest one

        Returns:
            dict: list of n nearest items (with mapping to its contract and distance) for each item of each query

            The mapping is:
            query_id:
                query_item:
                    list of reference items:
                        contract_id
                        item
                        distance
        """
        target_nvectors, nvec_to_query = self._count_mapping(df_query_items)
        most_similar_vecs = self._distance_vec_comp.compute_nearest(target_nvectors, self._nvectors, num_results)

        results = {}
        # backward mapping of indexes to items
        for index_target_item, target_item_row in enumerate(most_similar_vecs):
            index_df_query = nvec_to_query[index_target_item][0]
            query_id = df_query_items.loc[index_df_query, 'query_id']
            query_items = df_query_items.loc[index_df_query, 'items']
            index_query_item = nvec_to_query[index_target_item][1]
            query_item = query_items[index_query_item]

            query_results = results.get(query_id, {})
            results[query_id] = query_results
            item_results = query_results.get(query_item, [])
            query_results[query_item] = item_results

            for index_contr_item, distance in target_item_row:
                index_df_contr = self._nvec_to_contr[index_contr_item][0]
                contr_id = self._contract_ids[index_df_contr]
                contr_items = self._contract_items[index_df_contr]
                index_contr_item = self._nvec_to_contr[index_contr_item][1]
                contr_item = contr_items[index_contr_item]

                item_results.append({'contract_id': contr_id, 'item': contr_item, 'distance': distance})
        return results


class ClusteredItemDistanceComputer(ItemDistanceComputer):

    DEFAULT_COLUMNS = ('embeddings', 'items')

    def __init__(self, df_contract_items: DataFrame, distance_computer=None, clusterer_args=None, **kwargs):
        super().__init__(df_contract_items, distance_computer="fake_computer", **kwargs)
        self._clusterer_args = clusterer_args or {'min_cluster_size': 5, 'cluster_selection_epsilon': 0.5}
        self._clusters = self._build_clusters(self._nvectors, self._nvec_to_contr)
        self._nrepr_vectors, self._nvec_to_cluster = self._count_mappings_to_clusters(self._clusters)
        self._vec_to_vec_to_contr = [cluster['vec_to_entity'] for cluster in self._clusters.values()]
        self._distance_vec_comp = distance_computer or OptimalizedCosineDistanceVectorComputer(self._nrepr_vectors)

    def _count_mapping(self, df_items):
        embeddings_col = self._df_embeddings_col if self._df_embeddings_col in df_items.columns else \
            self.DEFAULT_COLUMNS[0]
        vectors = []
        vec_to_entity = []
        for index, row in df_items.iterrows():
            if not isinstance(row[embeddings_col], list):
                continue
            for i, e in enumerate(row[embeddings_col]):
                vectors.append(e)
                vec_to_entity.append((index, i))
        nvectors = numpy.array(vectors, dtype=numpy.float32)
        nvec_to_entity = numpy.array(vec_to_entity, dtype=numpy.int)
        return nvectors, nvec_to_entity

    def _build_clusters(self, nvectors, nvec_to_entity):
        normalizer = Normalizer(norm='l2')
        clusterer = hdbscan.HDBSCAN(**self._clusterer_args)
        nnvectors = normalizer.fit(nvectors).transform(nvectors)
        clusterer.fit_predict(nnvectors)
        clusters = {}
        off_clusters = {}
        num_clusters = clusterer.labels_.max()
        for i, label in enumerate(clusterer.labels_):
            if label == -1:
                vec = nvectors[i]
                tvec = tuple(vec)
                if tvec not in off_clusters:
                    num_clusters += 1
                    off_clusters[tvec] = num_clusters
                label = off_clusters[tvec]
            cluster = clusters.get(label,
                                   {'cluster_id': label, 'representatives': set(), 'vectors': [], 'vec_to_entity': []})
            cluster['representatives'].add(tuple(nvectors[i]))
            cluster['vectors'].append(nvectors[i])
            cluster['vec_to_entity'].append(nvec_to_entity[i])
            clusters[label] = cluster
        for cluster in clusters.values():
            cluster['representatives'] = [numpy.array(vec) for vec in cluster['representatives']]
        return clusters

    def _count_mappings_to_clusters(self, clusters):
        repr_vectors = []
        vec_to_cluster = []
        for cluster in clusters.values():
            for e in cluster['representatives']:
                repr_vectors.append(e)
                vec_to_cluster.append(cluster['cluster_id'])
        nrepr_vectors = numpy.array(repr_vectors, dtype=numpy.float32)
        nvec_to_cluster = numpy.array(vec_to_cluster, dtype=numpy.int)
        return nrepr_vectors, nvec_to_cluster

    def compute_nearest(self, df_query_items: DataFrame, num_results=1) -> Dict[int, Dict[int, List[Dict[str, any]]]]:
        target_nvectors, nvec_to_query = self._count_mapping(df_query_items)
        self._timer.start()
        nearest_clusters = self._distance_vec_comp.compute_nearest(target_nvectors, self._nrepr_vectors, num_results)
        self._timer.stop()

        self._timer.start()
        results = {}
        # backward mapping of indexes to items
        for index_target_item, target_item_row in enumerate(nearest_clusters):
            index_df_query = nvec_to_query[index_target_item][0]
            query_id = df_query_items.loc[index_df_query, 'query_id']
            query_items = df_query_items.loc[index_df_query, 'items']
            index_query_item = nvec_to_query[index_target_item][1]
            query_item = query_items[index_query_item]

            query_results = results.get(query_id, {})
            results[query_id] = query_results
            item_results = query_results.get(query_item, [])
            query_results[query_item] = item_results

            for index_cluster, distance in target_item_row:
                cluster_id = self._nvec_to_cluster[index_cluster]
                for vec_to_entity in self._vec_to_vec_to_contr[cluster_id]:
                    index_df_contr = vec_to_entity[0]
                    contr_id = self._contract_ids[index_df_contr]
                    contr_items = self._contract_items[index_df_contr]
                    index_contr_item = vec_to_entity[1]
                    contr_item = contr_items[index_contr_item]
                    item_results.append({'contract_id': contr_id, 'item': contr_item, 'distance': distance})
        self._timer.stop()
        return results


class SimilarItemsComputer(Component):
    """Computes pairwise similarities between items from two dataframes.

    Using specific distance computer computes the distances between two collections of items
    and transforms the distances to similarities using specific standardizer.
    """

    def __init__(self, df_contract_items: DataFrame, distance_computer=None, standardizer=None, **kwargs):
        """
        Args:
            df_contract_items (DataFrame): reference dataframe with items and embedding columns
            distance_computer (ItemDistanceComputer): item distance computer
            standardizer (Standardizer): standardizer of distance value to similarity value
        """
        super().__init__(**kwargs)
        self._distance_computer = distance_computer if distance_computer is not None else ItemDistanceComputer(
            df_contract_items, logger=self.logger)
        self._standardizer = standardizer if standardizer is not None else CosineStandardizer()

    def compute_most_similar(self, df_query_items, num_results=1) -> Dict[int, Dict[int, List[Dict[str, any]]]]:
        """Computes the pairwise similarities of query items to reference items.

        Uses the member distance computer to compute the distances between the items.
        Uses the member standardizer to transform computed distances to similarites.

        Args:
            df_query_items (DataFrame): target dataframe with items and embeddings columns
            num_results (int): number of most similar items to return, default is only the nearest one

        Returns:
            dict: list of n most similar reference items for each item of each query

            The mapping is:
            query_id:
                query_item:
                    list of reference items:
                        contract_id
                        item
                        distance
                        similarity
        """
        nearest = self._distance_computer.compute_nearest(df_query_items, num_results)
        for query in nearest.values():
            for item in query.values():
                for contract in item:
                    contract['similarity'] = self._standardizer.compute(contract['distance'])
        return nearest


class AggregatedItemSimilarityComputer(SimilarItemsComputer):
    """Computes similarities of queries and contracts.

    Extends pairwise item similarity computation with aggregation by queries and contracts.
    """

    @staticmethod
    def wavg(group, avg_name, weight_name) -> float:
        """Computes weighted average of dataframe's column.

        As the weights is used a column of the same dataframe.

        Args:
            group (DataFrame): dataframe with columns avg_name and weight_name
            avg_name (str): name of the column with values
            weight_name (str): name of the column with weights

        Returns:
            float: weighted average of column
        """
        d = group[avg_name]
        w = group[weight_name]
        return (d * w).sum() / w.sum()

    @staticmethod
    def _weighted_average(df, data_col, weight_col, by_col):
        # algorithm taken from https://stackoverflow.com/a/44683506/13484859
        df['_data_times_weight'] = df[data_col] * df[weight_col]
        df['_weight_where_notnull'] = df[weight_col] * pandas.notnull(df[data_col])
        g = df.groupby(by_col)
        result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
        del df['_data_times_weight'], df['_weight_where_notnull']
        return result

    def compute_most_similar(self, df_query_items, num_results=1) -> Dict[int, List[Dict[str, float]]]:
        """Computes the similarities of queries to reference contracts.

        Extends pairwise item similarity computation with aggregation by queries and contracts.
        Uses class function for computation of weighted average of item similarities.

        Args:
            df_query_items (DataFrame): target dataframe with items and embeddings columns
            num_results (int): number of most similar items to return, default is only the nearest one

        Returns:
            dict: list of n most similar reference contracts for each query

            The mapping is:
            query_id:
                list of reference contracts:
                    contract_id
                    similarity
        """
        similar_items = super().compute_most_similar(df_query_items, num_results=numpy.iinfo(numpy.int32).max)
        similar_items_flat = []
        # flatten similar items
        for query in similar_items:
            for iitem in similar_items[query]:
                for item in similar_items[query][iitem]:
                    item['iitem'] = iitem
                    item['query_id'] = query
                    similar_items_flat.append(item)
        if not similar_items_flat:
            return {}
        df_similar_items = pandas.DataFrame(similar_items_flat)
        # weighted similarities of items by queries and contracts
        self._timer.start()
        s_aggregated = self._weighted_average(df_similar_items, 'similarity', 'similarity', ['query_id', 'contract_id'])
        self._timer.stop('groupby2')
        s_aggregated = s_aggregated.rename('similarity')
        # filter n most similar
        df_nlargest = s_aggregated.groupby('query_id').nlargest(num_results).reset_index(drop=True,
                                                                                         level=1).reset_index()
        # transform to dict of list
        result = {query: g[['contract_id', 'similarity']].to_dict('record')
                  for query, g in df_nlargest.groupby('query_id')}
        return result
