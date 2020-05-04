import numpy
import pandas
from scipy.spatial import distance as spatial_distance

from recommender.component.base import Component
from recommender.component.similarity.standardization import CosineStandardizer


class DistanceVectorComputer(Component):

    def _compute_sorted_distances(self, target, vectors):
        similarities = numpy.dot(target, vectors.T)
        sorted_index = numpy.argsort(similarities).T[::-1].T
        return similarities, sorted_index

    def compute_nearest(self, target, vectors, nresults=1):
        distances, sorted_index = self._compute_sorted_distances(target, vectors)
        if sorted_index.size == 0:
            return []
        selected_index = sorted_index[:, :nresults]
        return [[(index, distances[i, index]) for index in row] for i, row in enumerate(selected_index)]


class CosineDistanceVectorComputer(DistanceVectorComputer):

    def _compute_sorted_distances(self, target, vectors):
        distances = spatial_distance.cdist(target, vectors, 'cosine')
        sorted_index = numpy.argsort(distances)
        return distances, sorted_index


class ItemDistanceComputer(Component):

    DEFAULT_COLUMNS = ('embeddings', 'items')

    def __init__(self, df_contract_items, distance_computer=None, cols=DEFAULT_COLUMNS, **kwargs):
        super().__init__(**kwargs)
        self._df_contract_items = df_contract_items
        self._df_embeddings_col, self._df_items_col = cols
        self._nvectors, self._nvec_to_contr = self._count_mapping(df_contract_items)
        self._distance_vec_comp = distance_computer if distance_computer is not None else CosineDistanceVectorComputer()

    def _count_mapping(self, df_items):
        embeddings_col = self._df_embeddings_col if self._df_embeddings_col in df_items.columns else self.DEFAULT_COLUMNS[0]
        vectors = []
        vec_to_entity = []
        for index, row in df_items.iterrows():
            if not isinstance(row[embeddings_col], list):
                continue
            for i, e in enumerate(row[embeddings_col]):
                vectors.append(e)
                vec_to_entity.append((index, i))
        nvectors = numpy.array(vectors)
        nvec_to_entity = numpy.array(vec_to_entity)
        return nvectors, nvec_to_entity

    def compute_nearest(self, df_query_items, num_results=1):
        target_nvectors, nvec_to_query = self._count_mapping(df_query_items)
        most_similar_vecs = self._distance_vec_comp.compute_nearest(target_nvectors, self._nvectors, num_results)

        results = {}

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
                contr_id = self._df_contract_items.loc[index_df_contr, 'contract_id']
                contr_items = self._df_contract_items.loc[index_df_contr, self._df_items_col]
                index_contr_item = self._nvec_to_contr[index_contr_item][1]
                contr_item = contr_items[index_contr_item]

                item_results.append({'contract_id': contr_id, 'item': contr_item, 'distance': distance})
        return results


class SimilarItemsComputer(Component):

    def __init__(self, df_contract_items, distance_computer=None, standardizer=None, **kwargs):
        super().__init__(**kwargs)
        self._distance_computer = distance_computer if distance_computer is not None else ItemDistanceComputer(df_contract_items)
        self._standardizer = standardizer if standardizer is not None else CosineStandardizer()

    def compute_most_similar(self, df_query_items, num_results=1):
        nearest = self._distance_computer.compute_nearest(df_query_items, num_results)
        for query in nearest.values():
            for item in query.values():
                for contract in item:
                    contract['similarity'] = self._standardizer.compute(contract['distance'])
        return nearest


class AggregatedItemSimilarityComputer(SimilarItemsComputer):

    @staticmethod
    def wavg(group, avg_name, weight_name):
        d = group[avg_name]
        w = group[weight_name]
        return (d * w).sum() / w.sum()

    def compute_most_similar(self, df_query_items, num_results=1):
        similar_items = super().compute_most_similar(df_query_items, num_results=numpy.iinfo(numpy.int32).max)
        similar_items_flat = []
        for query in similar_items:
            for iitem in similar_items[query]:
                for item in similar_items[query][iitem]:
                    item['iitem'] = iitem
                    item['query_id'] = query
                    similar_items_flat.append(item)
        if not similar_items_flat:
            return {}
        df_similar_items = pandas.DataFrame(similar_items_flat)
        s_aggregated = df_similar_items.groupby(['query_id', 'contract_id'])[['similarity']] \
            .apply(self.wavg, "similarity", "similarity")
        s_aggregated = s_aggregated.rename('similarity')
        df_nlargest = s_aggregated.groupby('query_id').nlargest(num_results).reset_index(drop=True, level=1).reset_index()
        result = {query: g[['contract_id', 'similarity']].to_dict('record')
                  for query, g in df_nlargest.groupby('query_id')}
        return result