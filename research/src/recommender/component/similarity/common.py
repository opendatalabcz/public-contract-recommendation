import numpy
import pandas

from recommender.component.similarity.vector_space import AggregatedItemSimilarityComputer
from recommender.component.similarity.geodesic import AggregatedLocalSimilarityComputer
from recommender.component.base import Component


class ComplexSimilarityComputer(Component):

    def __init__(self, df_contracts, similarity_computers=None, **kwargs):
        super().__init__(**kwargs)
        self._df_contracts = df_contracts
        self._similarity_computers = similarity_computers if similarity_computers is not None \
            else [
                AggregatedItemSimilarityComputer(self._df_contracts),
                AggregatedLocalSimilarityComputer(self._df_contracts)
            ]

    def compute_most_similar(self, df_user_profile, num_results=1):
        contract_ids = self._df_contracts['contract_id']
        similarities = {uid: {cid: 0.5 for cid in contract_ids} for uid in df_user_profile['user_id']}
        for sc in self._similarity_computers:
            most_similar = sc.compute_most_similar(df_user_profile, num_results=numpy.iinfo(numpy.int32).max)
            for uid in most_similar:
                for contract_res in most_similar[uid]:
                    cid = contract_res['contract_id']
                    similarities[uid][cid] *= contract_res['similarity']
        df_similar_contracts = pandas.DataFrame.from_dict(similarities, orient='index')
        s_similar_contracts = df_similar_contracts.stack()
        s_similar_contracts = s_similar_contracts.rename('similarity')
        s_similar_contracts = s_similar_contracts.rename_axis(['user_id', 'contract_id'])
        df_nlargest = s_similar_contracts.groupby(level=0).nlargest(num_results).reset_index(drop=True, level=1).reset_index()
        result = {user: g[['contract_id', 'similarity']].to_dict('record')
                  for user, g in df_nlargest.groupby('user_id')}
        return result