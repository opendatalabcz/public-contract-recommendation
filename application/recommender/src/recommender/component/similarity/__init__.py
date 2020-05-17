"""Package provides components for computation of similarity.

Module engine contains component SearchEngine that provides search/query interface.

Example usage::

    from recommender.component.similarity.common import ComplexSimilarityComputer
    ...
    complex_similarity_computer = ComplexSimilarityComputer(self.df_contracts,
                                                            similarity_computers=similarity_computers)
    ...
    result = complex_similarity_computer.compute_most_similar(df_query, self.num_results)

"""
