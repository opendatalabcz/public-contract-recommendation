"""Package provides component representing the engine (core) of the system.

Module engine contains component SearchEngine that provides search/query interface.

Example usage::

    from recommender.component.engine.engine import SearchEngine
    ...
    self.engine = SearchEngine(df_contracts, embedder=FastTextEmbedder(path_to_model, logger=self.logger),
                                num_results=10, logger=self.logger)
    ...
    result = self.engine.query(searchquery)
    ...
    result = self.engine.query_by_user_profile(df_user_profile, query_params)

"""
