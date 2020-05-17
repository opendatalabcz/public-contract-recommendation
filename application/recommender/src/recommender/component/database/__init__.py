"""Package provides components for similarity computation.

There are several ways these components computes similarity:

- Common - common module provides components for common usage or aggregations
- Geospatial - geospatial module provides component for calculation of geographic similarity and distance
- Vector space - vector_space module provides components for colculation of vector similarity and distance
- Standardization - standardization module provides standardizers for variate standardization of values

Example usage::

    from recommender.component.feature.contract_subject import SubjectExtractor
    from recommender.component.feature.embedding import FastTextEmbedder
    ...
    extractor = SubjectExtractor(text_annotator=annotator, concatenator=items_spliter, logger=root_logger)
    embedder = FastTextEmbedder(FASTTEXT_PATH, logger=root_logger)
    ...
    df_contracts['subject_items'] = df_contracts['doc_text'].apply(lambda text: extractor.process(text))
    df_contracts['embeddings'] = df_contracts['subject_items'].apply(lambda items: embedder.process(items))

"""
