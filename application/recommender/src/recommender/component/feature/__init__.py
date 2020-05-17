"""Package provides components for feature extraction.

There are several features these components extract:

- Document - document module provides common text utilities
- Embedding - embedding module provides components for extraction of embedding of text
- Geospatial - geospatial module provides component for geocoding
- Contract subject - contract subject package contains components for subject extraction from contract documents text

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
