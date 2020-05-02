import random
import string

import numpy

from recommender.component.base import Component
from recommender.component.feature.contract_subject.conllu_preprocessing import TextAnnotator, \
    UdapiFromConlluTransformer, ConlluSubjectContextPreprocessor, \
    UdapiToStrTransformer
from recommender.component.feature.contract_subject.context_extraction import AdvancedSubjectContextExtractor
from recommender.component.feature.contract_subject.subject_context_preprocessing import SubjectContextPreprocessor, \
    AttributeExtractor, AttributeTagger, \
    AttributeTagCleaner
from recommender.component.feature.document import DataProcessor


def randomTokens():
    """Generate a random string of fixed length """
    tokensLength = int(random.gauss(4, 2)) + 1
    letters = string.ascii_lowercase
    tokens = []
    for _ in range(tokensLength):
        stringLength = int(random.gauss(6, 3) + 1)
        tokens.append(''.join(random.choice(letters) for i in range(stringLength)))
    return tokens


class AttributeMerger(DataProcessor):

    def merge_attributes(self, pair):
        attrs1, attrs2 = pair
        return attrs1 + '\n' + attrs2

    def _process_inner(self, pair):
        return self.merge_attributes(pair)


class AttributeConcatenator(Component):

    def concatenate(self, attributes):
        if isinstance(attributes, list):
            attributes = '\n'.join(attributes)
        return attributes

    def process(self, data):
        return self.concatenate(data)


class ItemsSplitter(Component):

    def split(self, items):
        return [item for subitems in items for item in subitems.split('\n') if item]

    def process(self, items):
        return self.split(items)


class SubjectExtractor(DataProcessor):

    def extract(self, text=None):
        return ['nic']

    def _process_inner(self, data):
        return self.extract(data)


class RandomSubjectExtractor(SubjectExtractor):

    def extract(self, text=None):
        subjects_num = int(numpy.random.exponential()) + 1
        return [randomTokens() for _ in range(subjects_num)]


class ComplexSubjectExtractor(SubjectExtractor):

    def __init__(self,
                 subj_context_extractor=None,
                 subj_context_preprocessor=None,
                 attributes_extractor=None,
                 text_annotator=None,
                 udapi_doc_creator=None,
                 conllu_attributes_preprocessor=None,
                 udapi_to_str_transformer=None,
                 items_tagger=None,
                 attribute_merger=None,
                 tag_cleaner=None,
                 concatenator=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._subj_context_extractor = subj_context_extractor if subj_context_extractor is not None else \
            AdvancedSubjectContextExtractor()
        self._subj_context_preprocessor = subj_context_preprocessor if subj_context_preprocessor is not None else \
            SubjectContextPreprocessor()
        self._attribures_extractor = attributes_extractor if attributes_extractor is not None else \
            SubjectContextPreprocessor(transformers=[AttributeExtractor(keep_text=False, keep_attributes=True)])
        self._text_annotator = text_annotator if text_annotator is not None else \
            TextAnnotator()
        self._udapi_doc_creator = udapi_doc_creator if udapi_doc_creator is not None else \
            UdapiFromConlluTransformer()
        self._conllu_attributes_preprocessor = conllu_attributes_preprocessor if conllu_attributes_preprocessor is not None else \
            ConlluSubjectContextPreprocessor()
        self._udapi_to_str_transformer = udapi_to_str_transformer if udapi_to_str_transformer is not None else \
            UdapiToStrTransformer()
        self._items_tagger = items_tagger if items_tagger is not None else \
            SubjectContextPreprocessor(transformers=[AttributeTagger(attr_tag='<ITEM>;<ITEM/>', keep_text=False)])
        self._attribute_merger = attribute_merger if attribute_merger is not None else \
            AttributeMerger()
        self._tag_cleaner = tag_cleaner if tag_cleaner is not None else \
            SubjectContextPreprocessor(transformers=[AttributeTagCleaner(attr_pattern=r'<[A-Z_]+>(.*)<[A-Z_]+/>')])
        self._concatenator = concatenator if concatenator is not None else \
            AttributeConcatenator()

    def extract(self, text):
        subj_context = self._subj_context_extractor.process(text)
        filtered_context = self._subj_context_preprocessor.process(subj_context)
        attributes = self._attribures_extractor.process(filtered_context)
        decomposition = self._text_annotator.process(filtered_context)
        udapi_doc = self._udapi_doc_creator.process(decomposition)
        filtered_doc = self._conllu_attributes_preprocessor.process(udapi_doc)
        filtered_doc_text = self._udapi_to_str_transformer.process(filtered_doc)
        attributes2 = self._items_tagger.process(filtered_doc_text)
        merged_attributes = self._attribute_merger.process(list(zip(attributes, attributes2)))
        subject_items = self._tag_cleaner.process(merged_attributes)
        subject = self._concatenator.process(subject_items)
        return subject