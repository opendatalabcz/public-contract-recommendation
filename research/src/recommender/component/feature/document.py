import re

import numpy
import pandas

from recommender.component.base import DataProcessor


def count_occurence_vector(text, dim=500):
    arr = numpy.zeros(dim, numpy.int32)
    arr[0] = len(text)
    for c in text:
        arr[ord(c) % dim] += 1
    return arr


def char_ignore_mask(chars):
    occ = count_occurence_vector(chars)
    mask = numpy.zeros(occ.shape, numpy.int32)
    mask[occ == 0] = 1
    return mask


def find_all_occurrences_in_string(pattern, text, lower=True):
    if lower:
        if isinstance(pattern, str):
            pattern = pattern.lower()
        text = text.lower()
    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern)
    occurrences = [m.start() for m in pattern.finditer(text)]
    return occurrences


def chars_occurrence_ratio(text, chars='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž'):
    if len(text) == 0:
        return 0
    count_mask = count_occurence_vector(chars)
    count_mask[0] = 0
    vec = count_occurence_vector(text)
    chars_occ = vec * count_mask
    chars_num = chars_occ.sum()
    ratio = chars_num / len(text)
    return ratio


def get_most_frequent(List):
    dict = {}
    count, itm = 0, ''
    for item in reversed(List):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count:
            count, itm = dict[item], item
    return (itm)


def flatten_column(df, col):
    flat_col = pandas.DataFrame([[i, x]
                                 for i, y in df[col].apply(list).iteritems()
                                 for x in y], columns=list(['I', str(col) + '_flat']))
    flat_col = flat_col.set_index('I')
    return df.merge(flat_col, left_index=True, right_index=True)


def get_current_line(text, pos):
    end = text.find('\n', pos)
    if end == -1:
        end = len(text)
    inverse = text[::-1].find('\n', len(text) - pos)
    if inverse == -1:
        inverse = len(text)
    start = len(text) - inverse
    return text[start:end]


class DocumentMerger(DataProcessor):

    def __init__(self, doc_pattern=None, **kwargs):
        super().__init__(**kwargs)
        self._doc_pattern = doc_pattern if doc_pattern is not None else \
            '''\n<FILE id={}>\n{}'''

    def merge_docs(self, docs_id, docs_text):
        merged_text = ''.join(self._doc_pattern.format(doc_id, doc_text) \
                              for doc_id, doc_text in zip(docs_id, docs_text))
        return merged_text

    def _process_inner(self, data):
        return self.merge_docs(*data)