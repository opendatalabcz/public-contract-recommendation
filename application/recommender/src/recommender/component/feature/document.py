import re

import numpy
import pandas

from recommender.component.base import DataProcessor


def count_occurrence_vector(text, dim=500) -> numpy.ndarray:
    """Count character occurrence vector

    Counts character occurrences in occurrence vector of given dimension,
    where each position represents one character.
    The first position represents sum of all occurrences (total count of chars).

    Args:
        text (str): input text
        dim (int): dimension of the occurrence vector

    Returns:
        ndarray: vector of occurrences
    """
    arr = numpy.zeros(dim, numpy.int32)
    arr[0] = len(text)
    for c in text:
        arr[ord(c) % dim] += 1
    return arr


def char_ignore_mask(chars):
    """Builds character ignore mask

    Using given characters builds an ignore mask as an vector of 1/0 flags,
    where 0 stands for ignored char.

    Args:
        chars (str): characters to be ignored

    Returns:
        ndarray: vector of ignore flags
    """
    occ = count_occurrence_vector(chars)
    mask = numpy.zeros(occ.shape, numpy.int32)
    mask[occ == 0] = 1
    return mask


def find_all_occurrences_in_string(pattern, text, lower=True, max_occurrences=1000):
    """Finds all starting positions of a pattern in text

    Args:
        pattern: either str or re.Pattern
        text (str): text to find the pattern in
        lower (bool): whether to find the pattern in original text or lower cased text
        max_occurrences (int): maximum number of occurrences

    Returns:

    """
    if lower:
        if isinstance(pattern, str):
            pattern = pattern.lower()
        text = text.lower()
    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern)
    occurrences = []
    for i, m in enumerate(pattern.finditer(text)):
        occurrences.append(m.start())
        if i > max_occurrences:
            break
    return occurrences


def chars_occurrence_ratio(text, chars='aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž'):
    """Computes the occurrence ratio of characters

    Args:
        text (str): text to count the chars occurrence ration in
        chars (str): chars to be count

    Returns:
        ndarray: vector representing the occurrence ratio of each character
    """
    if len(text) == 0:
        return 0
    count_mask = count_occurrence_vector(chars)
    count_mask[0] = 0
    vec = count_occurrence_vector(text)
    chars_occ = vec * count_mask
    chars_num = chars_occ.sum()
    ratio = chars_num / len(text)
    return ratio


def get_most_frequent(orig_list):
    """Finds the most frequent item in the list

    Args:
        orig_list (list): list of items

    Returns:
        the most frequent item
    """
    item_dict = {}
    count, itm = 0, ''
    for item in reversed(orig_list):
        item_dict[item] = item_dict.get(item, 0) + 1
        if item_dict[item] >= count:
            count, itm = item_dict[item], item
    return itm


def flatten_column(df, col):
    """Flattens the column of dataframe

    Args:
        df (DataFrame): original dataframe containing the column
        col (str): name of the column

    Returns:
        DataFrame: dataframe enriched with the flattened column
    """
    flat_col = pandas.DataFrame([[i, x]
                                 for i, y in df[col].apply(list).iteritems()
                                 for x in y], columns=list(['I', str(col) + '_flat']))
    flat_col = flat_col.set_index('I')
    return df.merge(flat_col, left_index=True, right_index=True)


def get_current_line(text, pos):
    """Gets a whole text of line corresponding to a specific position in the text

    Args:
        text (str): text to find the line in
        pos (int): position in the text

    Returns:
        str: line corresponding to the position
    """
    end = text.find('\n', pos)
    if end == -1:
        end = len(text)
    inverse = text[::-1].find('\n', len(text) - pos)
    if inverse == -1:
        inverse = len(text)
    start = len(text) - inverse
    return text[start:end]


class DocumentMerger(DataProcessor):
    """Text document merger

    Provides an interface for merging (concatenating) of text documents.
    """
    def __init__(self, doc_pattern=None, **kwargs):
        """
        Args:
            doc_pattern (str): formating string to merge the documents with
        """
        super().__init__(**kwargs)
        self._doc_pattern = doc_pattern if doc_pattern is not None else \
            '''\n<FILE id={}>\n{}'''

    def merge_docs(self, docs_id, docs_text):
        """Merges the text documents with usage of formating header.

        Args:
            docs_id (any): document identifier
            docs_text (list of str): list of document texts to be merged

        Returns:
            str: the merged text in a whole
        """
        merged_text = ''.join(self._doc_pattern.format(doc_id, doc_text)
                              for doc_id, doc_text in zip(docs_id, docs_text))
        return merged_text

    def _process_inner(self, data):
        return self.merge_docs(*data)
