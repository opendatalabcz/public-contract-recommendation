import numpy
import re
import pandas
import glob

from utils.text_extraction import *


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


class DatabaseDocumentLoader():

    def __init__(self, connection):
        self._connection = connection
        self._raw_data = None
        self._documents = None

    def load_documents(self, query=None):
        cur = self._connection.cursor()
        if query is None:
            query = "select * from document where processed=True"
        print("Running query: " + query)
        cur.execute(query)
        self._raw_data = cur.fetchall()
        cur.close()
        return self._raw_data

    def prepare_documents(self, parts=10):
        documents = []
        total_docs = len(self._raw_data)
        print("Preparing total " + str(total_docs) + " documents")
        for i, doc in enumerate(self._raw_data):
            if i % (int(total_docs / parts)) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_docs)))
            doc_id = doc[0]
            contr_id = doc[1]
            doc_url = doc[2]
            doc_text = doc[8]
            doc_vec = count_occurence_vector(doc_text)
            documents.append({'id': doc_id, 'contr_id': contr_id, 'url': doc_url, 'text': doc_text, 'vector': doc_vec})
        self._documents = documents
        return self._documents


class ReferenceDocumentLoader():

    def __init__(self, path_do_docs):
        self._path_to_docs = path_do_docs
        self._documents = None

    def extract_text_from_documents(self, min_file_size=0, max_file_size=5242880):
        docs_paths = glob.glob(self._path_to_docs)
        filtered_docs_paths = [path for path in docs_paths \
                               if not path.endswith('_ref.json') \
                               and os.stat(path).st_size > min_file_size \
                               and os.stat(path).st_size < max_file_size]
        extractionMachine = ExtractionMachine(save=True, filter_extracted=True)
        total_docs = len(filtered_docs_paths)
        print("Running extraction on total " + str(total_docs) + " documents")
        extractionMachine.extractFromDirs(filtered_docs_paths)
        extractions = extractionMachine._extractions
        self._documents = {extract: {'text': extractions[extract]} for extract in extractions}
        return self._documents

    def load_documents_from_extracts(self, ignore_names=[]):
        docs_paths = glob.glob(self._path_to_docs)
        extracts_paths = []
        for path in docs_paths:
            if not path.endswith('_ext'):
                continue
            if 'test_src' in path:
                continue
            ignore = False
            for contr_name in ignore_names:
                if contr_name in path:
                    ignore = True
                    break
            if ignore:
                continue
            extracts_paths.append(path)

        documents = {}
        total_extracts = len(extracts_paths)
        print("Loading total " + str(total_extracts) + " extracts")
        for path in extracts_paths:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                key = path[:-4]
                documents[key] = {'text': text}
        self._documents = documents
        return self._documents

    def prepare_documents(self, parts=10):
        total_docs = len(self._documents)
        print("Preparing total " + str(total_docs) + " documents")
        for i, doc in enumerate(self._documents):
            if i % (int(total_docs / parts)) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_docs)))
            doc_text = self._documents[doc]['text']
            self._documents[doc]['vector'] = count_occurence_vector(doc_text)
        return self._documents


class DocumentMatcher():

    def __init__(self, ref_documents, documents):
        self._ref_documents = ref_documents
        self._documents = documents
        self._aggregated_documents = None

    def count_most_similar_documents(self, parts=10, ignore_mask=char_ignore_mask('')):
        print("Collecting document matrix")
        doc_matrix = numpy.array([doc['vector'] for doc in self._documents])
        masked_doc_matrix = doc_matrix * ignore_mask

        total_docs = len(self._ref_documents)
        print("Counting most similar documents for total " + str(total_docs) + " reference documents")
        for i, ref_doc in enumerate(self._ref_documents):
            if i % (int(total_docs / parts)) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_docs)))
            ref_doc_text = self._ref_documents[ref_doc]['text']
            ref_vec = self._ref_documents[ref_doc]['vector']
            masked_ref_vec = ref_vec * ignore_mask
            diff = numpy.abs(masked_doc_matrix - masked_ref_vec)
            self._ref_documents[ref_doc]['diff'] = diff
            diff_sum = diff.sum(axis=1)
            self._ref_documents[ref_doc]['diff_sum'] = diff_sum
            self._ref_documents[ref_doc]['link'] = numpy.argsort(diff_sum)
            self._ref_documents[ref_doc]['top_link'] = self._ref_documents[ref_doc]['link'][0]
            self._ref_documents[ref_doc]['top_diff'] = self._ref_documents[ref_doc]['diff_sum'][
                self._ref_documents[ref_doc]['top_link']]
        return self._ref_documents

    def get_doc_for_file(self, file):
        for ref in self._ref_documents:
            if file in ref:
                most_similar = self._ref_documents[ref]['top_link']
                return self._documents[most_similar]

    def aggregate_documents(self):
        aggregated_documents = []
        for ref in self._ref_documents:
            ref_doc = self._ref_documents[ref]
            db_doc = self._documents[ref_doc['top_link']]
            path_parts = ref.split('\\')

            doc_id = db_doc['id']
            doc_name = path_parts[-1]
            contr_id = db_doc['contr_id']
            contr_name = '/'.join(path_parts[1:4])
            doc_text = db_doc['text']
            ref_text = ref_doc['text']
            doc_url = db_doc['url']
            doc_path = ref
            diff = ref_doc['top_diff']

            if ref_text.count('\n') / (len(ref_text) + 1) > 0.25:
                continue

            agg_doc = {'doc_id': doc_id, 'doc_name': doc_name, 'contr_id': contr_id, 'contr_name': contr_name,
                       'doc_text': doc_text, 'ref_text': ref_text, 'doc_url': doc_url, 'doc_path': doc_path,
                       'diff': diff}
            aggregated_documents.append(agg_doc)
        self._aggregated_documents = aggregated_documents
        return self._aggregated_documents

    def filter_aggregated(self, filter_ratio=0.1):
        df_aggregated = pandas.DataFrame(self._aggregated_documents)
        df_aggregated['ref_text_len'] = df_aggregated['ref_text'].str.len()
        df_aggregated['diff_ratio'] = df_aggregated['diff'] / df_aggregated['ref_text_len']
        df_contract = df_aggregated[df_aggregated.diff_ratio < filter_ratio]
        return df_contract
