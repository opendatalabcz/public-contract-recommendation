import numpy
import re
import pandas
import glob
import time

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


class DatabaseContractsLoader(DatabaseDocumentLoader):

    def __init__(self, connection, doc_pattern=None):
        super().__init__(connection)
        self._contracts = None
        self._doc_pattern = doc_pattern if doc_pattern is not None else \
            '''\n<FILE id={}>\n{}'''

    def prepare_documents(self, parts=10):
        contracts = {}
        total_docs = len(self._raw_data)
        print("Preparing total " + str(total_docs) + " documents")
        for i, doc in enumerate(self._raw_data):
            if i % (int(total_docs / parts)) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_docs)))
            doc_id = doc[0]
            contr_id = doc[1]
            doc_text = doc[8]

            contr_docs = contracts.get(contr_id, {'docs': []})
            contr_docs['docs'].append({'id': doc_id, 'text': doc_text})
            contracts[contr_id] = contr_docs
        self._contracts = contracts
        return self._contracts

    def prepare_contracts(self, parts=10):
        total_contrs = len(self._contracts)
        print("Preparing total " + str(total_contrs) + " contracts")
        for i, contr_id in enumerate(self._contracts):
            if i % (int(total_contrs / parts)) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_contrs)))
            contr_docs = self._contracts[contr_id]['docs']
            contr_text = ''.join(self._doc_pattern.format(doc['id'], doc['text']) \
                                 for doc in contr_docs)
            self._contracts[contr_id]['text'] = contr_text
        return self._contracts


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


class StringFinder:
    _pattern = None
    _lower = None
    _show_prec_range = None
    _show_next_range = None

    def __init__(self, keyword, lower=True, show_range=30):
        self._pattern = keyword
        self._lower = lower
        self._show_prec_range = show_range[0] if isinstance(show_range, tuple) else show_range
        self._show_next_range = show_range[1] if isinstance(show_range, tuple) else show_range

    def process(self, text):
        occurrences = find_all_occurrences_in_string(self._pattern, text, self._lower)
        findings = {}
        for occ in occurrences:
            findings[occ] = text[max(0, occ - self._show_prec_range): min(occ + self._show_next_range, len(text))]
        return findings


class SearchEngine:
    _finders = None
    _findings = None

    def __init__(self, by_string=None, by_regex=None, by_conllu=None, match='exact', show_range=30):
        self._finders = []
        self._findings = {}
        for keyword in by_string:
            lower = False
            if match == 'lower':
                lower = True
            self._finders.append(StringFinder(keyword, lower, show_range))

    def preprocess(self, data):
        data_collection = []
        if isinstance(data, str):
            data_collection.append(data)
        if isinstance(data, list) or \
                isinstance(data, pandas.core.series.Series):
            data_collection.extend([str(x) for x in data])
        return data_collection

    def process(self, data):
        texts = self.preprocess(data)
        for i, text in enumerate(texts):
            findings = {}
            for finder in self._finders:
                finding = finder.process(text)
                if finding:
                    findings[finder._pattern] = finding
            if findings:
                self._findings[i] = findings
        return self._findings


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    timers = dict()

    def __init__(
            self,
            name=None,
            text="{}: Elapsed time: {:0.4f} seconds",
            logger=print,
    ):
        self._start_time = None
        self.name = name
        self.text = text
        self.logger = logger

        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)

    def start(self):
        """Start a new timer"""

        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            self.logger(self.text.format(self.name, elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time


class DataProcessor:

    def __init__(self, timer=None):
        self._timer = Timer(timer if timer is not None else type(self).__name__)

    def _process_inner(self, data):
        return data

    def _process_inner_with_time(self, data):
        self._timer.start()
        result = self._process_inner(data)
        self._timer.stop()
        return result

    def process(self, data, timer=False):
        process_func = self._process_inner if not timer else self._process_inner_with_time
        if isinstance(data, list):
            return [process_func(item) for item in data]
        return process_func(data)
