from __future__ import absolute_import, division, unicode_literals
import io
import numpy as np
from os import path
import hashlib
import json
import senteval

from transformers import BertTokenizer, BertModel, TFBertModel

from .evaluation import torch_embedding, tf_embedding, get_wordvec, EMBEDDINGS_path, STS_path, params_senteval


# SentEval prepare and batcher
class Prepare:

    def __init__(self, path_to_vec, wvec_dim=768):
        self.path_to_vec = path_to_vec
        self.wvec_dim = wvec_dim

    def run(self, params, sentences):
        params.sent2id = sentences
        params.word_vec = get_wordvec(self.path_to_vec, params.sent2id)
        params.wvec_dim = self.wvec_dim
        return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        if sent in params.word_vec:
            sentvec.append(params.word_vec[sent])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


class Job():

    def __init__(self,
                 test='STSCZ',
                 model_spec='bert-base-multilingual-cased',
                 technology='torch',
                 name=None,
                 sentences=None,
                 lang='CZ',
                 datasets=None,
                 embed_with=None,
                 embeddings=None,
                 path_to_embeddings=None,
                 tokenizer=None,
                 model=None,
                 wvec_dim=768):
        self.test = test
        self.model_spec = model_spec
        self.technology = technology
        self.name = \
            name \
                if name is not None else \
                '_'.join([test, model_spec, technology])
        self.sentences = sentences
        self.lang = lang
        self.datasets = datasets \
            if datasets is not None else \
            ['headlines', 'headlines2013-2015CZ_Lemma', 'headlines2013-2015CZ_POSTag',
               'headlines2013-2015CZ_Stem',
               'images', 'imagesCZ2014-2015shuffled', 'imagesCZ2014-2015shuffled_Lemma',
               'imagesCZ2014-2015shuffled_POSTag', 'imagesCZ2014-2015shuffled_Stem']
        self.embeddings = embeddings
        self.path_to_embeddings = \
            path_to_embeddings \
                if path_to_embeddings is not None else \
                EMBEDDINGS_path + '/' + self.name
        if embed_with is None:
            self.embed = \
                torch_embedding \
                    if technology == 'torch' else \
                    tf_embedding
        else:
            self.embed = embed_with
        self.tokenizer = tokenizer
        self.model = model
        self.wvec_dim = wvec_dim
        self.results = None

    def get_sentences(self):
        if self.sentences is not None:
            return self.sentences

        print('Getting sentences for ' + self.name)
        sentences = []

        fpath = STS_path + '/' + self.test + ('-cz-test' if self.lang == 'CZ' else '-en-test')
        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                                 io.open(fpath + '/STS.input.%s.txt' % dataset,
                                         encoding='utf8').read().splitlines()])
            sentences += sent1 + sent2

        self.sentences = sentences
        return self.sentences

    def get_embeddings(self):
        if self.embeddings is not None:
            return self.embeddings
        if path.exists(self.path_to_embeddings):
            self.embeddings = self.load_embeddings()
            return self.embeddings

        if self.tokenizer is None:
            print('Loading tokenizer for ' + self.name)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_spec)
        if self.model is None:
            print('Loading model for ' + self.name)
            self.model = \
                BertModel.from_pretrained(self.model_spec) \
                    if self.technology == 'torch' else \
                    TFBertModel.from_pretrained(self.model_spec)
        print('Processing embeddings for ' + self.name)
        embeddings = {}
        for i, sent in enumerate(list(self.get_sentences())):
            hashed = hashlib.md5(sent.encode('utf-8')).hexdigest()
            embeddings[hashed] = self.embed(sent, self.model, self.tokenizer)
            print("%i/%i:%s" % (i, len(self.get_sentences()), sent))
        self.embeddings = embeddings
        self.save_embeddings()
        return self.embeddings

    def load_embeddings(self):
        print('Loading embeddings for ' + self.name)
        embeddings = {}
        with io.open(EMBEDDINGS_path + '/' + self.name, 'r', encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                embeddings[word] = np.fromstring(vec, sep=' ')
        return embeddings

    def save_embeddings(self):
        embeddings = self.get_embeddings()
        print('Saving embeddings for ' + self.name)
        with io.open(self.path_to_embeddings, 'w', encoding='utf-8') as f:
            for token in embeddings:
                tokens = [token] + [str(x) for x in embeddings[token]]
                f.write(' '.join(tokens) + '\n')

    def run_evaluation(self):
        if not path.exists(self.path_to_embeddings):
            self.save_embeddings()
        print('Running evaluation for ' + self.name)
        se = senteval.engine.SE(params_senteval, batcher, Prepare(self.path_to_embeddings, wvec_dim=self.wvec_dim))
        transfer_tasks = [self.test]
        self.results = se.eval(transfer_tasks)
        return self.results

    def get_results(self):
        if self.results is not None:
            return self.results
        return self.run_evaluation()

    def save_results(self):
        results = self.get_results()
        print('Saving results for ' + self.name)
        with io.open(EMBEDDINGS_path + '/results.json', 'a', encoding='utf-8') as f:
            data = {self.name: results}
            json_data = json.dumps(data)
            f.write(json_data + '\n')
            return json_data
