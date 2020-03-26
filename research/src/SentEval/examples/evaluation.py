from __future__ import absolute_import, division, unicode_literals
import io
import numpy as np
from os import path
import senteval
import logging
import json

import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel, TFBertModel

# Set PATHs
PATH_TO_SENTEVAL = '..'
PATH_TO_DATA = PATH_TO_SENTEVAL + '/data'
STS_path = PATH_TO_SENTEVAL + '/data/downstream/STS'
EMBEDDINGS_path = PATH_TO_SENTEVAL + '/examples/embeddings'

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id): #TODO hierarchizace slov podle vet
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
class Prepare:

    def __init__(self, path_to_vec, wvec_dim=768, create_dictionary=create_dictionary):
        self.path_to_vec = path_to_vec
        self.wvec_dim = wvec_dim
        self.create_dictionary = create_dictionary

    def run(self, params, samples):
        _, params.word2id = self.create_dictionary(samples)
        params.word_vec = get_wordvec(self.path_to_vec, params.word2id)
        params.wvec_dim = self.wvec_dim
        return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word]) #TODO vyber podle vety, ne jen podle slowa
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

def torch_embedding(token, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(token)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states.detach().numpy()[0][0]


def tf_embedding(token, model, tokenizer):
    input_ids = tf.constant(tokenizer.encode(token))[None, :]  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states.numpy()[0][0]

class Job:

    def __init__(self,
                 test='STSCZ',
                 model_spec='bert-base-multilingual-cased',
                 technology='torch',
                 name=None,
                 dictionary=None,
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
        self.dictionary = dictionary
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

    def get_dictionary(self):
        if self.dictionary is not None:
            return self.dictionary

        print('Getting dictionary for ' + self.name)
        data = {}
        samples = []

        fpath = STS_path + '/' + self.test + ('-cz-test' if self.lang == 'CZ' else '-en-test')
        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                                 io.open(fpath + '/STS.input.%s.txt' % dataset,
                                         encoding='utf8').read().splitlines()])

            sent1 = np.array([s.split() for s in sent1])
            sent2 = np.array([s.split() for s in sent2])
            sorted_data = sorted(zip(sent1, sent2),
                                 key=lambda z: (len(z[0]), len(z[1])))
            sent1, sent2 = map(list, zip(*sorted_data))
            data[dataset] = (sent1, sent2)
            samples += sent1 + sent2

        words = {}
        for sent in samples:
            for word in sent:
                words[word] = words.get(word, 0) + 1

        self.dictionary = words
        return self.dictionary

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
        for i, token in enumerate(list(self.get_dictionary().keys())):
            embeddings[token] = self.embed(token, self.model, self.tokenizer)
            print("%i/%i:%s" % (i, len(self.dictionary), token))
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
