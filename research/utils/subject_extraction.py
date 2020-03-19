import json
import random
import string
import os
import numpy
import conllu

def randomTokens():
    """Generate a random string of fixed length """
    tokensLength = int(random.gauss(4,2))+1
    letters = string.ascii_lowercase
    tokens = []
    for _ in range(tokensLength):
        stringLength = int(random.gauss(6,3)+1)
        tokens.append(''.join(random.choice(letters) for i in range(stringLength)))
    return tokens

def save_connlu_to_file(connlu_data, path='../test-data/conllu_temp.txt'):
    if isinstance(connlu_data, list):
        connlu_data = '\n'.join([sentence.serialize() for sentence in connlu_data])
    if isinstance(connlu_data, conllu.models.TokenList):
        connlu_data = connlu_data.serialize()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(connlu_data)

class SubjectExtractor:
    
    def extract(self):
        return ['ahoj jak se mas']

class RandomSubjectExtractor(SubjectExtractor):
    
    def extract(self):
        subjects_num = int(numpy.random.exponential())+1
        return [randomTokens() for _ in range(subjects_num)]
    
class ReferenceSubjectExtractor(SubjectExtractor):
    
    _REF_FILENAME = '_ref.json'
    _path = None
    _ref = None
    _ref_subjects = None
    
    def __init__(self, path):
        self._path = path
    
    def extract(self):
        filename = os.path.join(self._path, self._REF_FILENAME)
        with open(filename, encoding='utf-8') as f:
            data = f.read()
            self._ref = json.loads(data)
            self._ref_subjects = [item['name'] for item in self._ref['subject']['items']]
        return self._ref_subjects
    
class ReferenceSubject2Extractor(ReferenceSubjectExtractor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extract(self):
        filename = os.path.join(self._path, self._REF_FILENAME)
        with open(filename, encoding='utf-8') as f:
            data = f.read()
            self._ref = json.loads(data)
            self._ref_subject = self._ref['subject2'] if 'subject2' in self._ref else None
        return self._ref_subject
    
class ReferenceSubjectContextExtractor(ReferenceSubjectExtractor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extract(self):
        filename = os.path.join(self._path, self._REF_FILENAME)
        with open(filename, encoding='utf-8') as f:
            data = f.read()
            self._ref = json.loads(data)
            self._ref_subject = self._ref['subject_context'] if 'subject_context' in self._ref else None
        return self._ref_subject
