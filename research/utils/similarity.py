import textdistance
import random
import numpy
import fasttext
import gensim
import editdistance

class SimilarityMachine:
    
    _text = None
    _reference = None
    _model = None
    
    def __init__(self, model=None):
        self._model = model
        
    def init_case(self, text, reference):
        self._text = text
        self._reference = reference
        
    def preprocess(self):
        None
    
    def compute(self, text, reference):
        self.init_case(text, reference)
        self.preprocess()
        return self._inner_compute()
    
    def _inner_compute(self):
        return 0.5
    
class RandomSimilarityMachine(SimilarityMachine):
    
    def _inner_compute(self):
        return random.random()
    
class JaccardSimilarityMachine(SimilarityMachine):
    
    def preprocess(self):
        if isinstance(self._text, str):
            self._text = self._text.split()
        if isinstance(self._reference, str):
            self._reference = self._reference.split()        

    def _inner_compute(self):
        return textdistance.jaccard(self._text, self._reference)

class SorensenDiceSimilarityMachine(JaccardSimilarityMachine):
    
    def _inner_compute(self):
        return textdistance.sorensen_dice(self._text, self._reference)

class FastTextSimilarityMachine(SimilarityMachine):
    
    def __init__(self, model=None):
        if model is None:
            path = 'fasttext/cc.cs.300.bin'
            print('Loading fasttext model from ' + path)
            model = fasttext.load_model(path)
        super().__init__(model)
    
    def _inner_compute(self):
        return 1/(numpy.linalg.norm(self._model[self._text] - self._model[self._reference])+1)
    
class GensimFastTextSimilarityMachine(SimilarityMachine):
    
    def __init__(self, model=None):
        if model is None:
            path = 'fasttext/cc.cs.300.bin'
            print('Loading fasttext model from ' + path)
            model = gensim.models.FastText.load_fasttext_format(path)
        super().__init__(model)
    
    def _inner_compute(self):
        return 1/(self._model.wmdistance(self._text, self._reference)+1)