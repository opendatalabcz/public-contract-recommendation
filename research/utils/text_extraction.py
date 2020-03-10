import sys
import importlib
import importlib.util

import tika
from tika import parser
from tika import language
from tika import config

import pdf2image

import pytesseract
from pathlib import Path
import os

class DocumentExtractor:

    _core = None
    
    def __init__(self, extractor):
        self._core = extractor
        
    def extract(self, path):
        data = self._preprocess(path)
        return self._process(data)
        
    def _preprocess(self, path):
        pass
    
    def _process(self, data):
        pass
    
class TikaDocExtractor(DocumentExtractor):
    
    def __init__(self):
        super().__init__(TikaTextExtractor())
        
    def _preprocess(self, path):
        return path
    
    def _process(self, path):
        return self._core.extract(path)

class PytesseractDocExtractor(DocumentExtractor):
    
    _dpi = None
    
    def __init__(self, dpi=300):
        super().__init__(PytesseractTextExtractor())
        if dpi is not None:
            self._dpi = dpi
        
    def _preprocess(self, path):
        return pdf2image.convert_from_path(path, self._dpi)
    
    def _process(self, pages):
        extractions = []
        for page in pages:
            extractions.append(self._core.extract(page))
        text = ''.join(extractions)            
        return text
    
class CombinedDocExtractor():
    
    _tika_core = None
    _pytesseract_core = None
    
    def __init__(self):
        self._tika_core = TikaDocExtractor()
        self._pytesseract_core = PytesseractDocExtractor()
    
    def extract(self, path):
        text = self._tika_core.extract(path)
        if text is None and path.lower().endswith('.pdf'):
            text = self._pytesseract_core.extract(path)
        return text
    
class CombinedDocExtractor2(CombinedDocExtractor):
    
    def extract(self, path):
        if path.lower().endswith('.pdf'):
            text = self._pytesseract_core.extract(path)
        else:
            text = self._tika_core.extract(path)
        return text
    
class TextExtractor:
    
    def __init__(self):
        pass
        
class TikaTextExtractor(TextExtractor):
    
    def extract(self, path):
        data = tika.parser.from_file(path, headers={'X-Tika-OCRLanguage': 'ces'})
        if data['status'] == 200:
            return data['content']
        else:
            return ''
    
class PytesseractTextExtractor(TextExtractor):
    
    def extract(self, image):
        return pytesseract.image_to_string(image, lang='ces')
    
class ExtractionMachine:
    
    _extractor = None
    _extractions = {}
    _save = None
    _filter_extracted = None
    
    def __init__(self, extractor=CombinedDocExtractor(), save=False, filter_extracted=False):
        self._extractor = extractor
        self._save = save
        self._filter_extracted = filter_extracted
    
    def extractFromDirs(self, dirs):
        for path in dirs:
            self.extractFromDir(path)
        return self._extractions
    
    def extractFromDir(self, path):
        filenames = []
        if os.path.isfile(path):
            filenames.append(path)
        else:
            filenames = [p.as_posix() for p in Path(path).rglob('*.*')]
        if self._filter_extracted:
            filenames = [p for p in filenames if not p.endswith('_ext')]
        for i, f in enumerate(filenames):
            print('{}/{}: {}'.format(i+1, len(filenames), f))
            self.extractDocument(f)
        return self._extractions
    
    def extractDocument(self, path):
        text = self._extractor.extract(path)
        if self._save:
            self.saveExtract(path, text)
        self._extractions[path] = text
            
    def saveExtract(self, path, text):
        with open(path+'_ext', 'w', encoding='utf-8') as f:
            f.write(text)
