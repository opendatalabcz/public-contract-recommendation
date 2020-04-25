import pytest
import os
import configparser

from recommender.component.embedding.embedder import FastTextEmbedder

ABS_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIGS_PATH = ABS_PATH + '/configs'


def create_config(config_filename):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg_filename = config_filename

    if os.access(cfg_filename, os.R_OK):
        with open(cfg_filename) as f:
            cfg.read_file(f)
    return cfg


class Context:

    def __init__(self):
        self.cfg = create_config('config.cfg')
        self.fasttext_embedder = None

    def get_fasttext_embedder(self):
        if self.fasttext_embedder is None:
            self.fasttext_embedder = FastTextEmbedder(model=self.cfg['fasttext']['path_to_model'])
        return self.fasttext_embedder


@pytest.fixture
def context():
    return Context()
