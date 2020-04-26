import pytest
import os
import configparser
import pandas

from recommender.component.feature.embedding import FastTextEmbedder

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
        self.df_user_profiles = None
        self.df_contracts = None

    def get_fasttext_embedder(self):
        if self.fasttext_embedder is None:
            self.fasttext_embedder = FastTextEmbedder(model=self.cfg['fasttext']['path_to_model'])
        return self.fasttext_embedder

    def get_user_profiles_data(self):
        if self.df_user_profiles is None:
            path = os.path.join(self.cfg['data']['path_to_data'], 'df_user_profiles.pickle')
            self.df_user_profiles = pandas.read_pickle(path)
        return self.df_user_profiles

    def get_contracts_data(self):
        if self.df_contracts is None:
            path = os.path.join(self.cfg['data']['path_to_data'], 'df_contracts.pickle')
            self.df_contracts = pandas.read_pickle(path)
        return self.df_contracts


@pytest.fixture
def context():
    return Context()
