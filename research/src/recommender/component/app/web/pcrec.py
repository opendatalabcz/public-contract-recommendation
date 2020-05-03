import configparser
import os
import logging
import sys

import flask
from flask_session import Session

from recommender.component.app.web import routes
from recommender.component.app.web.model import Contract, ContractFactory, Submitter
from recommender.component.database.postgres import PostgresManager, PostgresContractDataDAO
from recommender.component.engine.engine import SearchEngine
from recommender.component.feature import RandomEmbedder, FastTextEmbedder

DEFAULT_CONFIG_FILE = 'config.cfg'


class PCRecWeb(flask.Flask):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcrec_config = config
        self.init_logger()
        self.init_db()
        self.init_engine()
        self.contracts = {
            1: Contract(1, None, None, 'pěstování plodin', ['jablka', 'hrušky'], None,
                        Submitter('02589647', None, 'Čtveřín 60 Pěnčín u Liberce', None)),
            2: Contract(2, None, None, 'obchod s elektronikou', ['výpočetní technika', 'počítače'], None,
                        Submitter('36548210', None, 'V Jilmu 229 514 01 Jilemnice', None)),
        }

    def init_logger(self):
        level = self.pcrec_config.get('logger', 'level')
        file = self.pcrec_config.get('logger', 'file')
        logFormatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s")
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler(file)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        rootLogger.setLevel(level)
        self.logger = rootLogger
        self.logger.debug("Logger initialized!")

    def init_db(self):
        dbname = self.pcrec_config.get('db', 'name')
        user = self.pcrec_config.get('db', 'user')
        password = self.pcrec_config.get('db', 'password')
        host = self.pcrec_config.get('db', 'host')
        port = self.pcrec_config.get('db', 'port')
        self.logger.info("Initializing DB connection")
        self.dbmanager = PostgresManager(dbname=dbname, user=user, password=password, host=host, port=port,
                                         logger=self.logger)
        self.logger.debug("Done")

    def init_engine(self):
        self.logger.info("Initializing engine")
        cddao = self.dbmanager.get(PostgresContractDataDAO)
        df_contracts = cddao.load()
        df_contracts = df_contracts.rename(columns={'subject_items': 'items'})
        path_to_model = self.pcrec_config.get('embedder', 'path')
        self.engine = SearchEngine(df_contracts, embedder=FastTextEmbedder(path_to_model, logger=self.logger), num_results=10, logger=self.logger)
        self.logger.debug("Done")

    def get_contracts(self, contract_ids):
        cddao = self.dbmanager.get(PostgresContractDataDAO)
        df_contracts = cddao.load(condition=contract_ids)
        sorted_index = dict(zip(contract_ids, range(len(contract_ids))))
        df_contracts['contract_id_rank'] = df_contracts['contract_id'].map(sorted_index)
        df_contracts = df_contracts.sort_values('contract_id_rank')
        return ContractFactory.create_contracts(df_contracts)

    def search(self, form):
        searchquery = form.get_query()
        result = self.engine.query(searchquery)
        contract_ids = [res['contract_id'] for res in list(result.values())[0]]
        contracts = self.get_contracts(contract_ids)
        return contracts

    @staticmethod
    def create_app():
        cfg = PCRecWeb.create_config()
        app = PCRecWeb(cfg, import_name=__name__)
        # Check Configuration section for more details
        app.config['SESSION_TYPE'] = cfg.get('webapp', 'SESSION_TYPE')
        app.config['SECRET_KEY'] = cfg.get('webapp', 'SECRET_KEY')
        sess = Session()
        sess.init_app(app)
        routes.init_app(app)
        return app

    @staticmethod
    def create_config(config_filename=None):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg_filename = config_filename or DEFAULT_CONFIG_FILE

        if os.access(cfg_filename, os.R_OK):
            with open(cfg_filename) as f:
                cfg.read_file(f)
        return cfg

    @staticmethod
    def _error_page(error):
        return flask.render_template('error.html', error=error), error.code


app = PCRecWeb.create_app()
