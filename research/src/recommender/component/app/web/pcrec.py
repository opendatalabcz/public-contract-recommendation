import configparser
import logging
import os
import sys

import flask
from flask import flash, session
from flask_login import LoginManager
from flask_session import Session

from recommender.component.app.web import routes
from recommender.component.app.web.model import Contract, ContractFactory, Submitter, UserProfileFactory, User
from recommender.component.database.postgres import PostgresManager, PostgresContractDataDAO, SourceDAO, UserProfileDAO
from recommender.component.engine.engine import SearchEngine
from recommender.component.feature import RandomEmbedder

DEFAULT_CONFIG_FILE = 'config.cfg'


class PCRecWeb(flask.Flask):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcrec_config = config
        self.init_logger()
        self.init_db()
        self.init_engine()

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
        query = """
            select ico, array_agg(name) as names, array_agg(url) as urls
            from source
            where ico in %s
            group by ico"""
        edao = self.dbmanager.create_manager(SourceDAO, load_query=query)
        self.dbmanager.daos[SourceDAO] = edao
        self.logger.debug("Done")

    def init_engine(self):
        self.logger.info("Initializing engine")
        cddao = self.dbmanager.get(PostgresContractDataDAO)
        df_contracts = cddao.load()
        df_contracts = df_contracts.rename(columns={'subject_items': 'items'})
        path_to_model = self.pcrec_config.get('embedder', 'path')
        self.engine = SearchEngine(df_contracts, embedder=RandomEmbedder(logger=self.logger), num_results=10,
                                   logger=self.logger)
        self.logger.debug("Done")

    def get_contracts(self, contract_ids):
        cddao = self.dbmanager.get(PostgresContractDataDAO)
        df_contracts = cddao.load(condition=contract_ids)
        sorted_index = dict(zip(contract_ids, range(len(contract_ids))))
        df_contracts['contract_id_rank'] = df_contracts['contract_id'].map(sorted_index)
        df_contracts = df_contracts.sort_values('contract_id_rank')
        df_profiles = self.get_profiles(df_contracts['ico'].tolist())
        return ContractFactory.create_contracts(df_contracts, df_profiles)

    def get_profiles(self, icos):
        edao = self.dbmanager.get(SourceDAO)
        df_entities = edao.load(condition=icos)
        return df_entities

    def get_user_profiles(self, user_ids):
        updao = self.dbmanager.get(UserProfileDAO)
        df_user_profiles = updao.load(condition=user_ids)
        return UserProfileFactory.create_profiles(df_user_profiles)

    def load_user(self, user_id):
        user_profiles = self.get_user_profiles([user_id])
        if len(user_profiles) > 0:
            user_profile = user_profiles[0]
            return User(user_profile)
        return None

    def load_user_from_loginform(self, loginform):
        if loginform.icologin.data != '':
            pass
        elif loginform.idlogin.data != '':
            user = self.load_user(loginform.idlogin.data)
            if user:
                return user
        flash('Uživatel neexistuje!')

    def process_result(self, result):
        if not result:
            flash('Nenalezena žádná položka!')
            return []
        contract_ids = [res['contract_id'] for res in list(result.values())[0]]
        contracts = self.get_contracts(contract_ids)
        return contracts

    def search(self, form):
        searchquery = form.get_query()
        result = self.engine.query(searchquery)
        return self.process_result(result)

    def recommend(self, user_profile, query_params=None, nitems=10):
        df_user_profile = user_profile.to_pandas()
        result = self.engine.query_by_user_profile(df_user_profile, query_params)
        return self.process_result(result)[:nitems]

    @staticmethod
    def create_app(login_manager):
        cfg = PCRecWeb.create_config()
        app = PCRecWeb(cfg, import_name=__name__)
        # Check Configuration section for more details
        app.config['SESSION_TYPE'] = cfg.get('webapp', 'SESSION_TYPE')
        app.config['SECRET_KEY'] = cfg.get('webapp', 'SECRET_KEY')
        sess = Session()
        sess.init_app(app)
        routes.init_app(app)
        login_manager.init_app(app)
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

login_manager = LoginManager()
app = PCRecWeb.create_app(login_manager)

@login_manager.user_loader
def load_user(user_id):
    return app.load_user(user_id)
