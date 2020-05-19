import configparser
import logging
import os
import sys

import flask
import pandas
from flask import flash
from flask_login import LoginManager, current_user

from recommender.component.app.web import routes
from recommender.component.app.web.model import ContractFactory, UserProfileFactory, User, \
    InterestItem
from recommender.component.database.postgres import PostgresManager, PostgresContractDataDAO, SourceDAO, UserProfileDAO, \
    EntityDAO
from recommender.component.engine.engine import SearchEngine
from recommender.component.feature.embedding import FastTextEmbedder, RandomEmbedder

DEFAULT_CONFIG_FILE = 'config.cfg'


class PCRecWeb(flask.Flask):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcrec_config = config
        self.init_logger()
        self.init_db()
        self.init_engine()
        self.init_users()

    def init_logger(self):
        level = self.pcrec_config.get('logger', 'level')
        file = self.pcrec_config.get('logger', 'file')
        logFormatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s")
        logger = logging.getLogger(self.__class__.__name__)

        fileHandler = logging.FileHandler(file)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        self.logger = logger
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
        squery = """
            select ico, array_agg(name) as names, array_agg(url) as urls
            from source
            where ico in %s
            group by ico"""
        sdao = self.dbmanager.create_manager(SourceDAO, load_query=squery)
        self.dbmanager.daos[SourceDAO] = sdao
        equery = """
            select e.ico, e.dic, e.name, e.address, e.latitude, e.longitude,
                array_agg(es.description) as items, array_agg(es.embedding) as embeddings,
                null, null
            from entity e
            left join entity_subject es on e.entity_id=es.entity_id
            where e.ico in %s
            group by e.ico, e.dic, e.name, e.address, e.latitude, e.longitude"""
        equery = self.dbmanager.create_manager(EntityDAO, load_query=equery)
        self.dbmanager.daos[EntityDAO] = equery
        self.logger.debug("Done")

    def init_engine(self):
        self.logger.info("Initializing engine")
        cddao = self.dbmanager.get(PostgresContractDataDAO)
        df_contracts = cddao.load()
        df_contracts = df_contracts.rename(columns={'subject_items': 'items'})
        path_to_model = self.pcrec_config.get('embedder', 'path')
        self.engine = SearchEngine(df_contracts,
                                   embedder=FastTextEmbedder(path_to_model, logger=self.logger),
                                   # embedder=RandomEmbedder(logger=self.logger),
                                   num_results=10,
                                   random_bias_rate=0.0,
                                   logger=self.logger)
        self.logger.debug("Done")

    def init_users(self):
        self.cached_user_profiles = {}

    def get_contracts(self, contract_ids, similarities=None):
        cddao = self.dbmanager.get(PostgresContractDataDAO)
        df_contracts = cddao.load(condition=contract_ids)
        sorted_index = dict(zip(contract_ids, range(len(contract_ids))))
        df_contracts['contract_id_rank'] = df_contracts['contract_id'].map(sorted_index)
        df_contracts = df_contracts.sort_values('contract_id_rank').reset_index(drop=True)
        df_contracts['similarity'] = pandas.Series(similarities) if similarities else None
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

    def init_user_from_ico(self, icos):
        edao = self.dbmanager.get(EntityDAO)
        df_entity_profiles = edao.load(condition=icos)
        df_user_profiles = df_entity_profiles.rename(
            columns={'ico': 'user_id', 'entity_items': 'interest_items', 'entity_embeddings': 'embeddings'})
        return UserProfileFactory.create_profiles(df_user_profiles)

    def load_user(self, user_id):
        if user_id in self.cached_user_profiles:
            return User(self.cached_user_profiles[user_id])
        if isinstance(user_id, str) and len(user_id) == 8:
            user_profiles = self.init_user_from_ico([user_id])
        else:
            user_profiles = self.get_user_profiles([user_id])
        if len(user_profiles) > 0:
            user_profile = user_profiles[0]
            self.cached_user_profiles[user_id] = user_profile
            return User(user_profile)
        return None

    def load_user_from_loginform(self, loginform):
        login_data = loginform.icologin.data or loginform.idlogin.data
        user = self.load_user(login_data)
        if user:
            return user
        flash('Uživatel neexistuje!')

    def save_profile(self, profile_form):
        profile = current_user.user_profile
        address = profile_form.locality.data
        gps = self.engine.geocoder.gps_for_address(address)
        if not gps:
            flash('Adresa nenalezena')
        profile.locality.address = address
        profile.locality.gps = gps
        items = profile_form.interest_items.data.split('\n')
        embeddings = self.engine.embedder.process(items)
        profile.interest_items = [InterestItem(item, embedding) for item, embedding in zip(items, embeddings)]
        citems = profile_form.cached_items.data.split('\n')
        cembeddings = self.engine.embedder.process(items)
        profile.cache = [InterestItem(item, embedding) for item, embedding in zip(citems, cembeddings)]

    def update_user_profile(self, contract):
        profile = current_user.user_profile
        items = contract.subject_items
        embeddings = contract.embeddings
        profile.cache += [InterestItem(item, embedding) for item, embedding in zip(items, embeddings)]

    def process_result(self, result):
        if not result:
            flash('Nenalezena žádná položka!')
            return []
        contract_ids = [res['contract_id'] for res in list(result.values())[0]]
        similarities = [res['similarity'] for res in list(result.values())[0]]
        contracts = self.get_contracts(contract_ids, similarities)
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
        # app.config['SESSION_TYPE'] = cfg.get('webapp', 'SESSION_TYPE')
        app.config['SECRET_KEY'] = cfg.get('webapp', 'SECRET_KEY')
        routes.init_app(app)
        login_manager.init_app(app)
        return app

    @staticmethod
    def create_config(config_filename=None):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        if not config_filename:
            config_filename = os.getenv('PCREC_CONFIG', None)
        cfg_filename = config_filename or DEFAULT_CONFIG_FILE

        if os.access(cfg_filename, os.R_OK):
            with open(cfg_filename) as f:
                cfg.read_file(f)
        return cfg

    @staticmethod
    def _error_page(error):
        return flask.render_template('error.html', error=error), error.code

if __name__ == '__main__':
    login_manager = LoginManager()
    app = PCRecWeb.create_app(login_manager)
    @login_manager.user_loader
    def load_user(user_id):
        return app.load_user(user_id)

    @app.context_processor
    def inject_debug():
        return dict(debug=app.debug)


    @login_manager.user_loader
    def load_user(user_id):
        return app.load_user(user_id)
