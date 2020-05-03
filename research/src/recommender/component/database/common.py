import pandas

from recommender.component.base import Component


class ContractItemDAO(Component):

    def __init__(self, subject_item_dao, cpv_item_dao, **kwargs):
        super().__init__(**kwargs)
        self._subject_item_dao = subject_item_dao
        self._cpv_item_dao = cpv_item_dao

    @staticmethod
    def merge_columns(row):
        l = row[0]
        r = row[1]
        l.extend(r)
        return l

    def load(self, condition=None):
        df_contract_items = self._subject_item_dao.load(condition=condition)
        df_contract_cpv_items = self._cpv_item_dao.load(condition=condition)
        df_contract_items_merged = pandas.merge(df_contract_items, df_contract_cpv_items, how='outer', on='contract_id')
        df_contract_items_merged['embeddings_x'] = df_contract_items_merged['embeddings_x'].apply(
            lambda x: x if isinstance(x, list) else [])
        df_contract_items_merged['embeddings_y'] = df_contract_items_merged['embeddings_y'].apply(
            lambda x: x if isinstance(x, list) else [])
        df_contract_items_merged['subject_items'] = df_contract_items_merged['subject_items'].apply(
            lambda x: x if isinstance(x, list) else [])
        df_contract_items_merged['cpv_items'] = df_contract_items_merged['cpv_items'].apply(
            lambda x: x if isinstance(x, list) else [])
        df_contract_items_merged['embeddings'] = df_contract_items_merged[['embeddings_x', 'embeddings_y']].apply(
            self.merge_columns, axis=1)
        df_contract_items_merged['items'] = df_contract_items_merged[['subject_items', 'cpv_items']].apply(
            self.merge_columns, axis=1)
        return df_contract_items_merged.drop(columns=['embeddings_x', 'embeddings_y', 'subject_items', 'cpv_items'])


class ContractDataDAO(Component):

    def __init__(self, contact_dao, cpv_dao, item_dao, locality_dao, entity_subject_dao, **kwargs):
        super().__init__(**kwargs)
        self.contract_dao = contact_dao
        self.cpv_dao = cpv_dao
        self.item_dao = item_dao
        self.locality_dao = locality_dao
        self.entity_subject_dao = entity_subject_dao

    def load(self, condition=None):
        df_contracts = self.contract_dao.load(condition=condition)
        df_cpvs = self.cpv_dao.load(condition=condition).drop(columns=['embeddings'])
        df_items = self.item_dao.load(condition=condition)
        df_locality = self.locality_dao.load(condition=condition)
        df_entity_subject = self.entity_subject_dao.load(condition=condition)

        df_contracts = pandas.merge(df_contracts, df_cpvs, how='left', on='contract_id')
        df_contracts = pandas.merge(df_contracts, df_items, how='left', on='contract_id')
        df_contracts = pandas.merge(df_contracts, df_locality, how='left', on='contract_id')
        df_contracts = pandas.merge(df_contracts, df_entity_subject, how='left', on='contract_id')
        return df_contracts


class DBManager(Component):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.daos = {}

    def create_manager(self, dao_class, **kwargs):
        return dao_class(self._connection, logger=self.logger, **kwargs)

    def get(self, dao_class):
        if dao_class not in self.daos:
            self.daos[dao_class] = self.create_manager(dao_class)
        return self.daos[dao_class]
