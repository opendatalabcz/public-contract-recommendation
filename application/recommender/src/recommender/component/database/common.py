import pandas

from recommender.component.base import Component


class ContractItemDAO(Component):
    """Contract item data access object

    Merges subject item DAO and CPV item DAO.

    Transforms result to dataframe with:
        contract_id,
        items,
        embeddings
    """
    def __init__(self, subject_item_dao, cpv_item_dao, **kwargs):
        super().__init__(**kwargs)
        self._subject_item_dao = subject_item_dao
        self._cpv_item_dao = cpv_item_dao

    @staticmethod
    def merge_columns(row):
        """Merges two columns containing lists of items to one list.

        Args:
            row (Series): pandas dataframe record with two columns to be merged

        Returns:
            list: merged columns
        """
        l = row[0]
        r = row[1]
        l.extend(r)
        return l

    def load(self, condition=None):
        """Runs the partial DAO queries with optional condition and merges their results.

        Merges the results of partial queries to single column.

        Args:
            condition: condition parameters

        Returns:
            merged records
        """
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
    """Contract data data access object

    Merges contract DAO, CPV item DAO, subject item DAO, contract submitter DAO and contract entity subject DAO.

    Transforms result to dataframe with:
        contract_id,
        code1,
        code2,
        name,
        cpv_codes,
        cpv_items,
        subject_items,
        embeddings,
        address,
        gps,
        ico,
        entity_name,
        entity_items,
        entity_embeddings
    """

    def __init__(self, contact_dao, cpv_dao, item_dao, locality_dao, entity_subject_dao, **kwargs):
        super().__init__(**kwargs)
        self.contract_dao = contact_dao
        self.cpv_dao = cpv_dao
        self.item_dao = item_dao
        self.locality_dao = locality_dao
        self.entity_subject_dao = entity_subject_dao

    def load(self, condition=None):
        """Runs the partial DAO queries with optional condition and merges their results.

        Merges the results of partial queries by contract_id.

        Args:
            condition: condition parameters

        Returns:
            merged records
        """
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
    """Manages data access objects.

    Attributes:
        daos (dict of class: any): data access objects class names to their instances
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.daos = {}

    def create_manager(self, dao_class, **kwargs):
        """Creates instance of DAO class.

        Args:
            dao_class (class): class of the specific DAO

        Returns:
            the instance of created DAO
        """
        return dao_class(self._connection, logger=self.logger, **kwargs)

    def get(self, dao_class):
        """Gets the instance of DAO class

        If the instance does not exist, creates one.

        Args:
            dao_class (class): class of the specific DAO

        Returns:
            the instance of the DAO
        """
        if dao_class not in self.daos:
            self.daos[dao_class] = self.create_manager(dao_class)
        return self.daos[dao_class]
