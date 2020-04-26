import pandas

from recommender.component.base import Component
from recommender.component.database.postgres import ContractItemManager, ContractLocalityManager, \
    ContractEntitySubjectManager


class ContractDataManager(Component):

    def __init__(self, source, item_manager=None, locality_manager=None, entity_subject_manager=None):
        self.item_manager = item_manager if item_manager is not None else ContractItemManager(source)
        self.locality_manager = locality_manager if locality_manager is not None else ContractLocalityManager(source)
        self.entity_subject_manager = entity_subject_manager if entity_subject_manager is not None else ContractEntitySubjectManager(source)

    def load(self):
        df_items = self.item_manager.load()
        df_locality = self.locality_manager.load()
        df_entity_subject = self.entity_subject_manager.load()

        df_contracts = pandas.merge(df_items, df_locality, on='contract_id')
        df_contracts = pandas.merge(df_contracts, df_entity_subject, on='contract_id')
        return df_contracts
