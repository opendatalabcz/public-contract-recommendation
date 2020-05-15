import numpy
import pandas
import psycopg2

from recommender.component.base import Component
from recommender.component.database.common import DBManager, ContractDataDAO, ContractItemDAO


class PostgresManager(DBManager):
    """Manages the psycopg2 connection"""
    def __init__(self, dbname='postgres', user='postgres', password='admin', host='localhost', port='5432', **kwargs):
        super().__init__(**kwargs)
        self._connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)


class PostgresDAO(Component):
    """Abstract Postgres data access object.

    Uses psycopg2 implementation of connecction.
    """
    DEFAULT_QUERY = """select * from table"""
    DEFALUT_CONDITION = """where column in %s"""

    def __init__(self, connection, load_query=None, **kwargs):
        """
        Args:
            connection (Connection): psycopg2 connection
            load_query (str): default load query
        """
        super().__init__(**kwargs)
        self._connection = connection
        self._load_query = load_query if load_query is not None else self.DEFAULT_QUERY

    def build_query(self, values=None):
        """Build the default query with optional values
        Args:
            values (lsit): list of values

        Returns:
            str: full query
        """
        query = self._load_query
        if values is not None:
            if '%s' not in query:
                query += ' ' + self.DEFALUT_CONDITION
        return query

    def run_query(self, query=None, values=None):
        """Runs specific query, parses results and returns data.

        Args:
            query (str): specific query, if not set, default load query is used
            values: collection of values for the query

        Returns:
            fetched raw data from query result
        """
        query = query or self.build_query(values)
        self.print("Running query: {} with {}".format(query, values), 'debug')
        cursor = self._connection.cursor()
        cursor.execute(query, values)
        if 'select' in query.lower():
            data = cursor.fetchall()
        elif 'returning' in query.lower():
            data = cursor.fetchone()[0]
        else:
            data = True
        self._connection.commit()
        cursor.close()
        self.print("Result: {}".format(data if not isinstance(data, list) else len(data)), 'debug')
        return data

    def _process_result(self, raw_data):
        return raw_data

    def load(self, condition=None):
        """Runs the default laod query with optional condition

        Args:
            condition: condition parameters

        Returns:
            processed query result data
        """
        if isinstance(condition, list):
            condition = tuple([tuple(condition)])
        raw_data = self.run_query(values=condition)
        return self._process_result(raw_data)


class SubjectItemDAO(PostgresDAO):
    """Subject item data access object

    Provides loading and saving of the subject_item table.

    Transforms result to dataframe with:
        contract_id,
        subject_items,
        embeddings
    """
    DEFAULT_QUERY = 'select contract_id, item_desc, embedding from subject_item'
    DEFALUT_CONDITION = """where contract_id in %s"""

    def _process_result(self, raw_data):
        contract_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for item in raw_data:
            contract_id = item[0]
            item_desc = item[1]
            lembedding = item[2]
            embedding = numpy.array(lembedding)
            contract = contract_items.get(contract_id,
                                          {'contract_id': contract_id, 'subject_items': [], 'embeddings': []})
            contract['subject_items'].append(item_desc)
            contract['embeddings'].append(embedding)
            contract_items[contract_id] = contract
        return pandas.DataFrame(contract_items.values(), columns=['contract_id', 'subject_items', 'embeddings'])

    def _truncateDB(self):
        self.run_query('truncate table subject_item')

    def save(self, df_contract_items):
        self.print('Saving total: {} contracts'.format(len(df_contract_items.index)))
        for index, row in df_contract_items.iterrows():
            contract_id = row['contract_id']
            self.print('Saving contract: {}'.format(contract_id), 'debug')
            subject_items = row['subject_items']
            embeddings = row['embeddings']

            for i, item in enumerate(subject_items):
                self.print('    ' + item, 'debug')
                embedding = embeddings[i]
                lembedding = embedding.tolist()
                cursor = self._connection.cursor()

                postgres_insert_query = """INSERT INTO subject_item (contract_id, item_desc, embedding)
                                           VALUES (%s,%s,%s)"""
                record_to_insert = (contract_id, item, lembedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                cursor.close()


class CPVItemDAO(PostgresDAO):
    """CPV item data access object

    Provides loading from tables contract_cpv and cpv_code.

    Transforms result to dataframe with:
        contract_id,
        cpv_codes,
        cpv_items,
        embeddings
    """
    DEFAULT_QUERY = """
    select cntr.contract_id, cpv.code, cpv.name, cpv.embedding
    from contract_cpv cntr join cpv_code cpv on cntr.cpv_id=cpv.id """
    DEFALUT_CONDITION = """where contract_id in %s"""

    def _process_result(self, raw_data):
        contract_cpv_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for item in raw_data:
            contract_id = item[0]
            code = item[1]
            name = item[2]
            lembedding = item[3]
            embedding = numpy.array(lembedding)
            contract = contract_cpv_items.get(contract_id,
                                              {'contract_id': contract_id, 'cpv_codes': [], 'cpv_items': [],
                                               'embeddings': []})
            contract['cpv_codes'].append(code)
            contract['cpv_items'].append(name)
            contract['embeddings'].append(embedding)
            contract_cpv_items[contract_id] = contract
        return pandas.DataFrame(contract_cpv_items.values(),
                                columns=['contract_id', 'cpv_codes', 'cpv_items', 'embeddings'])


class ContractSubmitterDAO(PostgresDAO):
    """Contract submitter data access object

    Provides loading from tables contract, submitter and entity.

    Transforms result to dataframe with:
        contract_id,
        address,
        gps,
        ico,
        entity_name
    """
    DEFAULT_QUERY = """select c.contract_id, e.address, e.latitude, e.longitude, e.ico, e.name
                              from contract c join
                                submitter s on c.submitter_id=s.submitter_id join
                                entity e on s.entity_id=e.entity_id"""
    DEFALUT_CONDITION = """where contract_id in %s"""

    def _process_result(self, raw_data):
        locations = []
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " contracts")
        for item in raw_data:
            contract_id = item[0]
            address = item[1]
            gps_coords = item[2], item[3]
            ico = item[4]
            name = item[5]
            locations.append(
                {'contract_id': contract_id, 'address': address, 'gps': gps_coords, 'ico': ico, 'entity_name': name})
        return pandas.DataFrame(locations, columns=['contract_id', 'address', 'gps', 'ico', 'entity_name'])


class EntitySubjectDAO(PostgresDAO):
    """Entity subject data access object

    Provides loading and saving of table entity_subject.

    Transforms result to dataframe with:
        entity_id,
        entity_items,
        entity_embeddings
    """
    DEFAULT_QUERY = 'select entity_id, description, embedding from entity_subject'

    def _process_result(self, raw_data):
        entity_subjects = {}
        total_records = len(raw_data)
        self.print("Loading total " + str(total_records) + " records")
        for item in raw_data:
            entity_id = item[0]
            description = item[1]
            lembedding = item[2]
            embedding = numpy.array(lembedding)
            subject_items = entity_subjects.get(entity_id, {'entity_items': [], 'entity_embeddings': []})
            subject_items['entity_items'].append(description)
            subject_items['entity_embeddings'].append(embedding)
            entity_subjects[entity_id] = subject_items
        return pandas.DataFrame.from_dict(entity_subjects, orient='index')

    def _truncateDB(self):
        self.run_query('truncate table entity_subject')

    def save(self, df_entity_subject):
        self._truncateDB()
        for index, row in df_entity_subject.iterrows():
            entity_id = index
            self.print('Saving entity: {}'.format(entity_id), 'debug')
            subject_items = row['entity_items']
            embeddings = row['entity_embeddings']

            for i, (item, embedding) in enumerate(zip(subject_items, embeddings)):
                self.print('    ' + item, 'debug')
                lembedding = embedding.tolist()
                cursor = self._connection.cursor()

                postgres_insert_query = """INSERT INTO entity_subject (entity_id, description, embedding)
                                            VALUES (%s,%s,%s)"""
                record_to_insert = (entity_id, item, lembedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                cursor.close()


class ContractEntitySubjectDAO(PostgresDAO):
    """Contract entity subject data access object

    Provides loading from tables contract, submitter, entity and entity_subject.

    Transforms result to dataframe with:
        contract_id,
        entity_items,
        entity_embeddings
    """
    DEFAULT_QUERY = """select c.contract_id, es.description, es.embedding
                              from contract c join
                                submitter s on c.submitter_id=s.submitter_id join
                                entity e on s.entity_id=e.entity_id join
                                entity_subject es on e.entity_id=es.entity_id"""
    DEFALUT_CONDITION = """where contract_id in %s"""

    def _process_result(self, raw_data):
        entity_subjects = {}
        total_records = len(raw_data)
        self.print("Loading total " + str(total_records) + " records")
        for item in raw_data:
            contract_id = item[0]
            description = item[1]
            embedding = item[2]
            subject_items = entity_subjects.get(contract_id,
                                                {'contract_id': contract_id, 'entity_items': [],
                                                 'entity_embeddings': []})
            subject_items['entity_items'].append(description)
            subject_items['entity_embeddings'].append(embedding)
            entity_subjects[contract_id] = subject_items
        return pandas.DataFrame(entity_subjects.values(), columns=['contract_id', 'entity_items', 'entity_embeddings'])


class EntityDAO(PostgresDAO):
    """Entity data access object

    Provides loading from tables entity, entity_subject and source.

    Transforms result to dataframe with:
        ico,
        dic,
        name,
        address,
        gps,
        entity_items,
        entity_embeddings,
        names,
        urls
    """
    DEFAULT_QUERY = """
            select e.ico, e.dic, e.name, e.address, e.latitude, e.longitude,
                array_agg(es.description) as items, array_agg(es.embedding) as embeddings,
                array_agg(s.name) as names, array_agg(s.url) as urls
            from entity e
            left join entity_subject es on e.entity_id=es.entity_id
            left join source s on e.ico=s.ico
            group by e.ico"""

    def _process_result(self, raw_data):
        entities = {}
        total_entities = len(raw_data)
        self.print("Loading total " + str(total_entities) + " entities")
        for ent in raw_data:
            ico = ent[0]
            dic = ent[1]
            name = ent[2]
            address = ent[3]
            gps_coords = (ent[4], ent[5])
            items = ent[6]
            embeddings = ent[7]
            names = ent[8]
            urls = ent[9]
            entity = entities.get(ico,
                                  {'ico': ico, 'dic': dic, 'name': name, 'address': address, 'gps': gps_coords,
                                   'entity_items': items, 'entity_embeddings': embeddings, 'names': names,
                                   'urls': urls})
            entities[ico] = entity
        return pandas.DataFrame(entities.values(),
                                columns=['ico', 'dic', 'name', 'address', 'gps', 'entity_items', 'entity_embeddings',
                                         'names', 'urls'])


class SourceDAO(PostgresDAO):
    """Source data access object

    Provides loading from table source.

    Transforms result to dataframe with:
        ico,
        names,
        urls
    """
    DEFAULT_QUERY = """
    select ico, array_agg(name) as names, array_agg(url) as urls
    from source
    group by ico"""

    def _process_result(self, raw_data):
        entities = {}
        total_entities = len(raw_data)
        self.print("Loading total " + str(total_entities) + " entities")
        for ent in raw_data:
            ico = ent[0]
            names = ent[1]
            urls = ent[2]
            entity = entities.get(ico,
                                  {'ico': ico, 'names': names, 'urls': urls})
            entities[ico] = entity
        return pandas.DataFrame(entities.values(),
                                columns=['ico', 'names', 'urls'])


class InterestItemDAO(PostgresDAO):
    """Interest item data access object

    Provides loading and saving of table interest_item.

    Transforms result to dataframe with:
        user_id,
        interest_items,
        embeddings
    """
    DEFAULT_QUERY = 'select user_id, item_desc, embedding from interest_item'

    def _process_result(self, raw_data):
        user_profile_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for item in raw_data:
            user_id = item[0]
            item_desc = item[1]
            lembedding = item[2]
            embedding = numpy.array(lembedding)
            user_profile = user_profile_items.get(user_id, {'user_id': user_id, 'interest_items': [], 'embeddings': []})
            user_profile['interest_items'].append(item_desc)
            user_profile['embeddings'].append(embedding)
            user_profile_items[user_id] = user_profile
        return pandas.DataFrame(user_profile_items.values())

    def _truncateDB(self):
        self.run_query('truncate table interest_item')

    def save(self, df_user_profile):
        self._truncateDB()
        for index, row in df_user_profile.iterrows():
            user_id = row['user_id']
            self.print('Saving user: {}'.format(user_id), 'debug')
            interest_items = row['interest_items']
            embeddings = row['embeddings']

            for i, item in enumerate(interest_items):
                self.print('    ' + item, 'debug')
                embedding = embeddings[i]
                lembedding = embedding.tolist()
                cursor = self._connection.cursor()

                postgres_insert_query = """INSERT INTO interest_item (user_id, item_desc, embedding)
                                            VALUES (%s,%s,%s)"""
                record_to_insert = (user_id, item, lembedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                cursor.close()


class UserProfileDAO(PostgresDAO):
    """User profile item data access object

    Provides loading and saving of table user_profile and interest_item.

    Transforms result to dataframe with:
        user_id,
        address,
        gps,
        interest_items,
        embeddings
    """
    DEFAULT_QUERY = """select u.user_id, u.address, u.latitude, u.longitude, i.item_desc, i.embedding
                                from user_profile u join
                                interest_item i on u.user_id=i.user_id"""
    DEFALUT_CONDITION = """where u.user_id in %s"""

    def _process_result(self, raw_data):
        user_profile_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for item in raw_data:
            user_id = item[0]
            address = item[1]
            gps = (item[2], item[3])
            item_desc = item[4]
            lembedding = item[5]
            embedding = numpy.array(lembedding)
            user_profile = user_profile_items.get(user_id, {'user_id': user_id, 'address': address, 'gps': gps,
                                                            'interest_items': [], 'embeddings': []})
            user_profile['interest_items'].append(item_desc)
            user_profile['embeddings'].append(embedding)
            user_profile_items[user_id] = user_profile
        return pandas.DataFrame(user_profile_items.values(),
                                columns=['user_id', 'address', 'gps', 'interest_items', 'embeddings'])

    def _truncateDB(self):
        self.run_query('truncate table interest_item')
        self.run_query('truncate table user_profile cascade')

    def save(self, df_user_profile):
        self._truncateDB()
        for index, row in df_user_profile.iterrows():
            address = row['address']
            latitude, longitude = row['gps'][0], row['gps'][1]
            user_id = self.run_query("""insert into user_profile(address, latitude, longitude)
                                        values (%s,%s,%s) returning user_id;""",
                                     (address, latitude, longitude))
            self.print('Saving user: {}'.format(user_id), 'debug')
            interest_items = row['interest_items']
            embeddings = row['embeddings']

            for i, item in enumerate(interest_items):
                self.print('    ' + item, 'debug')
                embedding = embeddings[i]
                lembedding = embedding.tolist()
                cursor = self._connection.cursor()

                postgres_insert_query = """INSERT INTO interest_item (user_id, item_desc, embedding)
                                            VALUES (%s,%s,%s)"""
                record_to_insert = (user_id, item, lembedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                cursor.close()


class DocumentDAO(PostgresDAO):
    """Document data access object

    Provides loading from table document.

    Transforms result to dataframe with:
        contract_id,
        doc_ids,
        doc_texts
    """
    DEFAULT_QUERY = """select document_id, contract_id, data from document where processed=True"""
    DEFALUT_CONDITION = """where contract_id in %s"""

    def _process_result(self, raw_data):
        contract_documents = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " documents")
        for item in raw_data:
            doc_id = item[0]
            contract_id = item[1]
            doc_text = item[2]
            contract = contract_documents.get(contract_id,
                                              {'contract_id': contract_id, 'doc_ids': [], 'doc_texts': []})
            contract['doc_ids'].append(doc_id)
            contract['doc_texts'].append(doc_text)
            contract_documents[contract_id] = contract
        return pandas.DataFrame(contract_documents.values())


class ContractDAO(PostgresDAO):
    """Contract data access object

    Provides loading from table contract.

    Transforms result to dataframe with:
        contract_id,
        code1,
        code2,
        name
    """
    DEFAULT_QUERY = """select contract_id, code1, code2, name from contract"""
    DEFALUT_CONDITION = """where contract_id in %s"""

    def _process_result(self, raw_data):
        contracts = []
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " contracts")
        for item in raw_data:
            contract_id = item[0]
            code1 = item[1]
            code2 = item[2]
            name = item[3]
            contracts.append({'contract_id': contract_id, 'code1': code1, 'code2': code2, 'name': name})
        return pandas.DataFrame(contracts, columns=['contract_id', 'code1', 'code2', 'name'])


class PostgresContractItemDAO(ContractItemDAO):
    """Contract item data access object

    Merges subject item DAO and CPV item DAO.
    """
    def __init__(self, source, subject_item_dao=None, cpv_item_dao=None, **kwargs):
        subject_item_dao = subject_item_dao if subject_item_dao is not None else SubjectItemDAO(source, **kwargs)
        cpv_item_dao = cpv_item_dao if cpv_item_dao is not None else CPVItemDAO(source, **kwargs)
        super().__init__(subject_item_dao, cpv_item_dao, **kwargs)


class PostgresContractDataDAO(ContractDataDAO):
    """Contract data data access object

    Merges contract DAO, CPV item DAO, subject item DAO, contract submitter DAO and contract entity subject DAO.
    """
    def __init__(self, source, contact_dao=None, cpv_dao=None, item_dao=None, locality_dao=None,
                 entity_subject_dao=None, **kwargs):
        contract_dao = contact_dao if contact_dao is not None else ContractDAO(source, **kwargs)
        cpv_dao = cpv_dao if cpv_dao is not None else CPVItemDAO(source, **kwargs)
        item_dao = item_dao if item_dao is not None else SubjectItemDAO(source, **kwargs)
        locality_dao = locality_dao if locality_dao is not None else ContractSubmitterDAO(source, **kwargs)
        entity_subject_dao = entity_subject_dao if entity_subject_dao is not None else ContractEntitySubjectDAO(source,
                                                                                                                **kwargs)
        super().__init__(contract_dao, cpv_dao, item_dao, locality_dao, entity_subject_dao, **kwargs)
