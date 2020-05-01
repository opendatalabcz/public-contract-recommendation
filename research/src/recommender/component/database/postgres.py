import pandas
import numpy

from recommender.component.base import Component


class DBManager(Component):

    def __init__(self, connection, load_query=None, **kwargs):
        super().__init__(**kwargs)
        self._connection = connection
        self._load_query = load_query

    def run_query(self, query, values=None):
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
        return data


class SubjectItemManager(DBManager):

    DEFAULT_LOAD_QUERY = 'select contract_id, item_desc, embedding from subject_item'

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        contract_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts)) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)), 'debug')
            contract_id = item[0]
            item_desc = item[1]
            lembedding = item[2]
            embedding = numpy.array(lembedding)
            contract = contract_items.get(contract_id,
                                          {'contract_id': contract_id, 'subject_items': [], 'embeddings': []})
            contract['subject_items'].append(item_desc)
            contract['embeddings'].append(embedding)
            contract_items[contract_id] = contract
        return pandas.DataFrame(contract_items.values())

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


class CPVItemManager(DBManager):

    DEFAULT_LOAD_QUERY = """select cntr.contract_id, cpv.name, cpv.embedding
                                from contract_cpv cntr join cpv_code cpv on cntr.cpv_id=cpv.id """

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        contract_cpv_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts)) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)), 'debug')
            contract_id = item[0]
            name = item[1]
            lembedding = item[2]
            embedding = numpy.array(lembedding)
            contract = contract_cpv_items.get(contract_id,
                                              {'contract_id': contract_id, 'cpv_items': [], 'embeddings': []})
            contract['cpv_items'].append(name)
            contract['embeddings'].append(embedding)
            contract_cpv_items[contract_id] = contract
        return pandas.DataFrame(contract_cpv_items.values())


class ContractItemManager(DBManager):

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        self._simngr = SubjectItemManager(connection)
        self._cpvimngr = CPVItemManager(connection)

    @staticmethod
    def merge_columns(row):
        l = row[0]
        r = row[1]
        l.extend(r)
        return l

    def load(self, parts=10):
        df_contract_items = self._simngr.load(parts)
        df_contract_cpv_items = self._cpvimngr.load(parts)
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


class ContractLocalityManager(DBManager):

    DEFAULT_LOAD_QUERY = """select c.contract_id, e.address, e.latitude, e.longitude
                              from contract c join
                                submitter s on c.submitter_id=s.submitter_id join
                                entity e on s.entity_id=e.entity_id"""

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        locations = []
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " contracts")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts)) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)), 'debug')
            contract_id = item[0]
            address = item[1]
            gps_coords = item[2], item[3]
            locations.append({'contract_id': contract_id, 'address': address, 'gps': gps_coords})
        return pandas.DataFrame(locations)


class EntitySubjectManager(DBManager):

    DEFAULT_LOAD_QUERY = 'select entity_id, description, embedding from entity_subject'

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        entity_subjects = {}
        total_records = len(raw_data)
        self.print("Loading total " + str(total_records) + " records")
        for i, item in enumerate(raw_data):
            if i % (int(total_records / parts) + 1) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_records)), 'debug')
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


class ContractEntitySubjectManager(DBManager):

    DEFAULT_LOAD_QUERY = """select c.contract_id, es.description, es.feature
                              from contract c join
                                submitter s on c.submitter_id=s.submitter_id join
                                entity e on s.entity_id=e.entity_id join
                                entity_subject es on e.entity_id=es.entity_id"""

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        entity_subjects = {}
        total_records = len(raw_data)
        self.print("Loading total " + str(total_records) + " records")
        for i, item in enumerate(raw_data):
            if i % (int(total_records / parts) + 1) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_records)), 'debug')
            contract_id = item[0]
            description = item[1]
            embedding = item[2]
            subject_items = entity_subjects.get(contract_id,
                                                {'contract_id': contract_id, 'entity_items': [],
                                                 'entity_embeddings': []})
            subject_items['entity_items'].append(description)
            subject_items['entity_embeddings'].append(embedding)
            entity_subjects[contract_id] = subject_items
        return pandas.DataFrame(entity_subjects.values())


class EntityManager(DBManager):

    DEFAULT_LOAD_QUERY = """
            select e.entity_id, e.dic, e.ico, e.name, e.address, e.latitude, e.longitude,
                array_agg(es.description) as items, array_agg(es.embedding) as embeddings
            from entity e left join entity_subject es on e.entity_id=es.entity_id
            group by e.entity_id"""

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        entities = {}
        total_entities = len(raw_data)
        self.print("Loading total " + str(total_entities) + " entities")
        for i, ent in enumerate(raw_data):
            if i % (int(total_entities / parts) + 1) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_entities)), 'debug')
            entity_id = ent[0]
            dic = ent[1]
            ico = ent[2]
            name = ent[3]
            address = ent[4]
            gps_coords = (ent[5], ent[6])
            items = ent[7]
            embeddings = ent[8]
            entity = entities.get(entity_id,
                                  {'dic': dic, 'ico': ico, 'name': name, 'address': address, 'gps': gps_coords,
                                   'entity_items': items, 'entity_embeddings': embeddings})
            entities[entity_id] = entity
        return pandas.DataFrame.from_dict(entities, orient='index')

    def _truncateDB(self):
        self.run_query('truncate table entity_subject')

    def save(self, df_entities):
        self._truncateDB()
        for index, row in df_entities.iterrows():
            entity_id = index
            address = row['address']
            gps_coords = row['gps']
            latitude, longitude = (gps_coords[0], gps_coords[1]) if gps_coords else (None, None)
            items = row['entity_items']
            embeddings = row['entity_embeddings']
            to_print = [str(x)[:5] if x is not None else '-   ' \
                        for x in [entity_id, address, latitude, longitude, len(items) if isinstance(items, set) else 0]]
            self.print("Updating entity {} with address {}, gps:({},{}), #items:{}".format(*to_print))
            cursor = self._connection.cursor()
            postgres_update_query = """UPDATE entity
                                        SET address=%s, latitude=%s, longitude=%s
                                        WHERE entity_id=%s"""
            record_to_insert = (address, latitude, longitude, entity_id)
            cursor.execute(postgres_update_query, record_to_insert)

            self._connection.commit()
            cursor.close()

            if not items:
                continue
            for i, (item, embedding) in enumerate(zip(items, embeddings)):
                cursor = self._connection.cursor()
                postgres_insert_query = """INSERT INTO entity_subject (entity_id, description, embedding)
                                            VALUES (%s,%s,%s)"""
                record_to_insert = (entity_id, item, embedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                cursor.close()


class InterestItemManager(DBManager):

    DEFAULT_LOAD_QUERY = 'select user_id, item_desc, embedding from interest_item'

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        user_profile_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts) + 1) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)), 'debug')
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


class UserProfileManager(DBManager):

    DEFAULT_LOAD_QUERY = """select u.user_id, u.address, u.latitude, u.longitude, i.item_desc, i.feature
                                from user_profile u join
                                interest_item i on u.user_id=i.user_id"""

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        user_profile_items = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " items")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts) + 1) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)), 'debug')
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
        return pandas.DataFrame(user_profile_items.values())

    def _truncateDB(self):
        self.run_query('truncate table interest_item')
        self.run_query('truncate table user_profile cascade')

    def save(self, df_user_profile):
        self._truncateDB()
        for index, row in df_user_profile.iterrows():
            address = row['address']
            latitude, longitude = row['gps'][0], row['gps'][1]
            user_id = self.run_query("""insert into user_profile(address, latitude, longitude)
                                        values (%s,%s,%s) returning user_id;""", \
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


class DocumentManager(DBManager):

    DEFAULT_LOAD_QUERY = """select document_id, contract_id, data from document where processed=True"""

    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        if self._load_query is None:
            self._load_query = self.DEFAULT_LOAD_QUERY

    def load(self, parts=10):
        self.print("Running query: " + self._load_query, 'debug')
        raw_data = self.run_query(self._load_query)

        contract_documents = {}
        total_items = len(raw_data)
        self.print("Loading total " + str(total_items) + " documents")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts)) == 0:
                self.print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)), 'debug')
            doc_id = item[0]
            contract_id = item[1]
            doc_text = item[2]
            contract = contract_documents.get(contract_id,
                                          {'contract_id': contract_id, 'doc_ids': [], 'doc_texts': []})
            contract['doc_ids'].append(doc_id)
            contract['doc_texts'].append(doc_text)
            contract_documents[contract_id] = contract
        return pandas.DataFrame(contract_documents.values())
