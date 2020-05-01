import pandas
import numpy


class DBManager:

    def __init__(self, connection):
        self._connection = connection

    def runQuery(self, query, values):
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

    def __init__(self, connection):
        super().__init__(connection)
        self._load_query = 'select * from subject_item'

    def loadFromDB(self, parts=10):
        print("Running query: " + self._load_query)
        raw_data = self.runQuery(self._load_query)

        contract_items = {}
        total_items = len(raw_data)
        print("Loading total " + str(total_items) + " items")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts)) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)))
            item_id = item[0]
            contract_id = item[1]
            item_desc = item[2]
            lembedding = item[3]
            embedding = numpy.array(lembedding)
            contract = contract_items.get(contract_id,
                                          {'contract_id': contract_id, 'subject_items': [], 'embedding': []})
            contract['subject_items'].append(item_desc)
            contract['embedding'].append(embedding)
            contract_items[contract_id] = contract
        return pandas.DataFrame(contract_items.values())

    def _truncateDB(self):
        self.runQuery('truncate table subject_item')

    def saveToDB(self, df_contract_items):
        self._truncateDB()
        for index, row in df_contract_items.iterrows():
            contract_id = index
            print(contract_id)
            subject_items = row['subject_items']
            embeddings = row['embedding']

            for i, item in enumerate(subject_items):
                print('    ' + item)
                embedding = embeddings[i]
                lembedding = embedding.tolist()
                cursor = self._connection.cursor()

                postgres_insert_query = """INSERT INTO subject_item (contract_id, item_desc, embedding)
                                            VALUES (%s,%s,%s)"""
                record_to_insert = (contract_id, item, lembedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                count = cursor.rowcount
                cursor.close()


class UserProfileManager(DBManager):

    def __init__(self, connection):
        super().__init__(connection)
        self._load_query = 'select * from interest_item'

    def loadFromDB(self, parts=10):
        print("Running query: " + self._load_query)
        raw_data = self.runQuery(self._load_query)

        user_profile_items = {}
        total_items = len(raw_data)
        print("Loading total " + str(total_items) + " items")
        for i, item in enumerate(raw_data):
            if i % (int(total_items / parts) + 1) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_items)))
            item_id = item[0]
            user_id = item[1]
            item_desc = item[2]
            lembedding = item[3]
            embedding = numpy.array(lembedding)
            user_profile = user_profile_items.get(user_id, {'user_id': user_id, 'interest_items': [], 'embeddings': []})
            user_profile['interest_items'].append(item_desc)
            user_profile['embeddings'].append(embedding)
            user_profile_items[user_id] = user_profile
        return pandas.DataFrame(user_profile_items.values())

    def _truncateDB(self):
        self.runQuery('truncate table interest_item')
        self.runQuery('truncate table user_profile cascade')

    def saveToDB(self, df_user_profile):
        self._truncateDB()
        for index, row in df_user_profile.iterrows():
            user_id = self.runQuery('insert into user_profile default values returning user_id;')
            print(user_id)
            interest_items = row['interest_items']
            embeddings = row['embeddings']

            for i, item in enumerate(interest_items):
                print('    ' + item)
                embedding = embeddings[i]
                lembedding = embedding.tolist()
                cursor = self._connection.cursor()

                postgres_insert_query = """INSERT INTO interest_item (user_id, item_desc, embedding)
                                            VALUES (%s,%s,%s)"""
                record_to_insert = (user_id, item, lembedding)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                count = cursor.rowcount
                cursor.close()


class EntityManager(DBManager):

    def __init__(self, connection):
        super().__init__(connection)
        self._load_query = """
        select e.entity_id, e.dic, e.ico, e.name, e.address, e.latitude, e.longitude,
        array_agg(es.description) as items from entity e left join entity_subject es on e.entity_id=es.entity_id
        group by e.entity_id"""
        self._load_items_query = 'select * from entity_subject'

    def load_from_DB(self, parts=10):
        print("Running query: " + self._load_query)
        raw_data = self.runQuery(self._load_query)

        entities = {}
        total_entities = len(raw_data)
        print("Loading total " + str(total_entities) + " entities")
        for i, ent in enumerate(raw_data):
            if i % (int(total_entities / parts) + 1) == 0:
                print("Progress: {}%".format(numpy.ceil(i * 100 / total_entities)))
            entity_id = ent[0]
            dic = ent[1]
            ico = ent[2]
            name = ent[3]
            address = ent[4]
            gps_coords = (ent[5], ent[6])
            items = ent[7]
            entity = entities.get(entity_id,
                                  {'dic': dic, 'ico': ico, 'name': name, 'address': address, 'gps': gps_coords,
                                   'items': []})
            entities[entity_id] = entity
        print("Done")
        return pandas.DataFrame.from_dict(entities, orient='index')

    def truncate_DB(self):
        self.runQuery('truncate table entity_subject')

    def save_to_DB(self, df_entities):
        for index, row in df_entities.iterrows():
            entity_id = index
            address = row['address']
            gps_coords = row['gps']
            latitude, longitude = (gps_coords[0], gps_coords[1]) if gps_coords else (None, None)
            items = row['items']
            to_print = [str(x)[:5] if x is not None else '-   ' \
                        for x in [entity_id, address, latitude, longitude, len(items) if isinstance(items, set) else 0]]
            print("Updating entity {} with address {}, gps:({},{}), #items:{}".format(*to_print))
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
            for i, item in enumerate(items):
                cursor = self._connection.cursor()
                postgres_insert_query = """INSERT INTO entity_subject (entity_id, description)
                                            VALUES (%s,%s)"""
                record_to_insert = (entity_id, item)
                cursor.execute(postgres_insert_query, record_to_insert)

                self._connection.commit()
                count = cursor.rowcount
                cursor.close()