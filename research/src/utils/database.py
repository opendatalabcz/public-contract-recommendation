import pandas
import numpy


class DBManager:

    def __init__(self, connection):
        self._connection = connection

    def runQuery(self, query):
        cursor = self._connection.cursor()
        cursor.execute(query)
        if 'select' in query.lower():
            data = cursor.fetchall()
        elif 'returning' in query.lower():
            data = cursor.fetchone()[0]
        else:
            data = True
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
