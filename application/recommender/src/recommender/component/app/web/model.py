import pandas
from flask_login import UserMixin


class CPVCode:

    def __init__(self, code, name):
        self.code = code
        self.name = name


class Profile:

    def __init__(self, ico, name, url):
        self.ico = ico
        self.name = name
        self.url = url


class Submitter:

    def __init__(self, ico, name, address, subject_items, profiles):
        self.ico = ico
        self.name = name
        self.address = address
        self.subject_items = subject_items if isinstance(subject_items, list) else []
        self.profiles = profiles


class Contract:

    def __init__(self, contract_id, code1, code2, name, subject_items, embeddings, cpvs, submitter, similarity):
        self.id = contract_id
        self.code1 = code1
        self.code2 = code2
        self.name = name
        self.subject_items = subject_items
        self.embeddings = embeddings
        self.cpvs = cpvs
        self.submitter = submitter
        self.similarity = similarity


class ContractFactory:

    @staticmethod
    def create_contracts(df_contracts, df_profiles=None):
        contracts = []

        profiles = {}
        for index, row in df_profiles.iterrows():
            ico = row['ico'].strip()
            profile = [Profile(ico, name, url) for name, url in zip(row['names'], row['urls'])]
            profiles[ico] = profile

        for index, row in df_contracts.iterrows():
            contract_id = row['contract_id']
            code1 = row['code1']
            code2 = row['code2']
            name = row['name']
            subject_items = row['subject_items'] if isinstance(row['subject_items'], list) else []
            embeddings = row['embeddings'] if isinstance(row['embeddings'], list) else []
            cpvs = []
            if isinstance(row['cpv_codes'], list):
                cpvs = [CPVCode(cpv_code, cpv_name) for cpv_code, cpv_name in zip(row['cpv_codes'], row['cpv_items'])]
            ico = row['ico'].strip()
            submitter = Submitter(ico, row['entity_name'], row['address'], row['subject_items'], profiles.get(ico, []))
            similarity = row['similarity']
            contracts.append(
                Contract(contract_id, code1, code2, name, subject_items, embeddings, cpvs, submitter, similarity))
        return contracts


class Locality:

    def __init__(self, address, gps):
        self.address = address
        self.gps = gps


class InterestItem:

    def __init__(self, description, embedding):
        self.description = description
        self.embedding = embedding


class UserProfile:

    def __init__(self, user_id, locality, interest_items):
        self.id = user_id
        self.locality = locality
        self.interest_items = interest_items

        self.cache = []

    def to_pandas(self):
        updict = [{'user_id': self.id, 'address': self.locality.address, 'gps': self.locality.gps,
                   'interest_items': [item.description for item in self.interest_items + self.cache],
                   'embeddings': [item.embedding for item in self.interest_items + self.cache]}]
        return pandas.DataFrame(updict)


class UserProfileFactory:

    @staticmethod
    def create_profiles(df_user_profiles):
        profiles = []
        for index, row in df_user_profiles.iterrows():
            user_id = row['user_id']
            locality = Locality(row['address'], (row['gps']))
            interest_items = [InterestItem(item, embedding) for item, embedding in
                              zip(row['interest_items'], row['embeddings'])]
            profiles.append(UserProfile(user_id, locality, interest_items))
        return profiles


class User(UserMixin):

    def __init__(self, user_profile):
        self.user_profile = user_profile

    def get_id(self):
        return str(self.user_profile.id)
