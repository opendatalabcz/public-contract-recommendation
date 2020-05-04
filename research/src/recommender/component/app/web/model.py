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

    def __init__(self, contract_id, code1, code2, name, subject_items, cpvs, submitter):
        self.id = contract_id
        self.code1 = code1
        self.code2 = code2
        self.name = name
        self.subject_items = subject_items
        self.cpvs = cpvs
        self.submitter = submitter


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
            cpvs = []
            if isinstance(row['cpv_codes'], list):
                cpvs = [CPVCode(cpv_code, cpv_name) for cpv_code, cpv_name in zip(row['cpv_codes'], row['cpv_items'])]
            ico = row['ico'].strip()
            submitter = Submitter(ico, row['entity_name'], row['address'], row['subject_items'], profiles.get(ico, []))
            contracts.append(Contract(contract_id, code1, code2, name, subject_items, cpvs, submitter))
        return contracts
