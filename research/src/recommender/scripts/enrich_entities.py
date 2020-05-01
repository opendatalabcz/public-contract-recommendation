import re
import requests
import psycopg2
import logging
import pandas
import numpy
import sys
import time

from xml.etree import cElementTree as ElementTree

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


# taken from http://code.activestate.com/recipes/410469-xml-as-dictionary/
class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


# taken from http://code.activestate.com/recipes/410469-xml-as-dictionary/
class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''

    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


class XMLParser:

    def __init__(self, xml_string):
        transformed_content = re.sub(r'xmlns:([^=]+)="[^"]*"', r'xmlns:\1="\1"', xml_string)
        root = ElementTree.XML(transformed_content)
        xmldict = XmlDictConfig(root)
        self._xml_dict = xmldict


class ARESParser(XMLParser):

    def _check_structure(self, hierarchy=[], node=None):
        if node is None:
            node = self._xml_dict
        for next_node in hierarchy:
            node = node.get(next_node, None)
            if not node:
                return False
        return True

    def getAddress(self):
        if not self._check_structure(['{are}Odpoved', '{D}VBAS', '{D}AA']):
            return None
        adrdict = self._xml_dict['{are}Odpoved']['{D}VBAS']['{D}AA']
        address_fields = ['{D}N', '{D}NCO', '{D}NMC', '{D}NU', '{D}CD', '{D}PSC']
        address = ','.join([adrdict.get(field, '') for field in address_fields])
        return address

    def getSubject(self):
        if not self._check_structure(['{are}Odpoved', '{D}VBAS', '{D}PPI', '{D}PP']):
            return None
        subj_items = []
        items = self._xml_dict['{are}Odpoved']['{D}VBAS']['{D}PPI']['{D}PP']
        if isinstance(items, dict) and self._check_structure(['{D}T'], items):
            items = items['{D}T']
        if isinstance(items, str):
            subj_items.append(items)
        else:
            for item in items:
                if isinstance(item, dict) and self._check_structure(['{D}T'], item):
                    item = item['{D}T']
                if isinstance(item, list):
                    subj_items.extend(item)
                else:
                    subj_items.append(item)
        return set(subj_items)


class MapyParser(XMLParser):

    def __init__(self, xml_string):
        self._xml = xml_string
        self._xpattern = re.compile(r'x="([\d]+\.[\d]*)"')
        self._ypattern = re.compile(r'y="([\d]+\.[\d]*)"')

    def getGPS(self):
        if 'item' in self._xml:
            x = self._xpattern.search(self._xml).group(1)
            y = self._ypattern.search(self._xml).group(1)
            return y, x
        return None


class EntityManager:

    def __init__(self, connection):
        self._connection = connection
        self._load_query = """
        select e.entity_id, e.dic, e.ico, e.name, e.address, e.latitude, e.longitude,
        array_agg(es.description) as items from entity e left join entity_subject es on e.entity_id=es.entity_id
        where e.address is null
        group by e.entity_id"""

    def load_from_DB(self, parts=10):
        logging.info("Running query: " + self._load_query)
        cursor = self._connection.cursor()
        cursor.execute(self._load_query)
        raw_data = cursor.fetchall()
        cursor.close()

        entities = {}
        total_entities = len(raw_data)
        logging.info("Loading total " + str(total_entities) + " entities")
        for i, ent in enumerate(raw_data):
            if i % (int(total_entities / parts) + 1) == 0:
                logging.info("Progress: {}%".format(numpy.ceil(i * 100 / total_entities)))
            entity_id = ent[0]
            dic = ent[1]
            ico = ent[2]
            name = ent[3]
            address = ent[4]
            gps_coords = (ent[5], ent[6])
            items = ent[7]
            entity = entities.get(entity_id,
                                  {'dic': dic, 'ico': ico, 'name': name, 'address': address, 'gps': gps_coords,
                                   'items': items})
            entities[entity_id] = entity
        logging.info("Done")
        return pandas.DataFrame.from_dict(entities, orient='index')

    def save_to_DB(self, df_entities):
        logging.info("Saving total {} entities".format(len(df_entities.index)))
        for index, row in df_entities.iterrows():
            entity_id = index
            address = row['address']
            gps_coords = row['gps']
            latitude, longitude = (gps_coords[0], gps_coords[1]) if gps_coords else (None, None)
            items = row['items']
            to_print = [str(x)[:5] if x is not None else '-   ' \
                        for x in [entity_id, address, latitude, longitude, len(items) if isinstance(items, set) else 0]]
            logging.debug("Updating entity {} with address {}, gps:({},{}), #items:{}".format(*to_print))
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


class EntityEnricher:

    def __init__(self,
                 connection=None,
                 ares_url="http://wwwinfo.mfcr.cz/cgi-bin/ares/darv_bas.cgi?ico={}&active=false",
                 mapy_url="https://api.mapy.cz/geocode?query={}"):
        self._ares_url = ares_url
        self._mapy_url = mapy_url
        self._entity_manager = EntityManager(connection)

    def load_from_db(self):
        return self._entity_manager.load_from_DB()

    def save_to_db(self, df):
        self._entity_manager.save_to_DB(df)

    def enrich_dataframe(self, df):
        for i, (index, row) in enumerate(df.iterrows()):
            ico = str(row['ico'])
            logging.info("{}: Getting ico {}".format(str(i), ico))
            areresponse = requests.get(self._ares_url.format(ico))
            while areresponse.status_code == 429:
                logging.debug('Waiting 1h for ico: {}'.format(ico))
                time.sleep(3600)
                areresponse = requests.get(self._ares_url.format(ico))
            if areresponse.status_code != 200:
                logging.debug('Error {} with ico: {}'.format(areresponse.status_code, ico))
                continue
            parser = ARESParser(areresponse.content.decode('utf-8'))
            address = parser.getAddress()
            items = parser.getSubject()
            df.loc[index, 'items'] = items

            if address is None:
                continue
            logging.debug('    Getting GPS for address {}'.format(address))
            mapyresponse = requests.get(self._mapy_url.format(address))
            while mapyresponse.status_code == 429:
                logging.debug('Waiting 1h for address: {}'.format(address))
                time.sleep(3600)
                mapyresponse = requests.get(self._ares_url.format(ico))
            if mapyresponse.status_code != 200:
                logging.debug('Error {} with address: {}'.format(mapyresponse.status_code, address))
                continue
            mparser = MapyParser(mapyresponse.content.decode('utf-8'))
            gpspoint = mparser.getGPS()
            df.loc[index, 'address'] = address
            df.loc[index, 'gps'] = gpspoint
        return df

    def split_dataframe_to_batches(self, df=None, batch_size=100):
        i = 0
        batches = []
        while i < len(df.index):
            batches.append(df[i:i + batch_size])
            i += batch_size
        return batches

    def batch_process_enrichment(self, df=None, batch_size=100):
        if df is None:
            df = self.load_from_db()
        df_batches = self.split_dataframe_to_batches(df, batch_size)
        logging.info("Processing {} batches of total {}".format(len(df_batches), len(df.index)))
        for i, df_batch in enumerate(df_batches):
            logging.info("Processing batch {}/{}".format(i, len(df_batches)))
            self.enrich_dataframe(df_batch)
            self.save_to_db(df_batch)
        logging.info("Done")


psycopg2_conn = psycopg2.connect(dbname='public_contracts_v2', user='postgres', password='admin', host='localhost',
                                 port='5432')
enricher = EntityEnricher(psycopg2_conn)
enricher.batch_process_enrichment()
