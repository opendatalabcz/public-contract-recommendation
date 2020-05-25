import logging
import sys

import psycopg2

from recommender.component.database.postgres import EntitySubjectDAO
from recommender.component.feature.embedding import FastTextEmbedder

FASTTEXT_PATH = "/home/opendata/public_contracts/pcrec/model/wiki.cs.bin"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
root_logger = logging.getLogger()

psycopg2_conn = psycopg2.connect(dbname='public_contracts', user='postgres', password='admin', host='localhost',
                                 port='5432')

embedder = FastTextEmbedder(FASTTEXT_PATH, logger=root_logger)

esdao = EntitySubjectDAO(psycopg2_conn, logger=root_logger)


def split_items(origitems):
    items = []
    for item in origitems:
        items.extend(item.split('\n'))
    return items


logging.info("Starting the process.")
logging.info('Entity subjects loading...')
df_entity_subject = esdao.load()
logging.info('Item splitting...')
df_entity_subject['entity_items'] = df_entity_subject['entity_items'].apply(split_items)
logging.info('Item embedding...')
df_entity_subject['entity_embeddings'] = df_entity_subject['entity_items'].apply(lambda items: embedder.process(items))
logging.info('Items saving...')
esdao.save(df_entity_subject)
logging.info("Done.")
