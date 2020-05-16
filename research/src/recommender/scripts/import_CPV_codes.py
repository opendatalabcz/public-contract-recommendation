import logging
import sys

import numpy
import pandas
import psycopg2
from recommender.component.feature.embedding import FastTextEmbedder

CPV_CODES_PATH = "C:/x/diplomka/research/classifier/CPV_codes.txt"
FASTTEXT_PATH = "C:/x/diplomka/research/model/fasttext/wiki.cs/wiki.cs.bin"
INSERT_QUERY = """INSERT INTO cpv_code (id, code, name, parent_id, embedding)
VALUES (%s, %s, %s, %s, %s);"""

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
root_logger = logging.getLogger()

psycopg2_conn = psycopg2.connect(dbname='public_contracts_v2', user='postgres', password='Profinit1', host='193.85.191.172',
                                 port='5432')

embedder = FastTextEmbedder(FASTTEXT_PATH, logger=root_logger)

logging.info("Starting the process.")
logging.info("Reading CPV codes...")
df_cpv_codes = pandas.read_csv(CPV_CODES_PATH, delimiter='\t', dtype={'id': 'Int64', 'parent_id': 'Int64'})
total_codes = len(df_cpv_codes.index)
logging.info("Total codes: {}".format(total_codes))
logging.info("Codes embedding...")
df_cpv_codes['embedding'] = df_cpv_codes['name'].apply(lambda name: embedder.process(name))
logging.info("Saving codes...")
for i, (index, row) in enumerate(df_cpv_codes.iterrows()):
    if i % (int(total_codes / 10) + 1) == 0:
        logging.info("Progress: {}%".format(numpy.ceil(i * 100 / total_codes)))
    cpv_id = row['id']
    code = row['code'] if isinstance(row['code'], str) else None
    name = row['name']
    parent_id = row['parent_id'] if isinstance(row['parent_id'], int) else None
    lembedding = row['embedding'].tolist()
    cursor = psycopg2_conn.cursor()

    record_to_insert = (cpv_id, code, name, parent_id, lembedding)
    cursor.execute(INSERT_QUERY, record_to_insert)

    psycopg2_conn.commit()
    count = cursor.rowcount
    cursor.close()

logging.info("Done.")
