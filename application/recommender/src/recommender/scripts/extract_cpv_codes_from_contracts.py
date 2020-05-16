import logging
import sys

import psycopg2

from recommender.component.database.postgres import DocumentDAO, CPVItemDAO
from recommender.component.feature.contract_subject.subject_context_preprocessing import CPVCodeExtractor, \
    SubjectContextPreprocessor
from recommender.component.feature.contract_subject.subject_extraction import ItemsSplitter
from recommender.component.feature.document import DocumentMerger

LOAD_QUERY = """select document_id, contract_id, data
from document
where contract_id in (
  select contract_id
  from
    (select distinct(contract_id)
    from document
    where processed = True
    and contract_id not in (
        select distinct(contract_id)
        from contract_cpv
        )) as contract_ids
  order by random()
  limit 10)
;"""
MAX_BATCHES = 3

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
root_logger = logging.getLogger()

psycopg2_conn = psycopg2.connect(dbname='public_contracts', user='postgres', password='admin', host='localhost',
                                 port='5432')

merger = DocumentMerger(logger=root_logger)
items_spliter = ItemsSplitter()
extractors = [
    CPVCodeExtractor(keep_text=False)
]
attributes_extractor = SubjectContextPreprocessor(transformers = extractors)
dmngr = DocumentDAO(psycopg2_conn, load_query=LOAD_QUERY, logger=root_logger)
cpvidao = CPVItemDAO(connection=psycopg2_conn, logger=root_logger)
cpvidao.loadCPVEnumFromDB()

logging.info("Starting the process.")
cond = True
for i in range(MAX_BATCHES):
    logging.info('Contracts loading...')
    df_contracts = dmngr.load()
    ncontracts = len(df_contracts.index)
    if ncontracts == 0:
        logging.info("No more contracts.")
        break
    logging.info("Processing batch {}/{} with {} contracts.".format(i, MAX_BATCHES, ncontracts))

    logging.info('Texts merging...')
    df_contracts['doc_text'] = df_contracts.apply(lambda row: merger.process((row['doc_ids'], row['doc_texts'])),
                                                  axis=1)
    logging.info('CPV extracting...')
    df_contracts['cpv_code'] = df_contracts['doc_text'].apply(lambda text: attributes_extractor.process(text))
    df_contracts['cpv_codes'] = df_contracts['cpv_code'].apply(lambda text: set(text.split('\n')))
    logging.info('Items saving...')
    cpvidao.save(df_contracts)
logging.info("Done.")
