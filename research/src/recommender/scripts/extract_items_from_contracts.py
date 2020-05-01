import logging
import sys

import psycopg2
import ufal.udpipe

from recommender.component.database.postgres import DocumentManager, SubjectItemManager
from recommender.component.feature.document import DocumentMerger
from recommender.component.feature.contract_subject import SubjectExtractor
from recommender.component.feature.contract_subject.conllu_preprocessing import TextAnnotator
from recommender.component.feature.contract_subject.subject_extraction import ItemsSplitter
from recommender.component.feature.embedding import FastTextEmbedder

UDPIPE_PATH = "C:/x/diplomka/research/model/udpipe/udpipe-ud-2.5-191206/czech-pdt-ud-2.5-191206.udpipe"
FASTTEXT_PATH = "C:/x/diplomka/research/model/fasttext/wiki.cs/wiki.cs.bin"
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
        from subject_item
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

model = ufal.udpipe.Model.load(UDPIPE_PATH)
udp_pipeline = ufal.udpipe.Pipeline(model, "tokenize", ufal.udpipe.Pipeline.DEFAULT, ufal.udpipe.Pipeline.DEFAULT,
                                    "conllu")
merger = DocumentMerger(logger=root_logger)
annotator = TextAnnotator(pipeline=udp_pipeline, logger=root_logger)
items_spliter = ItemsSplitter()
extractor = SubjectExtractor(text_annotator=annotator, concatenator=items_spliter, logger=root_logger)
embedder = FastTextEmbedder(FASTTEXT_PATH, logger=root_logger)

dmngr = DocumentManager(psycopg2_conn, load_query=LOAD_QUERY, logger=root_logger)
simngr = SubjectItemManager(psycopg2_conn, logger=root_logger)

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
    logging.info('Subject extracting...')
    df_contracts['subject_items'] = df_contracts['doc_text'].apply(lambda text: extractor.process(text))
    logging.info('Item embedding...')
    df_contracts['embeddings'] = df_contracts['subject_items'].apply(lambda items: embedder.process(items))
    logging.info('Items saving...')
    simngr.save(df_contracts)
logging.info("Done.")
