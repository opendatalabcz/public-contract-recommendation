import json
import glob

from .similarity import JaccardSimilarityMachine
from .subject_extraction import ReferenceSubjectContextExtractor


def save_subjects(df_contracts, column, vzdirs='../../test-data/*/*/*'):
    dirs = [path for path in glob.glob(vzdirs) if 'test_src' not in path]
    for path in dirs:
        contr_name = '/'.join(path.split('\\')[1:4])
        rows = df_contracts[df_contracts['contr_name'] == contr_name]
        if len(rows) > 0:
            print('saving ' + contr_name)
            with open(path + '/_ref.json', encoding='utf-8') as f:
                data = f.read()
                ref_dict = json.loads(data)
            ref_dict[column] = rows.iloc[0][column] if len(rows) > 0 else None
            with open(path + '/_ref.json', 'w', encoding='utf-8') as f:
                json.dump(ref_dict, f, ensure_ascii=False, indent=4)


def validate_subjects(df_contracts, column, vzdirs='../../test-data/*/*/*',
                      similarity_machine=JaccardSimilarityMachine()):
    ref_column = column + '_ref'
    df_contracts['valid_score'] = 0
    df_contracts[ref_column] = None
    ref_paths = [path for path in glob.glob(vzdirs) if 'test_src' not in path]
    for path in ref_paths:
        contr_name = '/'.join(path.split('\\')[1:4])
        rows = df_contracts[df_contracts.contr_name == contr_name]
        if len(rows) > 0:
            row = rows.iloc[0]
            value = row[column]
            if isinstance(value, list):
                value = '\n==========\n'.join(value)
            ref_value = None
            with open(path + '/_ref.json', 'r', encoding='utf8') as f:
                ref = json.load(f)
                if column in ref:
                    ref_value = ref[column]
                if isinstance(ref_value, list):
                    ref_value = '\n==========\n'.join(ref_value)
            valid_score = similarity_machine.compute(value, ref_value)
            df_contracts.loc[row.name, 'valid_score'] = valid_score
            df_contracts.loc[row.name, ref_column] = ref_value
    return df_contracts


def save_valid_contexts(df_contracts, df_ref_documents):
    for index, row in df_contracts[df_contracts.valid == True].iterrows():
        path = df_ref_documents[df_ref_documents.contr_name == row.contr_name].iloc[0]['doc_path'].split('\\')[:-1]
        path.append('subj_context.txt')
        completed_path = '/'.join(path)
        with open(completed_path, 'w', encoding='utf8') as f:
            f.write(row.subj_context)


def validate_subj_contexts(df_contracts, vzdirs):
    df_contracts['valid_rat'] = 0
    df_contracts['ref_context'] = None
    subj_context_paths = [path for path in glob.glob(vzdirs) if 'subj_context.txt' in path]
    for path in subj_context_paths:
        contr_name = '/'.join(path.split('\\')[1:4])
        row = df_contracts[df_contracts.contr_name == contr_name].iloc[0]
        with open(path, 'r', encoding='utf8') as f:
            ref_context = f.read()
        valid_rat = JaccardSimilarityMachine().compute(row['subj_context'], ref_context)
        df_contracts.valid_rat.loc[row.name] = valid_rat
        df_contracts.ref_context.loc[row.name] = ref_context
    return df_contracts


def validate_subj_contexts_v2(df_contracts, vzdirs):
    df_contracts['valid_rat'] = 0
    df_contracts['ref_context'] = None
    ref_paths = [path for path in glob.glob(vzdirs) if 'test_src' not in path]
    for path in ref_paths:
        contr_name = '/'.join(path.split('\\')[1:4])
        row = df_contracts[df_contracts.contr_name == contr_name]
        if len(row) > 0:
            with open(path + '/_ref.json', 'r', encoding='utf8') as f:
                ref = f.read()
                ref_context = ReferenceSubjectContextExtractor(path).extract()
            valid_rat = JaccardSimilarityMachine().compute(row.iloc[0]['subj_context'], ref_context)
            df_contracts.valid_rat.loc[row.iloc[0].name] = valid_rat
            df_contracts.ref_context.loc[row.iloc[0].name] = ref_context
    return df_contracts
