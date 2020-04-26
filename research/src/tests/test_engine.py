import pytest
import numpy

from recommender.component.engine.engine import SearchEngine


def test_query_by_subject(context):
    embedder = context.get_fasttext_embedder()
    df_contracts = context.get_contracts_data()

    se = SearchEngine(df_contracts, embedder, num_results=2)
    result = se.query_by_subject('notebook')

    assert isinstance(result, dict)
    assert len(result) == 1

    query_result = result[1]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == 0
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 0.47347007131792146)


def test_query_by_address(context):
    embedder = context.get_fasttext_embedder()
    df_contracts = context.get_contracts_data()

    se = SearchEngine(df_contracts, embedder, num_results=2)
    result = se.query_by_address('Jičín')

    assert isinstance(result, dict)
    assert len(result) == 1

    query_result = result[1]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == 0
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 0.31935185257094634)


def test_query_by_entity_subject(context):
    embedder = context.get_fasttext_embedder()
    df_contracts = context.get_contracts_data()

    se = SearchEngine(df_contracts, embedder, num_results=2)
    result = se.query_by_entity_subject('obchod s elektronikou')

    assert isinstance(result, dict)
    assert len(result) == 1

    query_result = result[1]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == 0
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 1.0)


@pytest.mark.parametrize(
    ['address', 'ref_result'],
    [('Jilemnice', [(0, 0.33192310781406875), (5, 0.30039716583057874)]),
     ('Vrchlabí', [(5, 0.3377391198454613), (0, 0.29672834012545324)])]
)
def test_query(context, address, ref_result):
    embedder = context.get_fasttext_embedder()
    df_contracts = context.get_contracts_data()

    se = SearchEngine(df_contracts, embedder, num_results=2)
    result = se.query({'subject': 'výpočetní technika', 'locality': address, 'entity_subject': 'obchod s elektronikou'})

    assert isinstance(result, dict)
    assert len(result) == 1

    query_result = result[1]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == ref_result[0][0]
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], ref_result[0][1])

    contract_result2 = query_result[1]
    assert isinstance(contract_result2, dict)
    assert 'contract_id' in contract_result2
    assert contract_result2['contract_id'] == ref_result[1][0]
    assert 'similarity' in contract_result2
    assert numpy.isclose(contract_result2['similarity'], ref_result[1][1])


def test_query_by_user_profile(context):
    embedder = context.get_fasttext_embedder()
    user_profile = context.get_user_profiles_data()
    df_contracts = context.get_contracts_data()

    se = SearchEngine(df_contracts, embedder, num_results=2)
    result = se.query_by_user_profile(user_profile)

    assert isinstance(result, dict)
    assert len(result) == 4

    query_result = result[13]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == 2
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 0.1433232717589146)


def test_query_by_user_profile2(context):
    embedder = context.get_fasttext_embedder()
    user_profile = context.get_user_profiles_data()
    df_contracts = context.get_contracts_data()

    se = SearchEngine(df_contracts, embedder, num_results=2)
    result = se.query_by_user_profile(user_profile, {'subject', 'locality'})

    assert isinstance(result, dict)
    assert len(result) == 4

    query_result = result[13]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == 2
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 0.1662119375065183)
