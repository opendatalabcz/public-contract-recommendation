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
    assert numpy.isclose(contract_result['similarity'], 0.7275054000652961)


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
    assert numpy.isclose(contract_result['similarity'], 0.3189298806145963)


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
    [('Jilemnice', [(0, 0.8428498116752551), (5, 0.7665913483183965)]),
     ('Vrchlabí', [(5, 0.8459703073776009), (0, 0.7667012658151682)])]
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
    assert contract_result['contract_id'] == 0
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 0.7086131960032703)


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
    assert contract_result['contract_id'] == 0
    assert 'similarity' in contract_result
    assert numpy.isclose(contract_result['similarity'], 0.7097608944873063)
