import numpy

from recommender.component.similarity.common import ComplexSimilarityComputer
from recommender.component.similarity.geodesic import LocalityDistanceComputer, SimilarLocalityComputer, \
    AggregatedLocalSimilarityComputer
from recommender.component.similarity.vector_space import ItemDistanceComputer, SimilarItemsComputer, \
    AggregatedItemSimilarityComputer


def test_item_distance_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    idc = ItemDistanceComputer(df_contracts)
    result = idc.compute_nearest(df_user_profiles)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    user_result = result[13]
    assert isinstance(user_result, dict)
    assert len(user_result) == 1
    assert 'automobily' in user_result

    items_result = user_result['automobily']
    assert isinstance(items_result, list)
    assert len(items_result) == 1

    item_result = items_result[0]
    assert isinstance(item_result, dict)
    assert len(item_result) == 3
    assert 'contract_id' in item_result
    assert 2 == item_result['contract_id']
    assert 'item' in item_result
    assert 'dopravník' == item_result['item']
    assert 'distance' in item_result
    assert numpy.isclose(0.5445505853395554, item_result['distance'])


def test_item_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    sic = SimilarItemsComputer(df_contracts)
    result = sic.compute_most_similar(df_user_profiles)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    user_result = result[13]
    assert isinstance(user_result, dict)
    assert len(user_result) == 1
    assert 'automobily' in user_result

    items_result = user_result['automobily']
    assert isinstance(items_result, list)
    assert len(items_result) == 1

    item_result = items_result[0]
    assert isinstance(item_result, dict)
    assert len(item_result) == 4
    assert 'contract_id' in item_result
    assert 2 == item_result['contract_id']
    assert 'item' in item_result
    assert 'dopravník' == item_result['item']
    assert 'distance' in item_result
    assert numpy.isclose(0.5445505853395554, item_result['distance'])
    assert 'similarity' in item_result
    assert numpy.isclose(0.45544941466044464, item_result['similarity'])


def test_aggregated_item_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    aisc = AggregatedItemSimilarityComputer(df_contracts)
    result = aisc.compute_most_similar(df_user_profiles, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    user_result = result[13]
    assert isinstance(user_result, list)
    assert len(user_result) == 2

    item_result = user_result[0]
    assert isinstance(item_result, dict)
    assert len(item_result) == 2
    assert 'contract_id' in item_result
    assert 2 == item_result['contract_id']
    assert 'similarity' in item_result
    assert numpy.isclose(0.3969723406723996, item_result['similarity'])


def test_aggregated_item_similarity_computer2(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    aisc = AggregatedItemSimilarityComputer(df_contracts, distance_computer=ItemDistanceComputer(df_contracts, cols=(
        'entity_embeddings', 'entity_items')))
    result = aisc.compute_most_similar(df_user_profiles, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    user_result = result[13]
    assert isinstance(user_result, list)
    assert len(user_result) == 2

    item_result = user_result[0]
    assert isinstance(item_result, dict)
    assert len(item_result) == 2
    assert 'contract_id' in item_result
    assert 0 == item_result['contract_id']
    assert 'similarity' in item_result
    assert numpy.isclose(0.4034540073587843, item_result['similarity'])


def test_locality_distance_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    ldc = LocalityDistanceComputer(df_contracts)
    result = ldc.compute_nearest(df_user_profiles)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    user_result = result[13]
    assert isinstance(user_result, dict)
    assert len(user_result) == 1
    assert 'K Vejrychovsku 1074, Jilemnice' in user_result

    addresses_result = user_result['K Vejrychovsku 1074, Jilemnice']
    assert isinstance(addresses_result, list)
    assert len(addresses_result) == 1

    address_result = addresses_result[0]
    assert isinstance(address_result, dict)
    assert len(address_result) == 3
    assert 'contract_id' in address_result
    assert 0 == address_result['contract_id']
    assert 'address' in address_result
    assert 'V Jilmu 229 514 01 Jilemnice' == address_result['address']
    assert 'distance' in address_result
    assert numpy.isclose(0.7822094, address_result['distance'])


def test_locality_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    slc = SimilarLocalityComputer(df_contracts)
    result = slc.compute_most_similar(df_user_profiles)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    user_result = result[13]
    assert isinstance(user_result, dict)
    assert len(user_result) == 1
    assert 'K Vejrychovsku 1074, Jilemnice' in user_result

    addresses_result = user_result['K Vejrychovsku 1074, Jilemnice']
    assert isinstance(addresses_result, list)
    assert len(addresses_result) == 1

    address_result = addresses_result[0]
    assert isinstance(address_result, dict)
    assert len(address_result) == 4
    assert 'contract_id' in address_result
    assert 0 == address_result['contract_id']
    assert 'address' in address_result
    assert 'V Jilmu 229 514 01 Jilemnice' == address_result['address']
    assert 'distance' in address_result
    assert numpy.isclose(0.7822094, address_result['distance'])
    assert 'similarity' in address_result
    assert numpy.isclose(1, address_result['similarity'])


def test_aggregated_locality_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    alsc = AggregatedLocalSimilarityComputer(df_contracts)
    result = alsc.compute_most_similar(df_user_profiles, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 16 in result

    user_result = result[16]
    assert isinstance(user_result, list)
    assert len(user_result) == 2

    address_result = user_result[0]
    assert isinstance(address_result, dict)
    assert len(address_result) == 2
    assert 'contract_id' in address_result
    assert 1 == address_result['contract_id']
    assert 'similarity' in address_result
    assert numpy.isclose(0.9405868688654786, address_result['similarity'])


def test_complex_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    csc = ComplexSimilarityComputer(df_contracts)
    result = csc.compute_most_similar(df_user_profiles, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 16 in result

    user_result = result[16]
    assert isinstance(user_result, list)
    assert len(user_result) == 2

    contract_result = user_result[0]
    assert isinstance(contract_result, dict)
    assert len(contract_result) == 2
    assert 'contract_id' in contract_result
    assert 1 == contract_result['contract_id']
    assert 'similarity' in contract_result
    assert numpy.isclose(0.06861591318796488, contract_result['similarity'])


def test_complex_similarity_computer2(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()

    similarity_computers = [
        AggregatedItemSimilarityComputer(df_contracts),
        AggregatedItemSimilarityComputer(df_contracts, distance_computer=ItemDistanceComputer(df_contracts, cols=(
            'entity_embeddings', 'entity_items'))),
        AggregatedLocalSimilarityComputer(df_contracts),
    ]

    csc = ComplexSimilarityComputer(df_contracts, similarity_computers=similarity_computers)
    result = csc.compute_most_similar(df_user_profiles, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 16 in result

    user_result = result[16]
    assert isinstance(user_result, list)
    assert len(user_result) == 2

    contract_result = user_result[0]
    assert isinstance(contract_result, dict)
    assert len(contract_result) == 2
    assert 'contract_id' in contract_result
    assert 1 == contract_result['contract_id']
    assert 'similarity' in contract_result
    assert numpy.isclose(0.014519224953165382, contract_result['similarity'])