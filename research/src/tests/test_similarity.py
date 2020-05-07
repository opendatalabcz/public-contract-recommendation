import numpy
import pandas

from recommender.component.similarity.common import ComplexSimilarityComputer
from recommender.component.similarity.geospatial import LocalityDistanceComputer, SimilarLocalityComputer, \
    AggregatedLocalSimilarityComputer
from recommender.component.similarity.standardization import WeightedStandardizer
from recommender.component.similarity.vector_space import ItemDistanceComputer, SimilarItemsComputer, \
    AggregatedItemSimilarityComputer


def test_item_distance_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})

    idc = ItemDistanceComputer(df_contracts)
    result = idc.compute_nearest(df_query)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    query_result = result[13]
    assert isinstance(query_result, dict)
    assert len(query_result) == 1
    assert 'automobily' in query_result

    items_result = query_result['automobily']
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
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})

    sic = SimilarItemsComputer(df_contracts)
    result = sic.compute_most_similar(df_query)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    query_result = result[13]
    assert isinstance(query_result, dict)
    assert len(query_result) == 1
    assert 'automobily' in query_result

    items_result = query_result['automobily']
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
    assert numpy.isclose(0.7277247073302223, item_result['similarity'])


def test_aggregated_item_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})

    aisc = AggregatedItemSimilarityComputer(df_contracts)
    result = aisc.compute_most_similar(df_query, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    query_result = result[13]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    item_result = query_result[0]
    assert isinstance(item_result, dict)
    assert len(item_result) == 2
    assert 'contract_id' in item_result
    assert 2 == item_result['contract_id']
    assert 'similarity' in item_result
    assert numpy.isclose(0.693600969484643, item_result['similarity'])


def test_aggregated_item_similarity_computer2(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})

    aisc = AggregatedItemSimilarityComputer(df_contracts, distance_computer=ItemDistanceComputer(df_contracts, cols=(
        'entity_embeddings', 'entity_items')))
    result = aisc.compute_most_similar(df_query, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    query_result = result[13]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    item_result = query_result[0]
    assert isinstance(item_result, dict)
    assert len(item_result) == 2
    assert 'contract_id' in item_result
    assert 0 == item_result['contract_id']
    assert 'similarity' in item_result
    assert numpy.isclose(0.7017270036793921, item_result['similarity'])


def test_locality_distance_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id'})

    ldc = LocalityDistanceComputer(df_contracts)
    result = ldc.compute_nearest(df_query)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    query_result = result[13]
    assert isinstance(query_result, dict)
    assert len(query_result) == 1
    assert 'K Vejrychovsku 1074, Jilemnice' in query_result

    addresses_result = query_result['K Vejrychovsku 1074, Jilemnice']
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
    assert numpy.isclose(0.7850719900135416, address_result['distance'])


def test_locality_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id'})

    slc = SimilarLocalityComputer(df_contracts)
    result = slc.compute_most_similar(df_query)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 13 in result

    query_result = result[13]
    assert isinstance(query_result, dict)
    assert len(query_result) == 1
    assert 'K Vejrychovsku 1074, Jilemnice' in query_result

    addresses_result = query_result['K Vejrychovsku 1074, Jilemnice']
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
    assert numpy.isclose(0.7850719900135416, address_result['distance'])
    assert 'similarity' in address_result
    assert numpy.isclose(1, address_result['similarity'])


def test_aggregated_locality_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id'})

    alsc = AggregatedLocalSimilarityComputer(df_contracts)
    result = alsc.compute_most_similar(df_query, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 16 in result

    query_result = result[16]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    address_result = query_result[0]
    assert isinstance(address_result, dict)
    assert len(address_result) == 2
    assert 'contract_id' in address_result
    assert 1 == address_result['contract_id']
    assert 'similarity' in address_result
    assert numpy.isclose(0.9375343650766038, address_result['similarity'])


def test_complex_similarity_computer(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})

    csc = ComplexSimilarityComputer(df_contracts)
    result = csc.compute_most_similar(df_query, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 16 in result

    query_result = result[16]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert len(contract_result) == 2
    assert 'contract_id' in contract_result
    assert 1 == contract_result['contract_id']
    assert 'similarity' in contract_result
    assert numpy.isclose(0.6015673741365624, contract_result['similarity'])


def test_complex_similarity_computer2(context):
    df_contracts = context.get_contracts_data()
    df_user_profiles = context.get_user_profiles_data()
    df_query = df_user_profiles.rename(columns={'user_id': 'query_id', 'interest_items': 'items'})

    similarity_computers = [
        (AggregatedItemSimilarityComputer(df_contracts), WeightedStandardizer(1)),
        (AggregatedItemSimilarityComputer(df_contracts, distance_computer=ItemDistanceComputer(df_contracts, cols=(
            'entity_embeddings', 'entity_items'))), WeightedStandardizer(0.2)),
        (AggregatedLocalSimilarityComputer(df_contracts), WeightedStandardizer(0.2))
    ]

    csc = ComplexSimilarityComputer(df_contracts, similarity_computers=similarity_computers)
    result = csc.compute_most_similar(df_query, 2)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 16 in result

    query_result = result[16]
    assert isinstance(query_result, list)
    assert len(query_result) == 2

    contract_result = query_result[0]
    assert isinstance(contract_result, dict)
    assert len(contract_result) == 2
    assert 'contract_id' in contract_result
    assert contract_result['contract_id'] == 1
    assert 'similarity' in contract_result
    assert numpy.isclose(0.6261697794780968, contract_result['similarity'])


def test_item_distance_computer_efficieny(context):
    numpy.random.seed(42)
    ncontracts = 1000
    nqueries = 100
    df_contracts = create_random_dataframe(ncontracts).rename(columns={'id': 'contract_id'})
    df_query = create_random_dataframe(nqueries).rename(columns={'id': 'query_id'})

    idc = ItemDistanceComputer(df_contracts)
    result = idc.compute_nearest(df_query, num_results=ncontracts)
    assert numpy.isclose(0.19712971175935512, result[0][''][0]['distance'])


def create_random_dataframe(nentities):
    entities = []
    for i in range(nentities):
        nitems = numpy.random.randint(0, 20)
        embeddings = []
        for j in range(nitems):
            embedding = numpy.random.rand(300)
            embeddings.append(embedding)
        entities.append({'id':i, 'items': ['' for _ in range(nitems)], 'embeddings': embeddings})
    return pandas.DataFrame(entities)