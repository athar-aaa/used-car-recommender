from src.data import load_vehicle_data


def test_dataset_is_clean_and_mileage_is_converted():
    data = load_vehicle_data()
    assert len(data) > 500
    assert data.isna().sum().sum() == 0
    assert data["mileage_km"].median() > 10_000
    assert data["price_rp"].min() > 0


def test_display_name_and_ids_are_unique_enough():
    data = load_vehicle_data()
    assert data["car_id"].is_unique
    assert data["display_name"].nunique() > 100

