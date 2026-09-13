from src.data import FEATURE_LABELS, load_vehicle_data
from src.recommender import UsedCarRecommender, UserPreferences


def make_preferences(**overrides):
    values = {
        "budget_min": 100_000_000,
        "budget_max": 300_000_000,
        "year_min": 2016,
        "mileage_max": 120_000,
        "brands": ("Toyota", "Honda"),
        "transmission": "Automatic",
        "desired_features": ("rear_camera",),
    }
    values.update(overrides)
    return UserPreferences(**values)


def test_recommendations_are_ranked_and_diverse():
    data = load_vehicle_data()
    model = UsedCarRecommender().fit(data)
    results, relaxed = model.recommend(make_preferences(), n_results=8)

    assert not relaxed
    assert len(results) == 8
    assert results["display_name"].is_unique
    assert results["match_score"].is_monotonic_decreasing
    assert results["price_rp"].between(100_000_000, 300_000_000).all()
    assert results["year"].ge(2016).all()
    assert results["mileage_km"].le(120_000).all()


def test_tight_preferences_use_relaxed_fallback():
    data = load_vehicle_data()
    model = UsedCarRecommender().fit(data)
    prefs = make_preferences(
        budget_min=500_000_000,
        budget_max=510_000_000,
        year_min=2023,
        mileage_max=10_000,
        desired_features=tuple(FEATURE_LABELS),
    )
    _, relaxed = model.recommend(prefs, n_results=5)
    assert relaxed

