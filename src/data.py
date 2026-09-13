from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "used_car.csv"

COLUMN_NAMES = {
    "car name": "car_name",
    "mileage (km)": "mileage_km",
    "plate type": "plate_type",
    "rear camera": "rear_camera",
    "sun roof": "sun_roof",
    "auto retract mirror": "auto_retract_mirror",
    "electric parking brake": "electric_parking_brake",
    "map navigator": "map_navigator",
    "vehicle stability control": "vehicle_stability_control",
    "keyless push start": "keyless_push_start",
    "sports mode": "sports_mode",
    "360 camera view": "camera_360",
    "power sliding door": "power_sliding_door",
    "auto cruise control": "auto_cruise_control",
    "price (Rp)": "price_rp",
    "instalment (Rp|Monthly)": "instalment_rp",
}

FEATURE_LABELS = {
    "rear_camera": "Kamera belakang",
    "sun_roof": "Sunroof",
    "auto_retract_mirror": "Auto-retract mirror",
    "electric_parking_brake": "Rem parkir elektrik",
    "map_navigator": "Navigator",
    "vehicle_stability_control": "Kontrol stabilitas",
    "keyless_push_start": "Keyless push start",
    "sports_mode": "Mode sport",
    "camera_360": "Kamera 360 derajat",
    "power_sliding_door": "Pintu geser elektrik",
    "auto_cruise_control": "Cruise control",
}


@lru_cache(maxsize=4)
def load_vehicle_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and normalize the Kaggle Carsome Indonesia dataset."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {source}. Jalankan scripts/download_data.py."
        )

    data = pd.read_csv(source).rename(columns=COLUMN_NAMES)
    text_columns = [
        "car_name",
        "brand",
        "location",
        "transmission",
        "plate_type",
    ]
    for column in text_columns:
        data[column] = data[column].astype(str).str.strip()

    # The source uses Indonesian thousands formatting: 10.508 represents 10,508 km.
    data["mileage_km"] = (pd.to_numeric(data["mileage_km"]) * 1_000).round().astype(int)
    for column in ["year", "price_rp", "instalment_rp", *FEATURE_LABELS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna().drop_duplicates().copy()
    data = data[
        data["year"].between(1990, 2030)
        & data["price_rp"].gt(0)
        & data["mileage_km"].ge(0)
        & data["instalment_rp"].ge(0)
    ].reset_index(drop=True)

    for column in ["year", "price_rp", "instalment_rp", *FEATURE_LABELS]:
        data[column] = data[column].astype(int)

    data.insert(0, "car_id", range(1, len(data) + 1))
    data["display_name"] = data["brand"] + " " + data["car_name"]
    return data


def dataset_summary(data: pd.DataFrame) -> dict[str, int]:
    return {
        "vehicles": len(data),
        "brands": data["brand"].nunique(),
        "locations": data.loc[data["location"].ne("Unknown"), "location"].nunique(),
        "models": data["display_name"].nunique(),
    }

