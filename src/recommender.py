from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from src.data import FEATURE_LABELS


@dataclass(frozen=True)
class UserPreferences:
    budget_min: int
    budget_max: int
    year_min: int
    mileage_max: int
    brands: tuple[str, ...] = field(default_factory=tuple)
    transmission: str | None = None
    location: str | None = None
    plate_type: str | None = None
    instalment_max: int | None = None
    desired_features: tuple[str, ...] = field(default_factory=tuple)


class UsedCarRecommender:
    """Weighted content-based KNN for mixed numerical and categorical preferences."""

    BASE_WEIGHTS = {
        "price": 0.30,
        "year": 0.15,
        "mileage": 0.15,
        "brand": 0.10,
        "transmission": 0.08,
        "features": 0.15,
        "location": 0.05,
        "plate_type": 0.02,
        "instalment": 0.10,
    }

    def __init__(self) -> None:
        self.data: pd.DataFrame | None = None
        self.encoders: dict[str, OneHotEncoder] = {}

    def fit(self, data: pd.DataFrame) -> "UsedCarRecommender":
        self.data = data.reset_index(drop=True).copy()
        for column in ["brand", "transmission", "location", "plate_type"]:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(self.data[[column]])
            self.encoders[column] = encoder
        return self

    def _require_fit(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("Model harus di-fit sebelum digunakan.")
        return self.data

    @staticmethod
    def _filter(data: pd.DataFrame, prefs: UserPreferences, relaxed: bool) -> pd.DataFrame:
        price_low = prefs.budget_min * (0.85 if relaxed else 1.0)
        price_high = prefs.budget_max * (1.15 if relaxed else 1.0)
        year_min = prefs.year_min - (2 if relaxed else 0)
        mileage_max = prefs.mileage_max * (1.25 if relaxed else 1.0)

        mask = (
            data["price_rp"].between(price_low, price_high)
            & data["year"].ge(year_min)
            & data["mileage_km"].le(mileage_max)
        )
        if prefs.instalment_max:
            instalment_limit = prefs.instalment_max * (1.15 if relaxed else 1.0)
            mask &= data["instalment_rp"].le(instalment_limit)
        return data.loc[mask].copy()

    def _categorical_distance(
        self, candidates: pd.DataFrame, column: str, choices: tuple[str, ...]
    ) -> np.ndarray:
        encoded_candidates = self.encoders[column].transform(candidates[[column]])
        distances = []
        for choice in choices:
            encoded_choice = self.encoders[column].transform(pd.DataFrame({column: [choice]}))[0]
            distances.append(
                np.linalg.norm(encoded_candidates - encoded_choice, axis=1) / np.sqrt(2)
            )
        return np.min(np.column_stack(distances), axis=1).reshape(-1, 1)

    def _distance_blocks(
        self, candidates: pd.DataFrame, prefs: UserPreferences
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        data = self._require_fit()
        midpoint = (prefs.budget_min + prefs.budget_max) / 2
        half_range = max((prefs.budget_max - prefs.budget_min) / 2, 5_000_000)

        blocks: dict[str, np.ndarray] = {
            "price": np.clip(
                np.abs(candidates["price_rp"].to_numpy() - midpoint) / half_range, 0, 1
            ).reshape(-1, 1),
            "year": np.clip(
                (data["year"].max() - candidates["year"].to_numpy())
                / max(data["year"].max() - prefs.year_min, 1),
                0,
                1,
            ).reshape(-1, 1),
            "mileage": np.clip(
                candidates["mileage_km"].to_numpy() / max(prefs.mileage_max, 1), 0, 1
            ).reshape(-1, 1),
        }

        if prefs.brands:
            blocks["brand"] = self._categorical_distance(
                candidates, "brand", prefs.brands
            )
        if prefs.transmission:
            blocks["transmission"] = self._categorical_distance(
                candidates, "transmission", (prefs.transmission,)
            )
        if prefs.location:
            blocks["location"] = self._categorical_distance(
                candidates, "location", (prefs.location,)
            )
        if prefs.plate_type:
            blocks["plate_type"] = self._categorical_distance(
                candidates, "plate_type", (prefs.plate_type,)
            )
        if prefs.instalment_max:
            blocks["instalment"] = np.clip(
                candidates["instalment_rp"].to_numpy() / prefs.instalment_max, 0, 1
            ).reshape(-1, 1)
        if prefs.desired_features:
            missing = 1 - candidates[list(prefs.desired_features)].to_numpy(dtype=float)
            blocks["features"] = missing / np.sqrt(len(prefs.desired_features))

        active_weight = sum(self.BASE_WEIGHTS[name] for name in blocks)
        matrix = np.concatenate(
            [
                block * np.sqrt(self.BASE_WEIGHTS[name] / active_weight)
                for name, block in blocks.items()
            ],
            axis=1,
        )
        return matrix, blocks

    def recommend(
        self, prefs: UserPreferences, n_results: int = 5
    ) -> tuple[pd.DataFrame, bool]:
        data = self._require_fit()
        candidates = self._filter(data, prefs, relaxed=False)
        relaxed = len(candidates) < n_results
        if relaxed:
            candidates = self._filter(data, prefs, relaxed=True)
        if candidates.empty:
            return candidates, relaxed

        matrix, _ = self._distance_blocks(candidates, prefs)
        neighbors_to_fetch = min(len(candidates), max(n_results * 5, n_results))
        model = NearestNeighbors(n_neighbors=neighbors_to_fetch, metric="euclidean")
        model.fit(matrix)
        distances, positions = model.kneighbors(
            np.zeros((1, matrix.shape[1])), n_neighbors=neighbors_to_fetch
        )

        ranked = candidates.iloc[positions[0]].copy()
        ranked["match_score"] = np.clip((1 - distances[0]) * 100, 0, 100).round(1)
        ranked["relaxed"] = relaxed

        # Avoid a result page filled with repeated listings of the same model.
        ranked = ranked.drop_duplicates(subset=["display_name"], keep="first")
        if len(ranked) < n_results and neighbors_to_fetch < len(candidates):
            model.set_params(n_neighbors=len(candidates))
            distances, positions = model.kneighbors(
                np.zeros((1, matrix.shape[1])), n_neighbors=len(candidates)
            )
            ranked = candidates.iloc[positions[0]].copy()
            ranked["match_score"] = np.clip((1 - distances[0]) * 100, 0, 100).round(1)
            ranked["relaxed"] = relaxed
            ranked = ranked.drop_duplicates(subset=["display_name"], keep="first")

        return ranked.head(n_results).reset_index(drop=True), relaxed

    def explanation(self, row: pd.Series, prefs: UserPreferences) -> list[str]:
        reasons: list[str] = []
        midpoint = (prefs.budget_min + prefs.budget_max) / 2
        if not prefs.budget_min <= row["price_rp"] <= prefs.budget_max:
            reasons.append("Alternatif terdekat sedikit di luar anggaran awal")
        elif abs(row["price_rp"] - midpoint) <= max((prefs.budget_max - prefs.budget_min) * 0.25, 1):
            reasons.append("Harga dekat dengan titik tengah anggaran")
        else:
            reasons.append("Harga berada dalam rentang anggaran")
        if row["year"] >= prefs.year_min:
            reasons.append("Tahun kendaraan memenuhi batas minimum")
        else:
            reasons.append("Alternatif terdekat di bawah batas tahun awal")
        if row["mileage_km"] <= prefs.mileage_max * 0.6:
            reasons.append("Kilometer relatif rendah terhadap batas Anda")
        elif row["mileage_km"] > prefs.mileage_max:
            reasons.append("Kilometer sedikit melebihi batas awal")
        if prefs.brands and row["brand"] in prefs.brands:
            reasons.append("Merek termasuk dalam pilihan Anda")
        if prefs.transmission and row["transmission"] == prefs.transmission:
            reasons.append("Transmisi sesuai preferensi")
        if prefs.location and row["location"] == prefs.location:
            reasons.append("Lokasi sesuai preferensi")
        if prefs.desired_features:
            matched = sum(int(row[feature]) for feature in prefs.desired_features)
            reasons.append(
                f"Memenuhi {matched} dari {len(prefs.desired_features)} fitur pilihan"
            )
        return reasons[:4]

    def evaluate(self, sample_size: int = 60, k: int = 5) -> dict[str, float]:
        """Evaluate retrieval consistency using held-out listing preferences."""
        data = self._require_fit()
        sample = data.sample(min(sample_size, len(data)), random_state=42)
        hits = 0
        brand_precision: list[float] = []
        diversity: list[float] = []
        covered: set[int] = set()

        for _, target in sample.iterrows():
            available_features = tuple(
                feature for feature in FEATURE_LABELS if int(target[feature]) == 1
            )[:3]
            preferences = UserPreferences(
                budget_min=max(int(target["price_rp"] * 0.8), int(data["price_rp"].min())),
                budget_max=int(target["price_rp"] * 1.2),
                year_min=max(int(target["year"]) - 2, int(data["year"].min())),
                mileage_max=max(int(target["mileage_km"] * 1.25), 10_000),
                brands=(str(target["brand"]),),
                transmission=str(target["transmission"]),
                desired_features=available_features,
            )
            results, _ = self.recommend(preferences, n_results=k)
            if int(target["car_id"]) in results["car_id"].tolist():
                hits += 1
            if not results.empty:
                brand_precision.append((results["brand"] == target["brand"]).mean())
                diversity.append(results["brand"].nunique() / len(results))
                covered.update(results["car_id"].astype(int).tolist())

        count = max(len(sample), 1)
        return {
            "hit_rate_at_k": hits / count,
            "brand_precision_at_k": float(np.mean(brand_precision)) if brand_precision else 0,
            "diversity_at_k": float(np.mean(diversity)) if diversity else 0,
            "catalog_coverage": len(covered) / len(data),
        }
