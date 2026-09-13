from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET_HANDLE = "indraputra21/used-car-listings-in-indonesia"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "used_car.csv"


def main() -> None:
    downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_files = list(downloaded.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Tidak ada file CSV di dataset yang diunduh.")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_files[0], DESTINATION)
    print(f"Dataset tersimpan di {DESTINATION}")


if __name__ == "__main__":
    main()

