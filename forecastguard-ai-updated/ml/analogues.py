"""
analogues.py

For a given current atmospheric feature vector, finds the K most similar
historical situations (from the TRAIN+VAL period only, never test/current)
and reports how many of them went on to bust. This is the "historical
analogues" evidence shown alongside the ML probability - the two together
are much more convincing to a meteorologist than the ML number alone.

Run standalone to build the analogue index:
    python ml/analogues.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

MODEL_DIR = "models"
DATA_PATH = "data/processed/training_table.csv"


def build_and_save_index(k_max=50):
    from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, get_split_dates

    df = pd.read_csv(DATA_PATH)
    train_dates, val_dates, test_dates = get_split_dates(df)
    date_series = pd.to_datetime(df["init_date"]).dt.date
    pool = df[date_series.isin(train_dates + val_dates)].reset_index(drop=True)  # train+val period only, never test

    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
    X_pool = preprocessor.transform(pool[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_pool = np.asarray(X_pool.todense()) if hasattr(X_pool, "todense") else np.asarray(X_pool)

    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto")
    nn.fit(X_pool)

    joblib.dump(nn, f"{MODEL_DIR}/analogue_index.pkl")
    pool[["init_date", "lead_day", "lat", "lon", "region", TARGET]].to_csv(
        f"{MODEL_DIR}/analogue_pool_meta.csv", index=False
    )
    print(f"Built analogue index over {len(pool):,} historical (train+val prototype dates) samples.")


class AnalogueFinder:
    def __init__(self, model_dir=MODEL_DIR):
        self.nn = joblib.load(f"{model_dir}/analogue_index.pkl")
        self.meta = pd.read_csv(f"{model_dir}/analogue_pool_meta.csv")

    def find(self, X_row_encoded, k=20):
        X_row_encoded = np.asarray(X_row_encoded.todense()) if hasattr(X_row_encoded, "todense") else np.asarray(X_row_encoded)
        distances, indices = self.nn.kneighbors(X_row_encoded.reshape(1, -1), n_neighbors=k)
        matched = self.meta.iloc[indices[0]]
        bust_count = int(matched["bust"].sum())
        total = len(matched)
        cases = matched.head(5)[["init_date", "lead_day", "region", "bust"]].to_dict(orient="records")
        return {
            "k": total,
            "bust_count": bust_count,
            "bust_rate": bust_count / total if total else 0.0,
            "example_cases": cases,
        }


if __name__ == "__main__":
    build_and_save_index()
