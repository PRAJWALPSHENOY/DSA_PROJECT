from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder


@dataclass
class HeuristicModel:
    feature_dim: int
    random_state: int = 7

    def __post_init__(self) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=64,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=self.random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            self.model.fit(x, y)
        except ValueError as exc:
            if "validation set is too small" not in str(exc).lower():
                raise

            # Fallback for small datasets where early-stopping split cannot be formed.
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                batch_size=64,
                max_iter=250,
                early_stopping=False,
                random_state=self.random_state,
            )
            self.model.fit(x, y)

    def predict(self, features: np.ndarray) -> float:
        value = float(self.model.predict(features.reshape(1, -1))[0])
        return max(-1.0, min(1.0, value))

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        values = self.model.predict(x)
        return np.clip(values, -1.0, 1.0)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_dim": self.feature_dim,
            "random_state": self.random_state,
            "model": self.model,
        }
        with out.open("wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def load(path: str | Path) -> "HeuristicModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)

        loaded = HeuristicModel(feature_dim=payload["feature_dim"], random_state=payload["random_state"])
        loaded.model = payload["model"]
        return loaded


@dataclass
class MoveOrderingModel:
    feature_dim: int
    random_state: int = 7

    def __post_init__(self) -> None:
        self.encoder = LabelEncoder()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=64,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=self.random_state,
        )
        self._label_to_encoded: dict[str, int] = {}

    def fit(self, x: np.ndarray, labels: np.ndarray) -> None:
        encoded = self.encoder.fit_transform(labels)
        self._label_to_encoded = {label: idx for idx, label in enumerate(self.encoder.classes_)}
        try:
            self.model.fit(x, encoded)
        except ValueError as exc:
            if "validation set is too small" not in str(exc).lower():
                raise

            # Fallback for small datasets where early-stopping split cannot be formed.
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                batch_size=64,
                max_iter=250,
                early_stopping=False,
                random_state=self.random_state,
            )
            self.model.fit(x, encoded)

    def rank_labels(self, features: np.ndarray, legal_labels: list[str]) -> list[str]:
        if len(legal_labels) <= 1:
            return legal_labels

        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]

        scored = []
        for index, label in enumerate(legal_labels):
            encoded = self._label_to_encoded.get(label)
            prob = float(probabilities[encoded]) if encoded is not None and encoded < len(probabilities) else 0.0
            scored.append((label, prob, index))

        scored.sort(key=lambda item: (item[1], -item[2]), reverse=True)
        return [item[0] for item in scored]

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_dim": self.feature_dim,
            "random_state": self.random_state,
            "model": self.model,
            "encoder": self.encoder,
        }
        with out.open("wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def load(path: str | Path) -> "MoveOrderingModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)

        loaded = MoveOrderingModel(feature_dim=payload["feature_dim"], random_state=payload["random_state"])
        loaded.model = payload["model"]
        loaded.encoder = payload["encoder"]
        loaded._label_to_encoded = {label: idx for idx, label in enumerate(loaded.encoder.classes_)}
        return loaded
