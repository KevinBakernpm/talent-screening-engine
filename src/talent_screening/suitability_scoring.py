"""Candidate suitability scoring from structured application data.

Decision: a 0-100 suitability score and a band (``strong`` / ``possible`` /
``unlikely``) for each applicant, used to prioritise which shortlisted
applicants are invited to interview and which receive an automated decline.

Subjects: external job applicants.

Model: L2-regularised logistic regression trained on historical hiring
outcomes (``hired``). Numeric profile fields are standardised; education level
and home postcode area are one-hot encoded. The fitted coefficients are exposed
via :meth:`SuitabilityScorer.coefficients` so recruiters can see which
attributes move a score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE, SUITABILITY, TEST_SIZE, SuitabilityConfig


@dataclass
class SuitabilityMetrics:
    """Hold-out evaluation of the suitability model."""

    roc_auc: float
    accuracy: float


class SuitabilityScorer:
    """Logistic-regression suitability score for applicants."""

    def __init__(self, config: SuitabilityConfig = SUITABILITY) -> None:
        self.config = config
        preprocess = ColumnTransformer(
            [
                ("num", StandardScaler(), list(config.numeric_features)),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 list(config.categorical_features)),
            ]
        )
        self.model = Pipeline(
            [
                ("preprocess", preprocess),
                ("clf", LogisticRegression(
                    C=config.regularisation_c, class_weight="balanced", max_iter=1000
                )),
            ]
        )

    @property
    def feature_columns(self) -> list:
        return list(self.config.numeric_features) + list(self.config.categorical_features)

    def fit(self, candidates: pd.DataFrame, target: str = "hired") -> SuitabilityMetrics:
        """Train on historical outcomes and report hold-out metrics."""
        X = candidates[self.feature_columns]
        y = candidates[target].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        proba = self.model.predict_proba(X_te)[:, 1]
        return SuitabilityMetrics(
            roc_auc=float(roc_auc_score(y_te, proba)),
            accuracy=float(accuracy_score(y_te, (proba >= 0.5).astype(int))),
        )

    def band(self, score: float) -> str:
        """Map a 0-100 score to its suitability band."""
        for name, cutoff in sorted(self.config.bands.items(), key=lambda kv: -kv[1]):
            if score >= cutoff:
                return name
        return "unlikely"

    def score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score applicants; returns id, score (0-100) and band."""
        proba = self.model.predict_proba(candidates[self.feature_columns])[:, 1]
        out = pd.DataFrame({
            "candidate_id": candidates["candidate_id"].to_numpy(),
            "suitability_score": np.round(proba * 100, 1),
        })
        out["band"] = out["suitability_score"].map(self.band)
        return out

    def coefficients(self) -> pd.Series:
        """Model coefficients by encoded feature name, largest magnitude first."""
        names = self.model.named_steps["preprocess"].get_feature_names_out()
        coefs = self.model.named_steps["clf"].coef_.ravel()
        series = pd.Series(coefs, index=names)
        return series.reindex(series.abs().sort_values(ascending=False).index)
