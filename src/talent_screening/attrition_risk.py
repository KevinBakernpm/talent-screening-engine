"""Employee attrition (flight-risk) prediction.

Decision: the probability that each current employee leaves within the next
12 months and a risk tier (``high`` / ``medium`` / ``low``). The monthly
flight-risk report is sent to HR business partners and line managers, who use
it to prioritise retention conversations, counter-offers and succession
planning.

Subjects: current employees.

Model: random forest classifier trained on historical leaver data, with
balanced class weights. Global drivers come from impurity-based feature
importances; per-employee drivers are the features where the employee deviates
most from the workforce median, weighted by importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import ATTRITION, RANDOM_STATE, TEST_SIZE, AttritionConfig


@dataclass
class AttritionMetrics:
    """Hold-out evaluation of the attrition model."""

    roc_auc: float
    average_precision: float
    base_rate: float


class AttritionModel:
    """Random forest flight-risk model."""

    def __init__(self, config: AttritionConfig = ATTRITION) -> None:
        self.config = config
        self.features: List[str] = list(config.features)
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self._medians: pd.Series = pd.Series(dtype=float)
        self._scales: pd.Series = pd.Series(dtype=float)

    def fit(self, employees: pd.DataFrame, target: str = "left_within_12m") -> AttritionMetrics:
        """Train on historical leavers and report hold-out metrics."""
        X = employees[self.features]
        y = employees[target].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        self._medians = X_tr.median()
        self._scales = X_tr.std().replace(0, 1.0)
        proba = self.model.predict_proba(X_te)[:, 1]
        return AttritionMetrics(
            roc_auc=float(roc_auc_score(y_te, proba)),
            average_precision=float(average_precision_score(y_te, proba)),
            base_rate=float(y.mean()),
        )

    def tier(self, probability: float) -> str:
        """Map a leave probability to a risk tier."""
        if probability >= self.config.high_risk:
            return "high"
        if probability >= self.config.medium_risk:
            return "medium"
        return "low"

    def feature_importance(self) -> pd.Series:
        """Global feature importances, largest first."""
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(
            ascending=False
        )

    def top_drivers(self, employees: pd.DataFrame, n: int = 3) -> List[str]:
        """Comma-separated per-employee drivers for the flight-risk report."""
        importance = pd.Series(self.model.feature_importances_, index=self.features)
        deviation = (employees[self.features] - self._medians).abs() / self._scales
        weighted = deviation * importance
        return [", ".join(row.nlargest(n).index) for _, row in weighted.iterrows()]

    def predict(self, employees: pd.DataFrame) -> pd.DataFrame:
        """Score employees; returns id, probability, tier and drivers."""
        proba = self.model.predict_proba(employees[self.features])[:, 1]
        out = pd.DataFrame({
            "employee_id": employees["employee_id"].to_numpy(),
            "leave_probability": np.round(proba, 3),
        })
        out["risk_tier"] = out["leave_probability"].map(self.tier)
        out["drivers"] = self.top_drivers(employees)
        return out.sort_values("leave_probability", ascending=False).reset_index(drop=True)
