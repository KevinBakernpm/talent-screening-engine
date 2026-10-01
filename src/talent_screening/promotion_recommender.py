"""Performance rating calibration and promotion recommendation.

Decision: (a) a calibrated performance rating for every employee, which feeds
the annual bonus pool allocation, and (b) a promotion-readiness probability
and recommendation that pre-populates the promotion committee's shortlist.

Subjects: current employees and their line managers' ratings.

Stage 1 - **Rating calibration.** Managers differ in leniency. Each raw rating
is adjusted by removing the rating manager's mean offset from the company
mean, shrunk towards zero for managers with few direct reports
(empirical-Bayes style shrinkage with strength ``PromotionConfig.shrinkage``).

Stage 2 - **Promotion recommender.** A gradient-boosted classifier trained on
past promotion-cycle outcomes using the calibrated rating, tenure, time since
last promotion, training hours, salary band, age and recent leave history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import PROMOTION, RANDOM_STATE, TEST_SIZE, PromotionConfig


def calibrate_ratings(
    employees: pd.DataFrame,
    rating_col: str = "raw_rating",
    manager_col: str = "manager_id",
    shrinkage: float = PROMOTION.shrinkage,
) -> pd.Series:
    """Remove shrunken per-manager leniency offsets from raw ratings."""
    global_mean = employees[rating_col].mean()
    grouped = employees.groupby(manager_col)[rating_col].agg(["mean", "count"])
    offset = (grouped["mean"] - global_mean) * grouped["count"] / (grouped["count"] + shrinkage)
    calibrated = employees[rating_col] - employees[manager_col].map(offset)
    return calibrated.clip(1.0, 5.0).round(2).rename("calibrated_rating")


@dataclass
class PromotionMetrics:
    """Hold-out evaluation of the promotion model plus calibration effect."""

    roc_auc: float
    raw_rating_auc: float
    manager_variance_before: float
    manager_variance_after: float


class PromotionRecommender:
    """Calibrates ratings and recommends promotion candidates."""

    def __init__(self, config: PromotionConfig = PROMOTION) -> None:
        self.config = config
        self.features: List[str] = list(config.features)
        self.model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        )

    def prepare(self, employees: pd.DataFrame) -> pd.DataFrame:
        """Attach the calibrated rating column."""
        out = employees.copy()
        out["calibrated_rating"] = calibrate_ratings(out, shrinkage=self.config.shrinkage)
        return out

    def fit(self, employees: pd.DataFrame, target: str = "promoted_next_cycle") -> PromotionMetrics:
        """Calibrate ratings, train the recommender and report metrics."""
        data = self.prepare(employees)
        X = data[self.features]
        y = data[target].to_numpy()
        X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
            X, y, data["raw_rating"], test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y,
        )
        self.model.fit(X_tr, y_tr)
        proba = self.model.predict_proba(X_te)[:, 1]
        by_mgr_raw = data.groupby("manager_id")["raw_rating"].mean().var()
        by_mgr_cal = data.groupby("manager_id")["calibrated_rating"].mean().var()
        return PromotionMetrics(
            roc_auc=float(roc_auc_score(y_te, proba)),
            raw_rating_auc=float(roc_auc_score(y_te, raw_te)),
            manager_variance_before=float(by_mgr_raw),
            manager_variance_after=float(by_mgr_cal),
        )

    def recommend(self, employees: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Return the employees most ready for promotion."""
        data = self.prepare(employees)
        proba = self.model.predict_proba(data[self.features])[:, 1]
        out = pd.DataFrame({
            "employee_id": data["employee_id"].to_numpy(),
            "raw_rating": data["raw_rating"].to_numpy(),
            "calibrated_rating": data["calibrated_rating"].to_numpy(),
            "promotion_probability": np.round(proba, 3),
        })
        out["recommended"] = out["promotion_probability"] >= self.config.recommend_threshold
        out = out.sort_values("promotion_probability", ascending=False)
        return out.head(top_k).reset_index(drop=True)
