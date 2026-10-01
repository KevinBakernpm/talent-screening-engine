"""Central configuration for model hyper-parameters and decision thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

RANDOM_STATE = 42
TEST_SIZE = 0.25


@dataclass(frozen=True)
class CVRankingConfig:
    """Settings for CV-to-job matching and the shortlist classifier."""

    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    shortlist_threshold: float = 0.5
    top_k: int = 10


@dataclass(frozen=True)
class SuitabilityConfig:
    """Settings for the structured candidate suitability model."""

    numeric_features: Tuple[str, ...] = (
        "years_experience",
        "years_since_graduation",
        "employment_gap_months",
        "num_previous_employers",
        "certifications",
        "referred",
    )
    categorical_features: Tuple[str, ...] = ("education_level", "postcode_area")
    regularisation_c: float = 1.0
    bands: Dict[str, float] = field(
        default_factory=lambda: {"strong": 70.0, "possible": 45.0, "unlikely": 0.0}
    )


@dataclass(frozen=True)
class InterviewConfig:
    """Settings for transcript-based communication scoring."""

    nb_alpha: float = 0.5
    lexicon_weight: float = 0.3
    pass_threshold: float = 0.55


@dataclass(frozen=True)
class AttritionConfig:
    """Settings for the employee flight-risk model."""

    features: Tuple[str, ...] = (
        "age",
        "tenure_years",
        "salary_band",
        "months_since_promotion",
        "overtime_hours_month",
        "commute_km",
        "engagement_score",
        "manager_changes_2y",
        "remote_days_week",
        "training_hours_year",
        "parental_leave_last_2y",
    )
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 5
    high_risk: float = 0.5
    medium_risk: float = 0.3


@dataclass(frozen=True)
class PromotionConfig:
    """Settings for rating calibration and promotion recommendation."""

    shrinkage: float = 5.0
    features: Tuple[str, ...] = (
        "calibrated_rating",
        "tenure_years",
        "months_since_promotion",
        "training_hours_year",
        "salary_band",
        "age",
        "parental_leave_last_2y",
    )
    recommend_threshold: float = 0.4


CV_RANKING = CVRankingConfig()
SUITABILITY = SuitabilityConfig()
INTERVIEW = InterviewConfig()
ATTRITION = AttritionConfig()
PROMOTION = PromotionConfig()
