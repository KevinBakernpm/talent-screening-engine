"""talent-screening-engine: models for hiring and workforce decisions."""

from .attrition_risk import AttritionModel
from .cv_ranking import CVRanker
from .interview_scoring import InterviewScorer
from .promotion_recommender import PromotionRecommender, calibrate_ratings
from .suitability_scoring import SuitabilityScorer

__all__ = [
    "AttritionModel",
    "CVRanker",
    "InterviewScorer",
    "PromotionRecommender",
    "SuitabilityScorer",
    "calibrate_ratings",
]
__version__ = "0.4.1"
