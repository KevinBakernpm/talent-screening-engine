"""End-to-end training and scoring pipeline.

Generates (or loads) the four source tables, trains every model, prints
hold-out metrics and scores a handful of example records the way the
production batch jobs do.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .attrition_risk import AttritionModel
from .cv_ranking import CVRanker
from .interview_scoring import InterviewScorer
from .promotion_recommender import PromotionRecommender
from .suitability_scoring import SuitabilityScorer

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "data" / "generate_synthetic.py"


def load_synthetic(seed: int) -> Dict[str, pd.DataFrame]:
    """Import ``data/generate_synthetic.py`` and generate all tables."""
    spec = importlib.util.spec_from_file_location("generate_synthetic", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"synthetic data generator not found at {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_all(seed)


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _metrics(metrics: object) -> None:
    for key, value in asdict(metrics).items():
        print(f"  {key:<26} {value:.3f}")


def run(seed: int = 2026, data: Optional[Dict[str, pd.DataFrame]] = None,
        verbose: bool = True) -> Dict[str, object]:
    """Train all models and return them together with their metrics."""
    tables = data if data is not None else load_synthetic(seed)
    jobs, candidates = tables["jobs"], tables["candidates"]
    interviews, employees = tables["interviews"], tables["employees"]
    show = print if verbose else (lambda *a, **k: None)
    pd.set_option("display.width", 120)

    ranker = CVRanker()
    rank_metrics = ranker.fit(candidates, jobs)
    suitability = SuitabilityScorer()
    suit_metrics = suitability.fit(candidates)
    interviewer = InterviewScorer()
    int_metrics = interviewer.fit(interviews)
    attrition = AttritionModel()
    att_metrics = attrition.fit(employees)
    promotion = PromotionRecommender()
    promo_metrics = promotion.fit(employees)

    if verbose:
        _section("1. CV ranking + shortlist classifier")
        _metrics(rank_metrics)
        show("\nTop applicants for REQ-101 (Data Analyst):")
        show(ranker.rank(candidates, jobs, "data_analyst", top_k=5).to_string(index=False))

        _section("2. Candidate suitability scoring")
        _metrics(suit_metrics)
        show("\nLargest coefficients:")
        show(suitability.coefficients().head(6).round(3).to_string())
        show("\nExample scores:")
        show(suitability.score(candidates.head(5)).to_string(index=False))

        _section("3. Interview communication scoring")
        _metrics(int_metrics)
        show("\nExample interviews:")
        show(interviewer.score(interviews.head(5)).to_string(index=False))

        _section("4. Attrition / flight-risk model")
        _metrics(att_metrics)
        show("\nTop global drivers:")
        show(attrition.feature_importance().head(5).round(3).to_string())
        show("\nHighest-risk employees:")
        show(attrition.predict(employees).head(5).to_string(index=False))

        _section("5. Rating calibration + promotion recommender")
        _metrics(promo_metrics)
        show("\nPromotion shortlist:")
        show(promotion.recommend(employees, top_k=5).to_string(index=False))

    return {
        "models": {"cv_ranker": ranker, "suitability": suitability,
                   "interview": interviewer, "attrition": attrition,
                   "promotion": promotion},
        "metrics": {"cv_ranking": rank_metrics, "suitability": suit_metrics,
                    "interview": int_metrics, "attrition": att_metrics,
                    "promotion": promo_metrics},
    }


if __name__ == "__main__":
    run()
