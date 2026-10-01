"""Behavioural tests for each model and the synthetic data generator."""

from __future__ import annotations

import pandas as pd
import pytest

from talent_screening import (
    AttritionModel,
    CVRanker,
    InterviewScorer,
    PromotionRecommender,
    SuitabilityScorer,
    calibrate_ratings,
)
from talent_screening.interview_scoring import lexicon_score
from talent_screening.pipeline import load_synthetic


def test_generator_is_deterministic():
    a, b = load_synthetic(seed=11), load_synthetic(seed=11)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_cv_ranker_beats_chance_and_ranks(tables):
    ranker = CVRanker()
    metrics = ranker.fit(tables["candidates"], tables["jobs"])
    assert metrics.roc_auc > 0.7
    top = ranker.rank(tables["candidates"], tables["jobs"], "backend_engineer", top_k=5)
    assert len(top) == 5
    assert top["shortlist_probability"].is_monotonic_decreasing


def test_cv_ranker_requires_fit(tables):
    with pytest.raises(RuntimeError):
        CVRanker().rank(tables["candidates"], tables["jobs"], "data_analyst")


def test_suitability_scores_are_bounded(tables):
    scorer = SuitabilityScorer()
    metrics = scorer.fit(tables["candidates"])
    assert metrics.roc_auc > 0.6
    scores = scorer.score(tables["candidates"].head(50))
    assert scores["suitability_score"].between(0, 100).all()
    assert set(scores["band"]) <= {"strong", "possible", "unlikely"}


def test_lexicon_prefers_confident_answers():
    strong = "I led the project, delivered early and measured a clear improvement."
    weak = "Um, I guess maybe I sort of helped, I'm not really sure, no idea."
    assert lexicon_score(strong) > 0.5 > lexicon_score(weak)


def test_interview_scorer_accuracy(tables):
    scorer = InterviewScorer()
    metrics = scorer.fit(tables["interviews"])
    assert metrics.accuracy > 0.65
    scored = scorer.score(tables["interviews"].head(10))
    assert scored["communication_score"].between(0, 1).all()


def test_attrition_tiers_and_drivers(tables):
    model = AttritionModel()
    metrics = model.fit(tables["employees"])
    assert metrics.roc_auc > 0.65
    report = model.predict(tables["employees"].head(20))
    assert set(report["risk_tier"]) <= {"high", "medium", "low"}
    assert report["drivers"].str.count(",").eq(2).all()


def test_calibration_reduces_manager_spread(tables):
    emp = tables["employees"].copy()
    emp["calibrated_rating"] = calibrate_ratings(emp)
    before = emp.groupby("manager_id")["raw_rating"].mean().var()
    after = emp.groupby("manager_id")["calibrated_rating"].mean().var()
    assert after < before


def test_promotion_recommender(tables):
    rec = PromotionRecommender()
    metrics = rec.fit(tables["employees"])
    assert metrics.roc_auc > 0.6
    top = rec.recommend(tables["employees"], top_k=3)
    assert len(top) == 3
