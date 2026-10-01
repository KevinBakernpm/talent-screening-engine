"""CV ranking and automated shortlisting of job applicants.

Decision: which applicants for an open requisition are put forward to a
recruiter for review, and in what order.

Subjects: external job applicants.

Two stages are combined:

1. **Relevance matching** - every CV and the job description are embedded
   with a TF-IDF vectoriser (word uni- and bi-grams) and ranked by cosine
   similarity to the job description.
2. **Shortlist classifier** - a gradient-boosted tree ensemble trained on
   historical recruiter shortlisting decisions. It takes the similarity score
   plus profile features (experience, education, employment gaps, skill
   keyword hits) and outputs the probability that a recruiter would shortlist
   the applicant. Applicants above ``shortlist_threshold`` are auto-shortlisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .config import CV_RANKING, RANDOM_STATE, TEST_SIZE, CVRankingConfig
from .text_utils import normalise, tokenize

EDUCATION_ORDINAL: Dict[str, int] = {
    "secondary": 0, "diploma": 1, "bachelor": 2, "master": 3, "doctorate": 4,
}
SHORTLIST_FEATURES: List[str] = [
    "jd_similarity",
    "keyword_hits",
    "years_experience",
    "education_ordinal",
    "employment_gap_months",
    "num_previous_employers",
]


@dataclass
class ShortlistMetrics:
    """Hold-out evaluation of the shortlist classifier."""

    roc_auc: float
    precision: float
    recall: float


class CVRanker:
    """Ranks CVs against job descriptions and predicts shortlisting."""

    def __init__(self, config: CVRankingConfig = CV_RANKING) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalise,
            ngram_range=config.ngram_range,
            max_features=config.max_features,
            sublinear_tf=True,
            stop_words="english",
        )
        self.classifier = GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=RANDOM_STATE,
        )
        self._fitted = False

    def similarity(self, cvs: pd.Series, job_description: str) -> np.ndarray:
        """Cosine similarity between each CV and a job description."""
        cv_matrix = self.vectorizer.transform(cvs.fillna(""))
        jd_vector = self.vectorizer.transform([job_description])
        return cosine_similarity(cv_matrix, jd_vector).ravel()

    @staticmethod
    def keyword_hits(cvs: pd.Series, job_description: str) -> np.ndarray:
        """Number of distinct job-description tokens that appear in each CV."""
        jd_tokens = {t for t in tokenize(job_description) if len(t) > 3}
        return np.array([len(jd_tokens & set(tokenize(cv))) for cv in cvs.fillna("")])

    def build_features(self, candidates: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
        """Join each candidate to their requisition and compute model features."""
        descriptions = jobs.set_index("role")["description"]
        parts = []
        for role, group in candidates.groupby("applied_role", sort=False):
            jd = descriptions[role]
            feats = pd.DataFrame(index=group.index)
            feats["jd_similarity"] = self.similarity(group["cv_text"], jd)
            feats["keyword_hits"] = self.keyword_hits(group["cv_text"], jd)
            parts.append(feats)
        features = pd.concat(parts).loc[candidates.index]
        features["years_experience"] = candidates["years_experience"]
        features["education_ordinal"] = candidates["education_level"].map(EDUCATION_ORDINAL).fillna(0)
        features["employment_gap_months"] = candidates["employment_gap_months"]
        features["num_previous_employers"] = candidates["num_previous_employers"]
        return features[SHORTLIST_FEATURES]

    def fit(self, candidates: pd.DataFrame, jobs: pd.DataFrame) -> ShortlistMetrics:
        """Fit the vectoriser on all CVs and JDs, then train the shortlist model."""
        corpus = pd.concat([candidates["cv_text"], jobs["description"]], ignore_index=True)
        self.vectorizer.fit(corpus)
        X = self.build_features(candidates, jobs)
        y = candidates["shortlisted"].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.classifier.fit(X_tr, y_tr)
        self._fitted = True
        proba = self.classifier.predict_proba(X_te)[:, 1]
        pred = (proba >= self.config.shortlist_threshold).astype(int)
        return ShortlistMetrics(
            roc_auc=float(roc_auc_score(y_te, proba)),
            precision=float(precision_score(y_te, pred, zero_division=0)),
            recall=float(recall_score(y_te, pred, zero_division=0)),
        )

    def rank(
        self,
        candidates: pd.DataFrame,
        jobs: pd.DataFrame,
        role: str,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the top applicants for ``role`` with shortlist decisions."""
        if not self._fitted:
            raise RuntimeError("CVRanker.fit must be called before rank")
        pool = candidates[candidates["applied_role"] == role]
        X = self.build_features(pool, jobs)
        result = pool[["candidate_id"]].copy()
        result["jd_similarity"] = X["jd_similarity"].round(3)
        result["shortlist_probability"] = self.classifier.predict_proba(X)[:, 1].round(3)
        result["shortlisted"] = result["shortlist_probability"] >= self.config.shortlist_threshold
        result = result.sort_values(["shortlist_probability", "jd_similarity"], ascending=False)
        return result.head(top_k or self.config.top_k).reset_index(drop=True)
