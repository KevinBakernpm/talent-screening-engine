"""Communication scoring of interview transcripts.

Decision: a communication score (0-1) and pass/fail flag for each first-round
interview, used alongside the panel's notes to decide whether a candidate
progresses to the final round.

Subjects: job applicants who have completed a recorded interview. Transcripts
are produced from the interview recording by the video platform's
speech-to-text export before they reach this module.

Model: a blend of two signals.

* **Lexicon score** - rule-based counts of confident / positive phrasing,
  hedging, filler words and negative sentiment, normalised by length.
* **Naive Bayes classifier** - multinomial NB over word uni/bi-gram counts,
  trained on historical panel communication ratings (``strong`` / ``weak``).

The final score is ``(1 - w) * p_nb + w * lexicon`` where ``w`` is
``InterviewConfig.lexicon_weight``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import INTERVIEW, RANDOM_STATE, TEST_SIZE, InterviewConfig
from .text_utils import count_phrases, normalise, tokenize

CONFIDENT = ["led", "delivered", "achieved", "confident", "ownership", "succeeded",
             "concrete", "measured", "resolved", "improvement", "clear", "excited"]
HEDGES = ["maybe", "i think", "i guess", "sort of", "kind of", "probably",
          "not really sure", "i suppose", "hard to say"]
FILLERS = ["um", "er", "uh", "like", "you know", "honestly"]
NEGATIVE = ["didn't like", "problem", "unhappy", "difficult", "no idea",
            "never", "can't", "frankly"]


def lexicon_features(transcript: str) -> Dict[str, float]:
    """Per-100-token rates of confident, hedging, filler and negative phrases."""
    n_tokens = max(len(tokenize(transcript)), 1)
    scale = 100.0 / n_tokens
    return {
        "confident_rate": count_phrases(transcript, CONFIDENT) * scale,
        "hedge_rate": count_phrases(transcript, HEDGES) * scale,
        "filler_rate": count_phrases(transcript, FILLERS) * scale,
        "negative_rate": count_phrases(transcript, NEGATIVE) * scale,
    }


def lexicon_score(transcript: str) -> float:
    """Squash lexicon rates into a 0-1 communication score."""
    f = lexicon_features(transcript)
    raw = (0.9 * f["confident_rate"] - 0.6 * f["hedge_rate"]
           - 0.5 * f["filler_rate"] - 0.7 * f["negative_rate"])
    return float(1.0 / (1.0 + np.exp(-raw / 2.0)))


@dataclass
class InterviewMetrics:
    """Hold-out evaluation of the blended interview scorer."""

    accuracy: float
    f1: float
    nb_only_accuracy: float


class InterviewScorer:
    """Blended lexicon + Naive Bayes communication scorer."""

    def __init__(self, config: InterviewConfig = INTERVIEW) -> None:
        self.config = config
        self.model = Pipeline(
            [
                ("counts", CountVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2)),
                ("nb", MultinomialNB(alpha=config.nb_alpha)),
            ]
        )

    def _positive_index(self) -> int:
        return int(np.where(self.model.classes_ == "strong")[0][0])

    def fit(self, interviews: pd.DataFrame) -> InterviewMetrics:
        """Train the NB component and evaluate the blended score."""
        X = interviews["transcript"]
        y = interviews["communication_rating"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        blended = self.predict_proba(X_te)
        truth = (y_te == "strong").astype(int).to_numpy()
        pred = (blended >= self.config.pass_threshold).astype(int)
        nb_pred = (self.model.predict(X_te) == "strong").astype(int)
        return InterviewMetrics(
            accuracy=float(accuracy_score(truth, pred)),
            f1=float(f1_score(truth, pred)),
            nb_only_accuracy=float(accuracy_score(truth, nb_pred)),
        )

    def predict_proba(self, transcripts: pd.Series) -> np.ndarray:
        """Blended communication score in [0, 1] for each transcript."""
        nb = self.model.predict_proba(transcripts)[:, self._positive_index()]
        lex = np.array([lexicon_score(t) for t in transcripts])
        w = self.config.lexicon_weight
        return (1 - w) * nb + w * lex

    def score(self, interviews: pd.DataFrame) -> pd.DataFrame:
        """Score interviews; returns id, communication score and pass flag."""
        proba = self.predict_proba(interviews["transcript"])
        return pd.DataFrame({
            "interview_id": interviews["interview_id"].to_numpy(),
            "communication_score": np.round(proba, 3),
            "progress_to_final": proba >= self.config.pass_threshold,
        })
