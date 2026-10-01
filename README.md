# talent-screening-engine

Machine-learning models that support hiring and workforce decisions, from the
first CV screen through to retention and promotion planning. This repository is
the open-sourced reference implementation of the models our People Analytics
team runs in its weekly and monthly batch jobs. It ships with a deterministic
synthetic data generator so that every model can be trained and evaluated
without access to real applicant or employee records.

## What it does

| Stage | Module | Model | Output | Who acts on it |
|-------|--------|-------|--------|----------------|
| CV screen | `cv_ranking.py` | TF-IDF + cosine similarity, gradient-boosted shortlist classifier | Ranked applicant list per requisition, auto-shortlist flag | Recruiters |
| Application triage | `suitability_scoring.py` | Logistic regression on structured profile data | 0-100 suitability score and band (strong / possible / unlikely) | Recruiters, automated decline workflow |
| First interview | `interview_scoring.py` | Lexicon features + multinomial Naive Bayes on transcripts | Communication score and progress-to-final flag | Hiring panel |
| Retention | `attrition_risk.py` | Random forest | 12-month leave probability, risk tier, top drivers | HR business partners, line managers |
| Reward and promotion | `promotion_recommender.py` | Manager-leniency calibration + gradient boosting | Calibrated rating, promotion probability and recommendation | Bonus allocation, promotion committee |

## Architecture

```
                    +-----------------------------+
 ATS export ------->|  cv_ranking.CVRanker        |--> ranked shortlist per requisition
 (CV text, profile) |  suitability.SuitabilityScorer --> suitability score + band
                    +-----------------------------+
                                  |
 Interview platform               v
 (speech-to-text) ---->  interview_scoring.InterviewScorer --> progress-to-final flag

 HRIS snapshot ------->  attrition_risk.AttritionModel      --> monthly flight-risk report
 (employees, ratings)    promotion_recommender.PromotionRecommender
                                                            --> calibrated ratings, promotion shortlist
```

All models share a small configuration module (`config.py`) holding feature
lists, hyper-parameters and decision thresholds, so threshold changes are
reviewed in one place.

### How each model is used

**CV ranking.** For every open requisition, all applicant CVs are vectorised
together with the job description and ranked by cosine similarity. A
gradient-boosted classifier trained on past recruiter shortlisting decisions
combines that similarity with experience, education, employment gaps and
keyword hits. Applicants above the shortlist threshold are moved to the
recruiter's review queue; the rest stay in the pool.

**Suitability scoring.** Shortlisted applicants receive a suitability score
from a logistic regression trained on historical hire outcomes. Scores in the
`strong` band are invited to interview directly, `possible` applicants are
reviewed by a recruiter, and `unlikely` applicants receive the standard
decline email. Coefficients are exported with each model version.

**Interview scoring.** Transcripts of recorded first-round interviews are
scored for communication quality by blending a rule-based lexicon score with a
Naive Bayes classifier trained on past panel ratings. The resulting flag is
attached to the candidate's scorecard before the panel debrief.

**Attrition risk.** Each month, every current employee is scored for the
probability of leaving within 12 months. High and medium tiers, together with
their top drivers, are sent to HR business partners to plan retention
conversations and succession cover.

**Calibration and promotion.** Raw manager ratings are adjusted for manager
leniency before they feed bonus allocation. The calibrated rating and
career-history features are then used to pre-populate the promotion
committee's shortlist.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py             # generate data, train all models, print metrics
python -m pytest -q                # run the test suite
```

To write the synthetic tables to CSV for exploration:

```bash
python data/generate_synthetic.py --seed 2026 --out data/generated
```

Using a model directly:

```python
from talent_screening import AttritionModel
from talent_screening.pipeline import load_synthetic

tables = load_synthetic(seed=2026)
model = AttritionModel()
print(model.fit(tables["employees"]))
print(model.predict(tables["employees"]).head())
```

## Data

All data in this repository is synthetic and generated deterministically from
a seed by `data/generate_synthetic.py`. The generator reproduces the schema of
our ATS, interview-platform and HRIS extracts:

- `jobs`: requisition id, role family, title, job description
- `candidates`: CV text, years of experience, education level, years since
  graduation, postcode area, employment gaps, previous employers,
  certifications, referral flag, shortlisted / hired outcomes
- `interviews`: transcript text and panel communication rating
- `employees`: department, manager, age, tenure, salary band, time since
  promotion, overtime, commute distance, engagement, manager changes, remote
  days, training hours, parental leave, raw rating, leaver and promotion
  outcomes

## Project structure

```
talent-screening-engine/
├── data/
│   └── generate_synthetic.py      # deterministic synthetic data generator
├── src/talent_screening/
│   ├── __init__.py
│   ├── config.py                  # features, hyper-parameters, thresholds
│   ├── text_utils.py              # text normalisation helpers
│   ├── cv_ranking.py              # CV ranking + shortlist classifier
│   ├── suitability_scoring.py     # applicant suitability score
│   ├── interview_scoring.py       # transcript communication scoring
│   ├── attrition_risk.py          # employee flight-risk model
│   ├── promotion_recommender.py   # rating calibration + promotion model
│   └── pipeline.py                # train, evaluate and score end to end
├── tests/
│   ├── conftest.py
│   └── test_models.py
├── run_pipeline.py                # CLI entry point
├── requirements.txt
└── LICENSE
```

## Contributing

Pull requests are welcome. Please keep new models in their own module with a
docstring describing the decision they support, add thresholds to `config.py`
rather than hard-coding them, and include tests that run against the synthetic
data.

## License

MIT. See [LICENSE](LICENSE).
