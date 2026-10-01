"""Deterministic synthetic data for the talent-screening-engine.

Produces four tables that mirror the shape of the internal extracts the
models were originally developed against:

* ``jobs``        - open requisitions with a free-text job description.
* ``candidates``  - applicants with CV text, structured profile fields and
                    historical screening outcomes (shortlisted / hired).
* ``interviews``  - first-round interview transcripts with the panel's
                    communication rating.
* ``employees``   - current and former staff with HR attributes, a raw
                    manager rating, attrition outcome and promotion outcome.

Every value is generated from a seeded ``numpy`` random generator, so the same
seed always yields byte-identical tables.

Usage::

    python data/generate_synthetic.py --seed 2026 --out data/generated
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

DEFAULT_SEED = 2026

ROLES: Dict[str, Dict[str, object]] = {
    "data_analyst": {
        "title": "Data Analyst",
        "skills": ["sql", "python", "pandas", "tableau", "statistics", "excel",
                   "dashboards", "forecasting", "a/b testing", "data modelling"],
        "description": (
            "We are hiring a Data Analyst to build dashboards, write SQL and Python "
            "to analyse product data, run A/B testing and forecasting, and present "
            "statistics to stakeholders. Experience with pandas, Tableau and data "
            "modelling is essential."
        ),
    },
    "backend_engineer": {
        "title": "Backend Engineer",
        "skills": ["java", "python", "microservices", "kubernetes", "postgresql",
                   "rest apis", "docker", "aws", "ci/cd", "system design"],
        "description": (
            "Join our platform team as a Backend Engineer designing microservices "
            "and REST APIs in Java or Python. You will deploy with Docker and "
            "Kubernetes on AWS, own PostgreSQL schemas, maintain CI/CD pipelines "
            "and contribute to system design reviews."
        ),
    },
    "sales_executive": {
        "title": "Sales Executive",
        "skills": ["b2b sales", "crm", "negotiation", "pipeline management",
                   "prospecting", "account management", "salesforce", "quota",
                   "presentations", "lead generation"],
        "description": (
            "We need a Sales Executive to own B2B sales from prospecting and lead "
            "generation through negotiation and close. You will manage a pipeline "
            "in our CRM (Salesforce), deliver presentations, grow account "
            "management revenue and consistently exceed quota."
        ),
    },
    "operations_coordinator": {
        "title": "Operations Coordinator",
        "skills": ["logistics", "scheduling", "vendor management", "inventory",
                   "process improvement", "erp", "reporting", "budgeting",
                   "stakeholder communication", "compliance"],
        "description": (
            "The Operations Coordinator keeps our sites running: scheduling, "
            "logistics, inventory control and vendor management in the ERP. You "
            "will drive process improvement, prepare reporting and budgeting packs "
            "and handle stakeholder communication across teams."
        ),
    },
}

GENERIC_PHRASES = [
    "strong team player", "excellent communication skills", "self-motivated",
    "detail oriented", "fast learner", "cross-functional collaboration",
    "problem solving", "time management",
]

POSTCODE_AREAS = ["N1", "E3", "SE15", "SW6", "W2", "NW8", "M14", "B12", "LS6", "BS3"]
POSTCODE_EFFECT = dict(zip(POSTCODE_AREAS, [0.25, -0.30, -0.35, 0.30, 0.20, 0.35,
                                            -0.15, -0.25, -0.05, 0.05]))
EDUCATION_LEVELS = ["secondary", "diploma", "bachelor", "master", "doctorate"]
DEPARTMENTS = ["engineering", "sales", "operations", "finance", "people", "product"]

STRONG_LINES = [
    "In my last role I led the migration and we delivered two weeks early.",
    "I measured the impact carefully and the result was a clear improvement.",
    "I took ownership of the problem and explained the trade-offs to the team.",
    "I am confident I can contribute from day one and I enjoy learning quickly.",
    "We agreed clear goals, I communicated progress weekly and we succeeded.",
    "I resolved the conflict by listening first and then proposing a solution.",
    "I am excited about this opportunity because it matches my experience well.",
    "To give a concrete example, I reduced processing time by thirty percent.",
]
WEAK_LINES = [
    "Um, I think maybe I sort of helped with that, I'm not really sure.",
    "I guess it was okay, kind of, it depends I suppose.",
    "Honestly I didn't like the team and the manager was a problem.",
    "I don't know, I probably would just wait and see what happens.",
    "Er, it's hard to say, like, I wasn't really involved in the outcome.",
    "I have no idea, that was never really my responsibility.",
    "Sort of, um, maybe, I can't really remember the details.",
    "It was difficult and frankly I was unhappy most of the time.",
]
NEUTRAL_LINES = [
    "My previous position was in a mid-sized company.",
    "I studied at university and then joined my first employer.",
    "The project involved several teams and a number of systems.",
    "I usually work from the office three days a week.",
]


def _cv_text(rng: np.random.Generator, role_key: str, years: int, edu: str) -> str:
    role = ROLES[role_key]
    skills = list(rng.choice(role["skills"], size=rng.integers(3, 8), replace=False))
    if rng.random() < 0.4:
        other = ROLES[rng.choice(list(ROLES))]
        skills += list(rng.choice(other["skills"], size=2, replace=False))
    extras = list(rng.choice(GENERIC_PHRASES, size=2, replace=False))
    return (
        f"{role['title']} with {years} years of experience. "
        f"Education: {edu}. Skills: {', '.join(skills)}. "
        f"{extras[0].capitalize()}, {extras[1]}."
    )


def generate_jobs() -> pd.DataFrame:
    """Return one requisition per role family."""
    return pd.DataFrame(
        [{"job_id": f"REQ-{i + 101}", "role": key, "title": r["title"],
          "description": r["description"]} for i, (key, r) in enumerate(ROLES.items())]
    )


def generate_candidates(rng: np.random.Generator, n: int = 1200) -> pd.DataFrame:
    """Applicants with CV text, profile fields and historical outcomes."""
    role_keys = list(ROLES)
    rows = []
    for i in range(n):
        applied = role_keys[rng.integers(len(role_keys))]
        background = applied if rng.random() < 0.55 else role_keys[rng.integers(len(role_keys))]
        years = int(rng.integers(0, 21))
        edu_idx = int(rng.choice(5, p=[0.10, 0.15, 0.45, 0.25, 0.05]))
        postcode = POSTCODE_AREAS[rng.integers(len(POSTCODE_AREAS))]
        gap = int(rng.choice([0, 0, 0, 3, 6, 12, 24]))
        cv = _cv_text(rng, background, years, EDUCATION_LEVELS[edu_idx])
        rows.append({
            "candidate_id": f"C{i:05d}",
            "applied_role": applied,
            "cv_text": cv,
            "years_experience": years,
            "education_level": EDUCATION_LEVELS[edu_idx],
            "years_since_graduation": years + int(rng.integers(0, 5)),
            "postcode_area": postcode,
            "employment_gap_months": gap,
            "num_previous_employers": int(rng.integers(1, 8)),
            "certifications": int(rng.poisson(1.2)),
            "referred": int(rng.random() < 0.15),
            "_fit": 1.0 if background == applied else 0.0,
            "_edu": edu_idx,
        })
    df = pd.DataFrame(rows)
    latent = (
        1.6 * df["_fit"]
        + 0.06 * np.minimum(df["years_experience"], 12)
        + 0.25 * df["_edu"]
        - 0.03 * df["employment_gap_months"]
        - 0.025 * np.maximum(df["years_since_graduation"] - 15, 0)
        + 0.15 * df["certifications"]
        + 0.40 * df["referred"]
        + df["postcode_area"].map(POSTCODE_EFFECT)
        + rng.normal(0, 0.5, len(df))
    )
    df["shortlisted"] = (latent > latent.quantile(0.6)).astype(int)
    hire_score = latent + rng.normal(0, 0.4, len(df))
    df["hired"] = ((hire_score > hire_score.quantile(0.85)) & (df["shortlisted"] == 1)).astype(int)
    return df.drop(columns=["_fit", "_edu"])


def generate_interviews(rng: np.random.Generator, n: int = 800) -> pd.DataFrame:
    """Interview transcripts with a binary panel communication rating."""
    rows = []
    for i in range(n):
        quality = rng.beta(2, 2)
        lines = []
        for _ in range(int(rng.integers(5, 10))):
            r = rng.random()
            if r < 0.2:
                lines.append(NEUTRAL_LINES[rng.integers(len(NEUTRAL_LINES))])
            elif r < 0.2 + 0.8 * quality:
                lines.append(STRONG_LINES[rng.integers(len(STRONG_LINES))])
            else:
                lines.append(WEAK_LINES[rng.integers(len(WEAK_LINES))])
        noisy = quality + rng.normal(0, 0.12)
        rows.append({
            "interview_id": f"INT{i:05d}",
            "candidate_id": f"C{int(rng.integers(0, 1200)):05d}",
            "transcript": " ".join(lines),
            "communication_rating": "strong" if noisy > 0.5 else "weak",
        })
    return pd.DataFrame(rows)


def generate_employees(rng: np.random.Generator, n: int = 1500) -> pd.DataFrame:
    """Employee records with attrition and promotion outcomes."""
    n_managers = 60
    leniency = rng.normal(0, 0.45, n_managers)
    df = pd.DataFrame({
        "employee_id": [f"E{i:05d}" for i in range(n)],
        "department": rng.choice(DEPARTMENTS, n),
        "manager_id": rng.integers(0, n_managers, n),
        "age": rng.integers(21, 64, n),
        "tenure_years": np.round(rng.gamma(2.0, 2.2, n), 1),
        "salary_band": rng.integers(1, 7, n),
        "months_since_promotion": rng.integers(0, 72, n),
        "overtime_hours_month": np.round(rng.gamma(2.0, 6.0, n), 1),
        "commute_km": np.round(rng.gamma(2.0, 9.0, n), 1),
        "engagement_score": np.clip(np.round(rng.normal(6.5, 1.8, n), 1), 1, 10),
        "manager_changes_2y": rng.poisson(0.8, n),
        "remote_days_week": rng.integers(0, 6, n),
        "training_hours_year": np.round(rng.gamma(2.0, 10.0, n), 1),
        "parental_leave_last_2y": (rng.random(n) < 0.08).astype(int),
    })
    true_perf = rng.normal(3.2, 0.7, n)
    raw = true_perf + leniency[df["manager_id"]] + rng.normal(0, 0.25, n)
    df["raw_rating"] = np.clip(np.round(raw, 1), 1.0, 5.0)
    df["manager_id"] = df["manager_id"].map(lambda m: f"M{m:03d}")

    logit_leave = (
        -1.4
        - 0.35 * (df["engagement_score"] - 6.5)
        + 0.03 * df["overtime_hours_month"]
        + 0.02 * df["commute_km"]
        + 0.02 * df["months_since_promotion"]
        + 0.35 * df["manager_changes_2y"]
        - 0.15 * df["tenure_years"]
        - 0.20 * df["salary_band"]
        - 0.025 * (df["age"] - 40)
        + 0.60 * df["parental_leave_last_2y"]
    )
    df["left_within_12m"] = (rng.random(n) < 1 / (1 + np.exp(-logit_leave))).astype(int)

    logit_promo = (
        -3.0
        + 1.6 * (true_perf - 3.2)
        + 0.03 * df["months_since_promotion"]
        + 0.02 * df["training_hours_year"]
        + 0.10 * df["tenure_years"]
        - 0.02 * (df["age"] - 40)
        - 0.70 * df["parental_leave_last_2y"]
    )
    df["promoted_next_cycle"] = (rng.random(n) < 1 / (1 + np.exp(-logit_promo))).astype(int)
    return df


def generate_all(seed: int = DEFAULT_SEED) -> Dict[str, pd.DataFrame]:
    """Generate every table from a single seed."""
    rng = np.random.default_rng(seed)
    return {
        "jobs": generate_jobs(),
        "candidates": generate_candidates(rng),
        "interviews": generate_interviews(rng),
        "employees": generate_employees(rng),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in generate_all(args.seed).items():
        path = args.out / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {len(frame):>5} rows -> {path}")


if __name__ == "__main__":
    main()
