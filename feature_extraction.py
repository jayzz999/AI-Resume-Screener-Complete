# feature_extraction.py

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    """
    Comprehensive feature extraction for resumes and job descriptions.
    Includes:
      - TF-IDF vectorization and cosine similarity
      - Rule-based skill extraction and matching against curated taxonomy (100+ skills)
      - Education level detection
      - Years of experience estimation
      - Composite scoring with explanations
    """

    def __init__(self,
                 max_features: int = 500,
                 ngram_range: Tuple[int, int] = (1, 2),
                 stop_words: str | None = "english") -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            lowercase=True,
            strip_accents="unicode"
        )

        # Curated skills taxonomy (100+ skills across categories)
        self.skills: Dict[str, List[str]] = {
            "programming": [
                "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
                "ruby", "php", "scala", "kotlin", "swift", "matlab", "r", "bash", "shell",
                "powershell", "sql", "nosql", "html", "css", "sass", "less"
            ],
            "ml_ai": [
                "machine learning", "deep learning", "nlp", "natural language processing",
                "computer vision", "recommendation systems", "reinforcement learning",
                "transfer learning", "self-supervised", "foundation models", "llm",
                "large language models", "prompt engineering", "fine-tuning", "rag",
                "retrieval augmented generation", "feature engineering", "model interpretability",
                "explainable ai", "xai", "mlops", "model monitoring", "model deployment",
                "hyperparameter tuning", "bayesian optimization", "cross validation",
                "weak supervision", "active learning", "semi-supervised", "anomaly detection"
            ],
            "data": [
                "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "tensorflow", "keras",
                "pytorch", "mxnet", "jax", "xgboost", "lightgbm", "catboost",
                "matplotlib", "seaborn", "plotly", "altair", "tableau", "power bi",
                "airflow", "dbt", "spark", "pyspark", "hadoop", "hive", "presto",
                "trino", "kafka", "flink", "dask", "ray", "delta lake", "iceberg",
                "duckdb", "polars", "great expectations", "mlflow"
            ],
            "web": [
                "react", "angular", "vue", "next.js", "nuxt", "svelte", "node", "express",
                "fastapi", "flask", "django", "graphql", "rest", "grpc", "webpack",
                "vite", "esbuild", "storybook", "jest", "cypress", "playwright"
            ],
            "cloud": [
                "aws", "azure", "gcp", "google cloud", "amazon web services", "amazon s3",
                "ec2", "lambda", "dynamodb", "sqs", "sns", "eks", "kubernetes", "docker",
                "helm", "terraform", "cloudformation", "databricks", "bigquery", "redshift",
                "snowflake", "athena", "emr", "dataflow", "pubsub", "composer"
            ],
            "tools": [
                "git", "github", "gitlab", "bitbucket", "jira", "confluence", "jenkins",
                "github actions", "circleci", "travis", "pytest", "unittest", "tox",
                "black", "flake8", "pylint", "pre-commit", "make", "poetry", "pip",
                "conda", "virtualenv", "linux", "unix", "macos", "windows",
                "vim", "vscode", "intellij", "pycharm", "notebooks", "jupyter",
                "colab", "latex"
            ],
        }

        # Normalize skills lookup for matching (lowercase variants)
        self.skill_set = {s.lower() for cat in self.skills.values() for s in cat}

        # Education levels mapping to ranks
        self.education_levels: Dict[str, int] = {
            "phd": 4,
            "doctorate": 4,
            "masters": 3,
            "master": 3,
            "ms": 3,
            "msc": 3,
            "m.eng": 3,
            "meng": 3,
            "mba": 3,
            "bachelors": 2,
            "bachelor": 2,
            "bs": 2,
            "bsc": 2,
            "b.eng": 2,
            "beng": 2,
            "associate": 1,
            "diploma": 1
        }

        # Precompile regex patterns
        self.year_pattern = re.compile(r"(\b\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience)?", re.I)
        self.grad_pattern = re.compile(
            r"\b(phd|doctorate|masters?|m\.sc|msc|m\.eng|meng|mba|bachelors?|b\.sc|bsc|b\.eng|beng|associate|diploma)\b",
            re.I
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def extract_skills(self, text: str) -> Dict[str, Any]:
        norm = self._normalize(text)
        found: Dict[str, List[str]] = {}
        for cat, items in self.skills.items():
            hits = []
            for s in items:
                token = s.lower()
                # whole-word or phrase match
                if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", norm):
                    hits.append(token)
            if hits:
                found[cat] = sorted(set(hits))
        flat = sorted({h for v in found.values() for h in v})
        return {"by_category": found, "all": flat}

    def calculate_skill_match(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        resume_sk = self.extract_skills(resume_text)
        job_sk = self.extract_skills(job_text)

        resume_set = set(resume_sk["all"]) if resume_sk else set()
        job_set = set(job_sk["all"]) if job_sk else set()

        if not job_set:
            return {
                "match_ratio": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "job_skills": [],
                "resume_skills": list(sorted(resume_set))
            }

        matched = sorted(resume_set & job_set)
        missing = sorted(job_set - resume_set)
        ratio = len(matched) / max(1, len(job_set))
        return {
            "match_ratio": float(ratio),
            "matched_skills": matched,
            "missing_skills": missing,
            "job_skills": list(sorted(job_set)),
            "resume_skills": list(sorted(resume_set)),
        }

    def calculate_tfidf_similarity(self, resume_text: str, job_text: str) -> float:
        docs = [resume_text or "", job_text or ""]
        X = self.vectorizer.fit_transform(docs)
        if X.shape[0] < 2:
            return 0.0
        sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
        if np.isnan(sim):
            return 0.0
        return float(sim)

    def extract_years_experience(self, text: str) -> float:
        norm = self._normalize(text)
        years = []
        for m in self.year_pattern.finditer(norm):
            try:
                years.append(float(m.group(1)))
            except Exception:
                continue
        # Heuristic: also look for patterns like "X+ years" already captured, return max
        if not years:
            # sometimes resumes include 'over X years'
            more = re.findall(r"over\s+(\d{1,2})\s+years", norm)
            years = [float(x) for x in more] if more else []
        return float(max(years)) if years else 0.0

    def extract_education_level(self, text: str) -> Dict[str, Any]:
        norm = self._normalize(text)
        hits: List[str] = []
        for m in self.grad_pattern.finditer(norm):
            hits.append(m.group(1).lower().replace(".", ""))
        rank = 0
        level = None
        for h in set(hits):
            r = self.education_levels.get(h, 0)
            if r > rank:
                rank = r
                level = h
        return {"level": level, "rank": int(rank), "mentions": sorted(set(hits))}

    def calculate_rule_based_score(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        skills = self.calculate_skill_match(resume_text, job_text)
        edu = self.extract_education_level(resume_text)
        exp_resume = self.extract_years_experience(resume_text)
        exp_job = self.extract_years_experience(job_text)

        # Experience match score: full if resume >= job requirement, else proportional
        if exp_job > 0:
            exp_ratio = min(1.0, exp_resume / exp_job)
        else:
            exp_ratio = 1.0 if exp_resume > 0 else 0.5

        # Education score: bonus for higher levels
        edu_score = min(1.0, edu.get("rank", 0) / 4.0)

        return {
            "skill_match": skills,
            "education": edu,
            "experience": {
                "resume_years": float(exp_resume),
                "job_years": float(exp_job),
                "ratio": float(exp_ratio)
            },
            "edu_score": float(edu_score)
        }

    def extract_all_features(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        tfidf_sim = self.calculate_tfidf_similarity(resume_text, job_text)
        rule = self.calculate_rule_based_score(resume_text, job_text)
        return {
            "tfidf_similarity": float(tfidf_sim),
            "rule_based": rule
        }

    def calculate_overall_score(self, resume_text: str, job_text: str,
                                weights: Dict[str, float] | None = None) -> Dict[str, Any]:
        """
        Compute a composite score.
        Default weights:
          - TF-IDF similarity: 0.45
          - Skill match ratio: 0.35
          - Experience ratio: 0.15
          - Education score: 0.05
        """
        feats = self.extract_all_features(resume_text, job_text)
        rules = feats["rule_based"]
        skill_ratio = rules["skill_match"]["match_ratio"]
        exp_ratio = rules["experience"]["ratio"]
        edu_score = rules["edu_score"]
        tfidf_sim = feats["tfidf_similarity"]

        w = {
            "tfidf": 0.45,
            "skills": 0.35,
            "experience": 0.15,
            "education": 0.05,
        }
        if weights:
            w.update(weights)

        overall = (
            w["tfidf"] * tfidf_sim +
            w["skills"] * skill_ratio +
            w["experience"] * exp_ratio +
            w["education"] * edu_score
        )
        return {
            "overall_score": float(overall),
            "weights": w,
            "features": feats
        }

    def generate_explanation(self, resume_text: str, job_text: str,
                              weights: Dict[str, float] | None = None) -> Dict[str, Any]:
        res = self.calculate_overall_score(resume_text, job_text, weights)
        feats = res["features"]
        rules = feats["rule_based"]
        skills = rules["skill_match"]
        exp = rules["experience"]

        explanation = {
            "summary": {
                "overall_score": res["overall_score"],
                "tfidf_similarity": feats["tfidf_similarity"],
                "skill_match_ratio": skills["match_ratio"],
                "years_experience": exp["resume_years"],
                "required_experience": exp["job_years"],
                "education_level": rules["education"]["level"],
            },
            "details": {
                "matched_skills": skills["matched_skills"],
                "missing_skills": skills["missing_skills"],
                "job_skills": skills["job_skills"],
                "resume_skills": skills["resume_skills"],
                "education_mentions": rules["education"]["mentions"],
            }
        }
        return explanation
