"""
Evaluation module for AI Resume Screener.
Provides ranking and classification metrics and a demo usage example.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional

import numpy as np
from sklearn import metrics as skm


class ResumeScreeningEvaluator:
    """
    Evaluator for resume screening systems supporting both ranking and classification metrics.

    Conventions:
    - For ranking, inputs are relevance labels (1 for relevant, 0 for not) and predicted scores per query.
    - For classification, inputs are ground-truth labels (0/1) and predicted labels or scores.
    """

    # ---------------------------- Helper utilities ---------------------------- #
    @staticmethod
    def _validate_k(k: int) -> int:
        if k is None or k <= 0:
            raise ValueError("k must be a positive integer")
        return int(k)

    @staticmethod
    def _top_k_indices(scores: Sequence[float], k: int) -> np.ndarray:
        k = min(len(scores), k)
        # argsort descending efficiently
        return np.argpartition(-np.asarray(scores), np.arange(k))[:k][np.argsort(-np.asarray(scores)[:k])]

    # ---------------------------- Ranking metrics ---------------------------- #
    @staticmethod
    def precision_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
        k = ResumeScreeningEvaluator._validate_k(k)
        if len(y_true) == 0:
            return 0.0
        order = np.argsort(-np.asarray(y_score))
        topk = order[: min(k, len(order))]
        rel = np.asarray(y_true)[topk]
        return float(np.sum(rel) / max(1, len(topk)))

    @staticmethod
    def recall_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
        k = ResumeScreeningEvaluator._validate_k(k)
        y_true = np.asarray(y_true)
        if len(y_true) == 0:
            return 0.0
        total_pos = int(np.sum(y_true))
        if total_pos == 0:
            return 0.0
        order = np.argsort(-np.asarray(y_score))
        topk = order[: min(k, len(order))]
        rel = y_true[topk]
        return float(np.sum(rel) / total_pos)

    @staticmethod
    def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
        """
        Average Precision for a single query.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(y_true) == 0:
            return 0.0
        if np.sum(y_true) == 0:
            return 0.0
        # Use sklearn average_precision_score but ensure ranking semantics
        return float(skm.average_precision_score(y_true, y_score))

    @staticmethod
    def mean_average_precision(list_of_y_true: List[Sequence[int]], list_of_y_score: List[Sequence[float]]) -> float:
        aps = [ResumeScreeningEvaluator.average_precision(t, s) for t, s in zip(list_of_y_true, list_of_y_score)]
        if len(aps) == 0:
            return 0.0
        return float(np.mean(aps))

    @staticmethod
    def dcg_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
        k = ResumeScreeningEvaluator._validate_k(k)
        order = np.argsort(-np.asarray(y_score))[:k]
        gains = np.asarray(y_true)[order]
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
        return float(np.sum(gains * discounts))

    @staticmethod
    def ndcg_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
        k = ResumeScreeningEvaluator._validate_k(k)
        ideal_order = np.argsort(-np.asarray(y_true))[:k]
        ideal_gains = np.asarray(y_true)[ideal_order]
        ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2))
        idcg = float(np.sum(ideal_gains * ideal_discounts))
        if idcg == 0.0:
            return 0.0
        dcg = ResumeScreeningEvaluator.dcg_at_k(y_true, y_score, k)
        return float(dcg / idcg)

    @staticmethod
    def mrr(y_true: Sequence[int], y_score: Sequence[float]) -> float:
        order = np.argsort(-np.asarray(y_score))
        rel = np.asarray(y_true)[order]
        hits = np.where(rel > 0)[0]
        if hits.size == 0:
            return 0.0
        return float(1.0 / (hits[0] + 1))

    @staticmethod
    def ranking_metrics(
        y_trues: List[Sequence[int]],
        y_scores: List[Sequence[float]],
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> Dict[str, Any]:
        """
        Compute ranking metrics over multiple queries and a list of k values.
        Returns a dict with per-k Precision@K, Recall@K, NDCG@K, plus MAP and MRR.
        """
        if len(y_trues) != len(y_scores):
            raise ValueError("y_trues and y_scores must have the same length")
        k_values = [ResumeScreeningEvaluator._validate_k(int(k)) for k in k_values]

        results: Dict[str, Any] = {}
        # Per-k aggregates
        for k in k_values:
            precisions, recalls, ndcgs = [], [], []
            for t, s in zip(y_trues, y_scores):
                precisions.append(ResumeScreeningEvaluator.precision_at_k(t, s, k))
                recalls.append(ResumeScreeningEvaluator.recall_at_k(t, s, k))
                ndcgs.append(ResumeScreeningEvaluator.ndcg_at_k(t, s, k))
            results[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
            results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
            results[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0

        # MAP and MRR
        results["map"] = ResumeScreeningEvaluator.mean_average_precision(y_trues, y_scores)
        mrrs = [ResumeScreeningEvaluator.mrr(t, s) for t, s in zip(y_trues, y_scores)]
        results["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
        return results

    # ------------------------- Classification metrics ------------------------ #
    @staticmethod
    def classification_metrics(
        y_true: Sequence[int],
        y_pred: Optional[Sequence[int]] = None,
        y_score: Optional[Sequence[float]] = None,
        pos_label: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute standard classification metrics.
        If y_pred is None and y_score is provided, threshold at 0.5 to get y_pred.
        Returns: accuracy, precision, recall, f1, confusion_matrix, roc_auc (if y_score available)
        """
        y_true_arr = np.asarray(y_true)
        if y_pred is None:
            if y_score is None:
                raise ValueError("Provide y_pred or y_score for classification metrics")
            y_pred_arr = (np.asarray(y_score) >= 0.5).astype(int)
        else:
            y_pred_arr = np.asarray(y_pred)

        acc = float(skm.accuracy_score(y_true_arr, y_pred_arr))
        precision = float(skm.precision_score(y_true_arr, y_pred_arr, zero_division=0, pos_label=pos_label))
        recall = float(skm.recall_score(y_true_arr, y_pred_arr, zero_division=0, pos_label=pos_label))
        f1 = float(skm.f1_score(y_true_arr, y_pred_arr, zero_division=0, pos_label=pos_label))
        cm = skm.confusion_matrix(y_true_arr, y_pred_arr).tolist()

        result: Dict[str, Any] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }
        if y_score is not None:
            # ROC-AUC expects scores; handle cases with a single class gracefully
            try:
                result["roc_auc"] = float(skm.roc_auc_score(y_true_arr, np.asarray(y_score)))
            except ValueError:
                result["roc_auc"] = float("nan")
        else:
            result["roc_auc"] = None
        return result

    # ---------------------- Combined comprehensive evaluation ---------------- #
    @staticmethod
    def comprehensive_evaluation(
        ranking_y_trues: List[Sequence[int]],
        ranking_y_scores: List[Sequence[float]],
        classification_y_true: Optional[Sequence[int]] = None,
        classification_y_pred: Optional[Sequence[int]] = None,
        classification_y_score: Optional[Sequence[float]] = None,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> Dict[str, Any]:
        """
        Compute both ranking and classification metrics in one call.
        Returns a dict with keys: ranking (dict), classification (dict or None)
        """
        results: Dict[str, Any] = {}
        results["ranking"] = ResumeScreeningEvaluator.ranking_metrics(ranking_y_trues, ranking_y_scores, k_values)
        if classification_y_true is not None:
            results["classification"] = ResumeScreeningEvaluator.classification_metrics(
                classification_y_true, classification_y_pred, classification_y_score
            )
        else:
            results["classification"] = None
        return results

    # ------------------------------ Pretty print ----------------------------- #
    @staticmethod
    def _print_evaluation_summary(summary: Dict[str, Any]) -> None:
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        print("=== Ranking Metrics ===")
        for k, v in summary.get("ranking", {}).items():
            print(f"{k}: {fmt(v)}")
        if summary.get("classification") is not None:
            print("\n=== Classification Metrics ===")
            for k, v in summary["classification"].items():
                print(f"{k}: {fmt(v)}")


# ---------------------------------- Demo ---------------------------------- #

def demo_evaluation() -> None:
    """
    Demonstrates the evaluator with small synthetic examples.
    """
    # Ranking example: two job queries each with 5 resumes (1=relevant)
    ranking_y_trues = [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
    ]
    ranking_y_scores = [
        [0.9, 0.6, 0.8, 0.2, 0.1],
        [0.3, 0.7, 0.2, 0.1, 0.8],
    ]

    # Classification example over 8 resumes
    cls_y_true = [1, 0, 1, 0, 0, 1, 0, 1]
    cls_y_score = [0.92, 0.22, 0.76, 0.31, 0.12, 0.84, 0.47, 0.66]
    cls_y_pred = [int(s >= 0.5) for s in cls_y_score]

    evaluator = ResumeScreeningEvaluator()
    summary = evaluator.comprehensive_evaluation(
        ranking_y_trues, ranking_y_scores,
        classification_y_true=cls_y_true,
        classification_y_pred=cls_y_pred,
        classification_y_score=cls_y_score,
        k_values=(1, 3, 5)
    )

    evaluator._print_evaluation_summary(summary)


if __name__ == "__main__":
    demo_evaluation()
