"""
Shared evaluation protocol used by ALL THREE models (classical ML, CNN, ViT)
so the Phase 7 comparison is apples-to-apples: same folds, same metrics,
same significance test.

Phase 4/5/6 (per-model training) and Phase 7 (comparison) both import from here.
"""

from dataclasses import dataclass


@dataclass
class FoldResult:
    accuracy: float
    f1: float
    precision: float
    recall: float


def run_five_fold_cv(model, X, y, random_seed: int = 42) -> list[FoldResult]:
    """
    Stratified 5-fold CV. Every model (RF/SVM, CNN, ViT) must call this
    with the SAME random_seed so folds are identical across all three,
    per Section 4.3's "same experimental protocol" requirement.
    """
    raise NotImplementedError("Fill in during Phase 4 — implement once, reuse for all 3 models")


def summarize_folds(results: list[FoldResult]) -> dict:
    """Returns mean +/- std for each metric across folds — required reporting format."""
    raise NotImplementedError


def paired_significance_test(results_a: list[FoldResult], results_b: list[FoldResult], metric: str = "accuracy"):
    """
    Paired t-test or Wilcoxon signed-rank between two models' fold results.
    Used in Phase 7, Step 31.
    """
    raise NotImplementedError
