import numpy as np

from possession.vendor import possession_metrics as pm

HEADLINE_KEYS = (
    "spell_f1_mean",
    "macro_f1",
    "n_gt_spells",
    "n_pred_spells",
    "mean_spell_duration_s",
    "gt_mean_spell_duration_s",
    "alive_dead_f1",
    "team_acc_both_alive",
)


def evaluate(reference: np.ndarray, prediction: np.ndarray, bootstrap_n: int = 0) -> dict:
    if reference.shape != prediction.shape:
        raise ValueError(f"shape mismatch {reference.shape} vs {prediction.shape}")
    return pm.evaluate(reference.astype(np.int8), prediction.astype(np.int8), bootstrap_n=bootstrap_n)


def headline(result: dict) -> dict:
    return {key: result[key] for key in HEADLINE_KEYS}
