
import numpy as np
import pandas as pd

LABELS = ("H", "A", "D", "X")
SCORED = ("H", "A", "D")
H, A, D, X = 0, 1, 2, 3
N_LABELS = len(LABELS)
N_SCORED = len(SCORED)

TAUS = (12, 25)

POSSESSION_LABELS = (H, A)
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
SPELL_IOU_HEADLINE = 0.5
FPS = 25

SPELL_NOISE_BAND = 0.04

BOOTSTRAP_N = 0
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (5, 95)

SHARE_UNDEFINED = 1.0


def read_label_csv(path):
    frame = pd.read_csv(path, dtype={"label": "object"})
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "label" not in frame.columns or "frame" not in frame.columns:
        raise ValueError("expected columns 'frame' and 'label', got {}".format(
            list(frame.columns)))
    return frame


def validate_submission(frame, n_expected):
    if len(frame) != n_expected:
        return "row count {} != {}".format(len(frame), n_expected)

    frames = frame["frame"].values
    if not np.issubdtype(frames.dtype, np.integer):
        return "'frame' column is not integer (dtype {})".format(frames.dtype)
    if not np.array_equal(frames, np.arange(n_expected)):
        return "'frame' column is not 0..{} in order".format(n_expected - 1)

    labels = frame["label"].values
    blank = pd.isna(labels)
    if blank.any():
        return "empty label in {} rows; write X to abstain".format(int(blank.sum()))

    cleaned = np.array([str(v).strip().upper() for v in labels], dtype=object)
    bad = sorted(set(cleaned) - set(LABELS))
    if bad:
        return "labels outside {}: {}".format(list(LABELS), bad[:5])
    return None


def label_codes(frame_or_labels):
    values = frame_or_labels
    if isinstance(values, pd.DataFrame):
        values = values["label"].values
    values = np.array([str(v).strip().upper() for v in values], dtype=object)

    codes = np.full(len(values), -1, dtype=np.int8)
    for index, name in enumerate(LABELS):
        codes[values == name] = index
    if (codes < 0).any():
        raise ValueError("labels outside {}: {}".format(
            list(LABELS), sorted(set(values[codes < 0]))))
    return codes


def confusion(gt_codes, pred_codes):
    pair = gt_codes.astype(np.int64) * N_LABELS + pred_codes.astype(np.int64)
    return np.bincount(pair, minlength=N_LABELS * N_LABELS).reshape(N_LABELS, N_LABELS)


def scored_confusion(cm):
    return cm[:N_SCORED, :]


def per_class_prf(cm3, k):
    true_positive = float(cm3[k, k])
    false_positive = float(cm3[:, k].sum()) - true_positive
    false_negative = float(cm3[k, :].sum()) - true_positive

    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def macro_f1(cm3):
    return float(np.mean([per_class_prf(cm3, k)[2] for k in range(N_SCORED)]))


def accuracy(cm3):
    total = float(cm3.sum())
    if not total:
        return 0.0
    return float(sum(cm3[k, k] for k in range(N_SCORED))) / total


def cohens_kappa(cm3):
    square = np.zeros((N_LABELS, N_LABELS), dtype=np.float64)
    square[:N_SCORED, :] = cm3
    total = square.sum()
    if not total:
        return 0.0

    observed = np.trace(square) / total
    expected = sum(square[k, :].sum() * square[:, k].sum()
                   for k in range(N_LABELS)) / total ** 2
    if expected >= 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def alive_dead_f1(cm3):
    true_positive = float(cm3[:D, :D].sum())
    false_positive = float(cm3[D, :D].sum())
    false_negative = float(cm3[:D, D:].sum())

    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def team_accuracy(cm3):
    correct = float(cm3[H, H] + cm3[A, A])
    gt_alive = float(cm3[:D, :].sum())
    both_alive = float(cm3[:D, :D].sum())
    return (correct / gt_alive if gt_alive else 0.0,
            correct / both_alive if both_alive else 0.0)


def possession_share(cm3):
    gt_alive = float(cm3[:D, :].sum())
    pred_alive = float(cm3[:, :D].sum())
    gt_share = float(cm3[H, :].sum()) / gt_alive if gt_alive else 0.0
    if not pred_alive:
        return gt_share, 0.0, SHARE_UNDEFINED, False
    pred_share = float(cm3[:, H].sum()) / pred_alive
    return gt_share, pred_share, abs(pred_share - gt_share), True


def valid_runs(gt_codes):
    graded = (gt_codes != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def fill_abstentions(pred_codes, runs):
    filled = pred_codes.copy()
    for start, stop in runs:
        window = filled[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[start:stop] = window[carry]
    return filled


def segments(codes, runs):
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]),
                        int(window[bounds[i]])))
    return out


def transitions(codes, runs):
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        for index in (np.flatnonzero(window[1:] != window[:-1]) + 1).tolist():
            out.append((start + int(index), int(window[index - 1]), int(window[index])))
    return out


def match_transitions(gt_trans, pred_trans, tol, typed="to"):
    def key(transition):
        if typed == "pair":
            return (transition[1], transition[2])
        if typed == "none":
            return 0
        return transition[2]

    pairs, missed, spurious = [], [], []
    gt_by_key, pred_by_key = {}, {}
    for transition in gt_trans:
        gt_by_key.setdefault(key(transition), []).append(transition)
    for transition in pred_trans:
        pred_by_key.setdefault(key(transition), []).append(transition)

    for group in sorted(set(gt_by_key) | set(pred_by_key)):
        gts = gt_by_key.get(group, [])
        preds = pred_by_key.get(group, [])
        i = j = 0
        while i < len(gts) and j < len(preds):
            delta = preds[j][0] - gts[i][0]
            if delta < -tol:
                spurious.append(preds[j])
                j += 1
            elif delta > tol:
                missed.append(gts[i])
                i += 1
            else:
                pairs.append((gts[i], preds[j]))
                i += 1
                j += 1
        missed.extend(gts[i:])
        spurious.extend(preds[j:])
    return pairs, missed, spurious


def transition_scores(gt_trans, pred_trans, tol, typed="to"):
    pairs, missed, spurious = match_transitions(gt_trans, pred_trans, tol, typed)
    true_positive, false_positive, false_negative = len(pairs), len(spurious), len(missed)

    precision = true_positive / float(true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / float(true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    latencies = np.array([pred[0] - gt[0] for gt, pred in pairs], dtype=np.float64)
    ambiguous = _count_ambiguous(gt_trans, pred_trans, tol, typed)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": len(gt_trans),
        "n_pred": len(pred_trans),
        "n_matched": true_positive,
        "n_missed": false_negative,
        "n_spurious": false_positive,
        "n_ambiguous": ambiguous,
        "mean_latency": float(latencies.mean()) if len(latencies) else None,
        "median_latency": float(np.median(latencies)) if len(latencies) else None,
        "mean_abs_latency": float(np.abs(latencies).mean()) if len(latencies) else None,
        "p90_abs_latency": float(np.percentile(np.abs(latencies), 90)) if len(latencies) else None,
        "pairs": pairs,
        "missed": missed,
        "spurious": spurious,
    }


def _count_ambiguous(gt_trans, pred_trans, tol, typed):
    def key(transition):
        if typed == "pair":
            return (transition[1], transition[2])
        if typed == "none":
            return 0
        return transition[2]

    by_key = {}
    for transition in pred_trans:
        by_key.setdefault(key(transition), []).append(transition[0])
    for group in by_key:
        by_key[group] = np.array(sorted(by_key[group]))

    count = 0
    for transition in gt_trans:
        frames = by_key.get(key(transition))
        if frames is None:
            continue
        if int(np.sum(np.abs(frames - transition[0]) <= tol)) > 1:
            count += 1
    return count


def fragmentation(gt_codes, pred_codes, runs):
    n_gt = len(segments(gt_codes, runs))
    n_pred = len(segments(pred_codes, runs))
    return n_gt, n_pred, (n_pred / float(n_gt) if n_gt else 0.0)


def possession_spells(codes, runs):
    return [(a, b, k) for a, b, k in segments(codes, runs) if k in POSSESSION_LABELS]


def spell_overlaps(gt_spells, pred_spells):
    out = []
    for i, (gt_start, gt_stop, gt_code) in enumerate(gt_spells):
        for j, (pred_start, pred_stop, pred_code) in enumerate(pred_spells):
            if pred_code != gt_code or pred_stop <= gt_start or gt_stop <= pred_start:
                continue
            intersection = min(gt_stop, pred_stop) - max(gt_start, pred_start)
            union = max(gt_stop, pred_stop) - min(gt_start, pred_start)
            out.append((intersection / float(union), i, j))
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def matched_gt_flags(overlaps, n_gt, threshold):
    flags = np.zeros(n_gt, dtype=bool)
    matched_pred = set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        if flags[gt_index] or pred_index in matched_pred:
            continue
        flags[gt_index] = True
        matched_pred.add(pred_index)
    return flags


def match_spells(overlaps, n_gt, n_pred, threshold):
    true_positive = int(matched_gt_flags(overlaps, n_gt, threshold).sum())
    return true_positive, n_pred - true_positive, n_gt - true_positive


def spell_scores(gt_codes, pred_codes, runs, ious=SPELL_IOUS):
    gt_spells = possession_spells(gt_codes, runs)
    pred_spells = possession_spells(pred_codes, runs)
    overlaps = spell_overlaps(gt_spells, pred_spells)

    per_iou, f1s = {}, []
    for threshold in ious:
        true_positive, false_positive, false_negative = match_spells(
            overlaps, len(gt_spells), len(pred_spells), threshold)
        precision = true_positive / float(true_positive + false_positive) \
            if true_positive + false_positive else 0.0
        recall = true_positive / float(true_positive + false_negative) \
            if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_iou[threshold] = {"precision": precision, "recall": recall, "f1": f1,
                              "n_matched": true_positive, "n_spurious": false_positive,
                              "n_missed": false_negative}
        f1s.append(f1)

    return {
        "per_iou": per_iou,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_gt": len(gt_spells),
        "n_pred": len(pred_spells),
    }


def possession_stats(codes, runs):
    spells = possession_spells(codes, runs)
    lengths = [b - a for a, b, _ in spells]
    alive = int(np.sum((codes == H) | (codes == A)))
    return {
        "n_spells": len(spells),
        "mean_duration_s": float(np.mean(lengths)) / FPS if lengths else 0.0,
        "home_share": float(np.sum(codes == H)) / alive if alive else 0.0,
    }


def bootstrap_macro_f1(gt_codes, pred_codes, runs, n_draws,
                       seed=BOOTSTRAP_SEED, percentiles=BOOTSTRAP_PERCENTILES):
    if not n_draws:
        return None, None

    gt_segments = segments(gt_codes, runs)
    if not gt_segments:
        return None, None

    gt_of_segment = np.array([code for _, _, code in gt_segments], dtype=np.int64)
    contributions = np.zeros((len(gt_segments), N_LABELS), dtype=np.int64)
    for index, (start, stop, _) in enumerate(gt_segments):
        contributions[index] = np.bincount(pred_codes[start:stop].astype(np.int64),
                                           minlength=N_LABELS)

    rng = np.random.RandomState(seed)
    scores = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        picks = rng.randint(0, len(gt_segments), len(gt_segments))
        cm3 = np.zeros((N_SCORED, N_LABELS), dtype=np.int64)
        np.add.at(cm3, gt_of_segment[picks], contributions[picks])
        scores[draw] = macro_f1(cm3)
    low, high = np.percentile(scores, percentiles)
    return float(low), float(high)


def bootstrap_spell_f1(gt_codes, pred_codes, runs, n_draws,
                       seed=BOOTSTRAP_SEED, percentiles=BOOTSTRAP_PERCENTILES):
    if not n_draws:
        return None, None

    gt_spells = possession_spells(gt_codes, runs)
    if not gt_spells:
        return None, None
    pred_spells = possession_spells(pred_codes, runs)
    overlaps = spell_overlaps(gt_spells, pred_spells)
    n_gt, n_pred = len(gt_spells), len(pred_spells)

    flags = np.vstack([matched_gt_flags(overlaps, n_gt, threshold)
                       for threshold in SPELL_IOUS])

    rng = np.random.RandomState(seed)
    scores = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        f1s = []
        for row in flags:
            true_positive = int(row[picks].sum())
            false_negative = n_gt - true_positive
            false_positive = max(0, n_pred - true_positive)
            f1s.append(2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                       if true_positive else 0.0)
        scores[draw] = np.mean(f1s)
    low, high = np.percentile(scores, percentiles)
    return float(low), float(high)


def evaluate(gt_codes, pred_codes, taus=TAUS, bootstrap_n=BOOTSTRAP_N, typed="to"):
    if len(gt_codes) != len(pred_codes):
        raise ValueError("length mismatch: gt {} vs pred {}".format(
            len(gt_codes), len(pred_codes)))

    cm = confusion(gt_codes, pred_codes)
    cm3 = scored_confusion(cm)

    runs = valid_runs(gt_codes)
    filled = fill_abstentions(pred_codes, runs)
    gt_trans = transitions(gt_codes, runs)
    pred_trans = transitions(filled, runs)

    n_gt_seg, n_pred_seg, frag = fragmentation(gt_codes, filled, runs)
    gt_share, pred_share, share_err, share_defined = possession_share(cm3)
    strict_team, both_alive_team = team_accuracy(cm3)
    boot_low, boot_high = bootstrap_macro_f1(gt_codes, pred_codes, runs, bootstrap_n)
    spell_boot_low, spell_boot_high = bootstrap_spell_f1(gt_codes, filled, runs, bootstrap_n)

    spells = spell_scores(gt_codes, filled, runs)
    gt_stats = possession_stats(gt_codes, runs)
    pred_stats = possession_stats(filled, runs)

    result = {
        "macro_f1": macro_f1(cm3),
        "spell_f1_mean": spells["mean_f1"],
        "spell_f1_at_headline": spells["per_iou"][SPELL_IOU_HEADLINE]["f1"],
        "spell_iou_headline": SPELL_IOU_HEADLINE,
        "spell_bootstrap_low": spell_boot_low,
        "spell_bootstrap_high": spell_boot_high,
        "n_gt_spells": spells["n_gt"],
        "n_pred_spells": spells["n_pred"],
        "spell_count_error": spells["n_pred"] - spells["n_gt"],
        "mean_spell_duration_s": pred_stats["mean_duration_s"],
        "gt_mean_spell_duration_s": gt_stats["mean_duration_s"],
        "accuracy": accuracy(cm3),
        "kappa": cohens_kappa(cm3),
        "alive_dead_f1": alive_dead_f1(cm3),
        "team_acc_gt_alive": strict_team,
        "team_acc_both_alive": both_alive_team,
        "gt_home_share": gt_share,
        "pred_home_share": pred_share,
        "share_err": share_err,
        "share_defined": share_defined,
        "n_gt_segments": n_gt_seg,
        "n_pred_segments": n_pred_seg,
        "fragmentation": frag,
        "n_frames": int(len(gt_codes)),
        "n_graded": int(cm3.sum()),
        "n_abstain_on_graded": int(cm3[:, X].sum()),
        "bootstrap_low": boot_low,
        "bootstrap_high": boot_high,
        "bootstrap_n": bootstrap_n,
        "confusion": cm.tolist(),
        "per_class": {},
        "transitions": {},
        "spell_f1": spells["per_iou"],
        "gt_home_share_spells": gt_stats["home_share"],
    }
    for k, name in enumerate(SCORED):
        precision, recall, f1 = per_class_prf(cm3, k)
        result["per_class"][name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(cm3[k, :].sum()),
        }
    for tol in taus:
        result["transitions"][tol] = transition_scores(gt_trans, pred_trans, tol, typed)
    return result
