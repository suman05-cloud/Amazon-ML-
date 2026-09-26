"""Official entity-macro F0.5, including singletons and blocked-out links."""
import numpy as np


def evaluate(truth_counts, pair_groups, labels, probabilities, threshold):
    truth_counts = np.asarray(truth_counts)
    n = len(truth_counts)
    chosen = np.asarray(probabilities) >= threshold
    predicted = np.bincount(np.asarray(pair_groups, dtype=int)[chosen], minlength=n)
    correct = np.bincount(np.asarray(pair_groups, dtype=int)[chosen & (np.asarray(labels) == 1)], minlength=n)
    denominator = .25 * truth_counts + predicted
    score = np.divide(1.25 * correct, denominator, out=np.zeros(n), where=denominator != 0)
    score[(truth_counts == 0) & (predicted == 0)] = 1.
    return {"macro_f0_5": float(score.mean()) if n else 0.,
            "micro_precision": float(correct.sum() / max(1, predicted.sum())),
            "micro_recall": float(correct.sum() / max(1, truth_counts.sum())),
            "singleton_accuracy": float((predicted[truth_counts == 0] == 0).mean()) if np.any(truth_counts == 0) else None,
            "entities": n, "true_links": int(truth_counts.sum()), "predicted_links": int(predicted.sum()),
            "true_positives": int(correct.sum()), "false_positives": int(predicted.sum() - correct.sum()),
            "false_negatives": int(truth_counts.sum() - correct.sum())}


def tune_threshold(truth_counts, groups, labels, probabilities):
    scores = [(evaluate(truth_counts, groups, labels, probabilities, t)["macro_f0_5"], float(t))
              for t in np.linspace(.05, .99, 189)]
    score, threshold = max(scores)
    return threshold, score
