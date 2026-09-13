import numpy as np
from sklearn.metrics import balanced_accuracy_score

def score_fold(true_labels, predicted_labels, in_buffer):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    in_buffer = np.asarray(in_buffer)

    assert len(true_labels) == len(predicted_labels) == len(in_buffer)

    keep = ~in_buffer
    score_full = balanced_accuracy_score(true_labels, predicted_labels)
    score_nonbuffer = balanced_accuracy_score(true_labels[keep], predicted_labels[keep])

    return {
        'balanced_accuracy_full': score_full,
        'balanced_accuracy_nonbuffer': score_nonbuffer,
        'n_full': len(true_labels),
        'n_nonbuffer': int(keep.sum()),
    }