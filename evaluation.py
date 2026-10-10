import numpy as np
from sklearn.metrics import balanced_accuracy_score

# score one set of predictions on both populations
# returns a dictionary with:
# - balanced accurancy for full population
# - balanced accuracy for non-buffer only
# - row counts for full population and non-buffer
def score_fold(true_labels, predicted_labels, in_buffer):

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    in_buffer = np.asarray(in_buffer)

    # true labels, predicted labels and in_buffer needs to be the same
    assert len(true_labels) == len(predicted_labels) == len(in_buffer)

    keep = ~in_buffer
    score_full = balanced_accuracy_score(true_labels, predicted_labels)
    score_nonbuffer = balanced_accuracy_score(true_labels[keep], predicted_labels[keep])

    # a dictionary so that fields can be called by name
    return {
        'balanced_accuracy_full': score_full,
        'balanced_accuracy_nonbuffer': score_nonbuffer,
        'n_full': len(true_labels),
        'n_nonbuffer': int(keep.sum()),
    }