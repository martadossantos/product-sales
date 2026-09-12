import numpy as np
from sklearn.metrics import balanced_accuracy_score

def score_fold(true_labels, predicted_labels, in_buffer):
    return