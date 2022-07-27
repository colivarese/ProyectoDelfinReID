import numpy as np

def get_single_accuracy(true_label, labels, diffs, model):
    if model == 'color':
        best_diff_arg = np.array(diffs).argmax()
        best_label = labels[best_diff_arg]
    elif model == 'siamese':
        best_diff_arg = np.array(diffs).argmax()
        best_label = labels[best_diff_arg]
    elif model == 'combination':
        best_diff_arg = np.array(diffs).argmax()
        best_label = labels[best_diff_arg]
    return 1 if true_label == best_label else 0
