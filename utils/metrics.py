"""
Utility functions for computing evaluation metrics.
"""


def calculate_f1_scores(y_true, y_pred):
    """
    Calculate F1-Micro and F1-Macro scores.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    dict
        A dictionary with keys 'f1_micro' and 'f1_macro' containing the
        respective scores as floats.
    """
    # TODO: implement metric calculation using sklearn.metrics.f1_score
    raise NotImplementedError("calculate_f1_scores is not yet implemented.")
