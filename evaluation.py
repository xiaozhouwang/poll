'''
evaluation functions
'''
from math import log

def LogLoss(p_list, y_list):
    '''
    return the sum of logloss for the list
    '''
    result = 0
    for p, y in zip(p_list, y_list):
        p = max(min(p, 1. - 10e-15), 10e-15)
        result += -log(p) if y == 1. else -log(1. - p)
    return result


def apk(predicted, actual, k = 12):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def proba_apk(p_list, y_list, k = 12):
    actual = [x for x in xrange(len(y_list)) if y_list[x] == 1]
    predicted = sorted(range(len(p_list)), key = lambda k: p_list[k], reverse=True)
    return apk(predicted, actual, k)