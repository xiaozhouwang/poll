'''
evaluation functions
'''
from math import log

def logloss(p_list, y_list):
    '''
    return the sum of logloss for the list
    '''
    result = 0
    for  p, y in zip(p_list, y_list):
        p = max(min(p, 1. - 10e-15), 10e-15)
        result += -log(p) if y == 1. else -log(1. - p)
    return result

