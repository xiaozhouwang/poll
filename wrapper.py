'''
model wrapper
'''
from datetime import datetime
from model import LogLossLearner
from data import read_ffm
from evaluation import LogLoss, proba_apk

class FTRL(object):

    def __init__(self,alpha = 0.1, beta = 1., L1 = 1., L2 = 1., D = 2**24,
           interaction = False, objective = 'logloss', evaluation = 'map@12', verbose = False):
        '''
        alpha:  learning rate
        beta: smoothing parameter for adaptive learning rate
        L1:  L1 regularization, larger value means more regularized
        L2:  L2 regularization, larger value means more regularized
        D:  number of weights to use
        interaction: whether to enable poly2 feature interactions
        objective: objective function to optimize on (e.g. logloss, rank:pairwise, rank:lambdaRank)
        evalution: evaluation function (e.g. logloss, map@k)
        '''
        if evaluation != 'logloss' and 'map@' not in evaluation:
            raise ValueError("only logloss and map can be used for evaluation function")

        self.evalution = evaluation
        self.evalution_function = LogLoss if evaluation == 'logloss' else proba_apk
        self.verbose = verbose
        self.outside_weights = None
        if objective == 'logloss':
            self.learner = LogLossLearner(alpha, beta, L1, L2, D, interaction)
        else:
            raise ValueError("objective function is invalid.")

    def _get_validation_score(self, validation_path, validation_group_path=None):
        '''
        get validation score for validation data with current model
        '''
        score = 0
        count = 0
        validation_data = read_ffm(validation_path, validation_group_path)
        for i, (t, x, y) in enumerate(validation_data):
            p = self.learner.predict(x)
            score += self.evalution_function(p, y)
            count += t
        return score / count if self.evalution == 'logloss' else score / i

    def train(self, train_path, train_group_path = None, validation_path = None, validation_group_path = None,
              early_stop = False, epoch = 1):
        '''
        train_path: path to train file
        train_group_path: optional. path to train group size file
        validation_path: path to validation file
        validation_group_path: path to validation group path
        early_stop: stop when validation score gets worse
        '''
        start = datetime.now()
        best_score = 100 if self.evalution == "logloss" else 0
        for e in xrange(epoch):
            count_processed_points = 0
            for i, (t, x, y) in enumerate(read_ffm(train_path, train_group_path)):
                count_processed_points += t
                if i % 100000 == 10:
                    if self.verbose:
                        print "training data point %i..." % count_processed_points
                p = self.learner.predict(x)
                self.learner.update(x, p, y)

            print('Epoch %d finished, elapsed time: %s' % (e, str(datetime.now() - start)))

            if validation_path:
                val_score = self._get_validation_score(validation_path, validation_group_path)
                print "validation score: ", val_score
                if early_stop:

                    conditoin = best_score > val_score if self.evalution == "logloss" else best_score < val_score

                    if conditoin:
                        best_score = val_score
                        self.outside_weights = list(self.learner.get_weights())
                    else:
                        print "early stop at epoch %i..." % (e - 1)
                        return {'best_score': best_score, 'best_num_epoch': e}


    def save_prediction(self, output_path, test_path, test_group_path = None):
        '''
        test_path: path to test file
        test_group_path: path to test group file
        '''

        with open(output_path, "wb") as outfile:
            for t, x, y in read_ffm(test_path, test_group_path):
                p = self.learner.predict(x, self.outside_weights)
                for i in p:
                    outfile.write('%s\n' % str(i))




