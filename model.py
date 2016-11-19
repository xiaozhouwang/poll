'''
main algorithm
'''
from math import exp, log, sqrt

class poll(object):

    def __init__(self, alpha, beta, L1, L2, D, interaction, loss_func):
        # parameters
        self.loss_func = loss_func
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x_row):
        ''' A helper generator that yields the indices in x per row

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for fea in x_row:
            yield abs(hash(fea)) % self.D

        # now yield interactions (if applicable)
        if self.interaction:
            L = len(x_row)
            x_row = sorted(x_row)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x_row[i]) + '_' + str(x_row[j]))) % self.D

    def predict(self, x_row):

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x_row):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 1e5), -1e-5)))

    def logloss_update(self, x, p, y):
        '''
        x, p, y are lists
        '''
        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g


