'''
train model
'''
from datetime import datetime
from model import poll
from data import read_ffm
start = datetime.now()

## data
train = "cvdata/tr.ffm"
train_group_size = "cvdata/tr.group"
test = "cvdata/te.ffm"
test_group_size = "cvdata/te.group"


alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

epoch = 1

# C, feature/hash trick
D = 2 ** 5             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

learner = poll(alpha, beta, L1, L2, D, interaction)

# start training
for e in xrange(epoch):
    loss = 0.
    count = 0

    for t, x, y in read_ffm(train, D):

        p = learner.predict(x)

        learner.logloss_update(x, p, y)

    print('Epoch %d finished, validation loss: %f, elapsed time: %s' % (
        e, loss/count, str(datetime.now() - start)))
