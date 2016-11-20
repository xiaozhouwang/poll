'''
train model
'''
from datetime import datetime
from model import LogLossLearner
from data import read_ffm
from evaluation import LogLoss, proba_apk
start = datetime.now()

## data
train = "../data/tr.fm"
train_group_size = "../data/tr_group.csv"
test = "../data/va.fm"
test_group_size = "../data/va_group.csv"

alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized
epoch = 1

D = 2**24            # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

learner = LogLossLearner(alpha, beta, L1, L2, D, interaction)

# start training
#logloss = 0.
map12 = 0.
size = 0
for e in xrange(epoch):
    count_processed_points = 0
    for i, (t, x, y) in enumerate(read_ffm(train, train_group_size)):
        count_processed_points += t
        if i % 100000 == 10:
            print "processing data point %i..." % count_processed_points
            #print "logloss:", logloss / (count_processed_points + size)
            print "map12:", map12 / i
        p = learner.predict(x)
        #logloss += LogLoss(p, y)
        map12 += proba_apk(p, y, 12)
        learner.update(x, p, y)

    #size = count_processed_points * (e + 1) # number of training points
    print('Epoch %d finished, map12: %f, elapsed time: %s' % (
        e, map12 / (i * (e + 1)), str(datetime.now() - start)))
    #print "logloss: ", logloss / size

print "predicting..."
with open('ftrl_va.csv', 'wb') as outfile:
    for t, x, y in read_ffm(test, test_group_size):
        p = learner.predict(x)
        for i in p:
            outfile.write('%s\n' % str(i))
