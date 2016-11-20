from wrapper import FTRL

train = "../data/tr.fm"
train_group_size = "../data/tr_group.csv"
test = "../data/va.fm"
test_group_size = "../data/va_group.csv"
alpha = .1
beta = 1.
L1 = 1.
L2 = 1.
epoch = 1
D = 2**28
interaction = True

clf = FTRL(D = D, interaction=interaction)

train_result = clf.train(train_path=train, train_group_path=train_group_size,
          validation_path=test, validation_group_path=test_group_size, early_stop=True, epoch=10)

print train_result

clf.save_prediction(output_path="../data/ftrl_test.csv", test_path=test, test_group_path=test_group_size)
