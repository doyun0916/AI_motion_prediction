import AReM
import numpy as np
import check_class

dataset = AReM.load_AReM(one_hot_label=True)
train_data = dataset[0][0]
train_target = dataset[0][1]
val_data = dataset[1][0]
val_target = dataset[1][1]

test_data = np.array([[1, 2, 3, 4, 5, 6]])
test_target = np.array([[0, 0, 0, 0, 1, 0]])
test = check_class.Model()
test.update(train_data, train_target)
