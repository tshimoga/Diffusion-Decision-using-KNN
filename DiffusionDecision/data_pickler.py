def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
import numpy as np

file_prefix = "data/data_batch_"
X = np.array([])
y = []
for i in range(1,6):
    file_name = file_prefix + str(i)
    data_batch = unpickle(file_name)
    if(len(X) == 0):
        X = data_batch['data']
    else:
        X = np.vstack((X,data_batch['data']))
    y += data_batch['labels']

# performing LDA
from sklearn.lda import LDA
lda = LDA(n_components=9)
X_transformed = lda.fit(X,y).transform(X)



training_class_sample_map = {}
for i in range(0,len(y)):
    if(y[i] in training_class_sample_map):
        training_class_sample_map[y[i]] = np.vstack((training_class_sample_map[y[i]], X_transformed[i]))
    else:
        training_class_sample_map[y[i]] = X_transformed[i]

# Loading Test data
test_data_batch = unpickle("data/test_batch")
X_test = test_data_batch['data']
y_test = test_data_batch['labels']
X_test_transformed = lda.transform(X_test)


test_class_sample_map = {}
for i in range(0,len(y_test)):
    if(y_test[i] in test_class_sample_map):
        test_class_sample_map[y_test[i]] = np.vstack((test_class_sample_map[y_test[i]], X_test_transformed[i]))
    else:
        test_class_sample_map[y_test[i]] = X_test_transformed[i]

import pickle
training_class_sample_map_dump = open("data/training_class_sample_map", "wb")
test_class_sample_map_dump = open("data/test_class_sample_map", "wb")
pickle.dump(training_class_sample_map, training_class_sample_map_dump)
pickle.dump(test_class_sample_map,test_class_sample_map_dump)