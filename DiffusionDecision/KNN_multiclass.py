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

# Loading Test data
test_data_batch = unpickle("data/test_batch")
X_test = test_data_batch['data']
y_test = test_data_batch['labels']
X_test_transformed = lda.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1078)
knn.fit(X_transformed,y)
y_pred = knn.predict(X_test_transformed)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))