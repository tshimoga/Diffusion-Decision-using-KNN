from scipy.spatial.distance import euclidean
import numpy as np

alpha = 7
distance_step = 0.1
neighbours_needed = []
n_neighbors_for_test = {}
neighbor_limit = 400

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict



euc_dist = {}

def euc(test_point, point):
    test_point_str = test_point
    test_point_str = ''.join(test_point_str.astype(np.str))
    point_str = point
    point_str = '.'.join(point_str.astype(np.str))
    if(test_point_str in euc_dist and point_str in euc_dist[test_point_str]):
        return euc_dist[test_point_str][point_str]
    else:
        if(test_point_str not in euc_dist):
            euc_dist[test_point_str] = {}
        euc_dist[test_point_str][point_str] = euclidean(test_point,point)
        return euc_dist[test_point_str][point_str]

def free_euc(test_point):
    test_point_str = test_point
    test_point_str = ''.join(test_point_str.astype(np.str))
    euc_dist.pop(test_point_str)

def nnn_distance_traning_sample_class(test_point,n, image_class):
    test_point_eucs = []
    class_traning_sample = training_class_sample_map[image_class]
    for training_point in class_traning_sample:
        test_point_eucs.append(euc(test_point,training_point))
    test_point = ''.join(test_point.astype(np.str))
    if(test_point not in n_neighbors_for_test):
        n_neighbors_for_test[test_point] = {}
    if(image_class not in n_neighbors_for_test[test_point]):
            n_neighbors_for_test[test_point][image_class] = []
    for i in range(0,n):
        min_distance = -1
        for test_point_euc in test_point_eucs:
            if(((test_point_euc not in n_neighbors_for_test[test_point][image_class]) and min_distance > test_point_euc) or min_distance == -1):
                min_distance = test_point_euc
        n_neighbors_for_test[test_point][image_class].append(min_distance)
    return n_neighbors_for_test[test_point][image_class][n-1]

def knn(test_point,n,class_one,class_two):
    distance_one = nnn_distance_traning_sample_class(test_point,n,class_one)
    distance_two = nnn_distance_traning_sample_class(test_point,n,class_two)

    if(distance_one < distance_two):
        return class_one
    else:
        return class_two



training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")

correct = 0
wrong = 0
for n in [2,3]:
    for test_key in range(0,10):
        for test_point in test_class_sample_map[test_key]:
            for validation_key in range(0,10):
                if(test_key == validation_key):
                    continue
                expected = knn(test_point,n,test_key,validation_key)
                if(test_key == expected):
                    correct += 1
                else:
                    wrong += 1
            free_euc(test_point=test_point)
        accuracy = (float(correct))/(correct + wrong)

    print("N: " + str(n))
    print("accuracy" + str(accuracy))