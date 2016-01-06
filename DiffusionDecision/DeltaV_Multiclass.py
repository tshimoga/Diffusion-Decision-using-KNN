from scipy.spatial.distance import euclidean
import numpy as np
import math

alpha = 3
distance_step = 1
neighbours_step = 1
n_dimensions = 9
n_neighbors_for_test = {}
neighbours_needed = []
gamma = (math.pi ** (float(n_dimensions)/2))/ math.factorial(5)

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
    test_point = ''.join(test_point.astype(np.str))
    euc_dist.pop(test_point)
    n_neighbors_for_test.pop(test_point)

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


def deltaV(test_point,lamda_values,class_one,class_two):
    if(class_one == class_two):
        return class_one

    lamda_one = lamda_values[class_one]
    lamda_two = lamda_values[class_two]
    if(lamda_two > lamda_one):
        class_plus = class_two
        class_minus = class_one
    else:
        class_plus = class_one
        class_minus = class_two
    lamda_plus = lamda_values[class_plus]
    lamda_minus = lamda_values[class_minus]
    confidence_value = math.log(alpha,2)/(lamda_plus - lamda_minus)
    neighbors = 1
    while True:
        distance_one = nnn_distance_traning_sample_class(test_point,neighbors,class_plus)
        volume_one = gamma * (distance_one ** (float(n_dimensions)/2))
        distance_two = nnn_distance_traning_sample_class(test_point,neighbors,class_minus)
        volume_two = gamma * (distance_two ** (float(n_dimensions)/2))

        volume_diff = volume_one - volume_two
        if(volume_diff >= confidence_value):
            neighbours_needed.append(2*neighbors)
            return class_minus
        elif(volume_diff <= (-1 * confidence_value)):
            neighbours_needed.append(2*neighbors)
            return class_plus
        neighbors += neighbours_step



training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")

lamda_values =  unpickle("data/lamda_values")
correct = 0
wrong = 0
for test_key in range(0,10):
    for test_point in test_class_sample_map[test_key]:
        expected = [[0 for x in range(0,10)] for y in range(0,10)]
        for validation_key in range(0,10):
            for validation_key2 in range(0,10):
                if(validation_key2 == validation_key):
                    continue
                pred = deltaV(test_point,lamda_values,validation_key,validation_key2)
                if(pred == validation_key):
                    expected[validation_key][validation_key2] = 1
        counts = [sum(x) for x in expected]
        predicted = counts.index(max(counts))
        if(test_key == predicted):
            correct += 1
        else:
            wrong += 1
        free_euc(test_point=test_point)
    accuracy = (float(correct))/(correct + wrong)

print("alpha" + str(alpha))
print("Neigbors used:" + str(sum(neighbours_needed)/float(len(neighbours_needed))))
print("accuracy" + str(accuracy))
